"""
training/diagnostics.py — Per-epoch diagnostic suite for CTC sequence models.

Why
---
The Run-8 post-mortem identified that CER alone is too coarse. The model
exhibited multiple distinct pathologies (mode collapse to short n-grams,
blank attractor, train-val NLL gap, prediction-length compression) that a
single CER number masks.  This module computes the full suite of signals
recommended in the iterative-ablation plan:

  * length-bucketed CER  (1-4, 5-8, 9-12, 13+)
  * train/val NLL (CTC loss on val, separate from CER)
  * mean_pred_len, mean_target_len, length ratio
  * mean blank probability over T (averaged across B)
  * mean entropy of posterior over T (averaged across B)
  * per-character marginal of predictions and labels
  * KL(pred_marginal || label_marginal)
  * insertion / deletion / substitution decomposition
  * argmax run-length histogram (consecutive same-symbol emissions)
  * per-parameter-group gradient L2 norm (call grad_norms() during training)

All functions are stateless / pure where possible.  Call signature is
designed to be drop-in inside the trainer's val loop.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import math
import logging
import torch
import torch.nn.functional as F
import numpy as np
import editdistance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Length-bucketed CER
# ---------------------------------------------------------------------------

LENGTH_BUCKETS: list[tuple[int, int, str]] = [
    (1,  4,  "1-4"),
    (5,  8,  "5-8"),
    (9,  12, "9-12"),
    (13, 999, "13+"),
]


def length_bucketed_cer(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """
    Compute mean CER per ground-truth length bucket.

    Parameters
    ----------
    pairs : list of (gt_str, pred_str) tuples

    Returns
    -------
    dict mapping bucket label to {"cer", "n_clips", "mean_target_len",
                                  "mean_pred_len"}.
    """
    buckets: dict[str, dict[str, list]] = {
        name: {"errs": [], "lens": [], "pred_lens": [], "target_lens": []}
        for _, _, name in LENGTH_BUCKETS
    }
    for gt, pred in pairs:
        L = len(gt)
        for lo, hi, name in LENGTH_BUCKETS:
            if lo <= L <= hi:
                err = editdistance.eval(gt, pred)
                buckets[name]["errs"].append(err)
                buckets[name]["lens"].append(L)
                buckets[name]["pred_lens"].append(len(pred))
                buckets[name]["target_lens"].append(L)
                break

    out: dict[str, dict[str, float]] = {}
    for name in buckets:
        b = buckets[name]
        n = len(b["errs"])
        if n == 0:
            out[name] = {"cer": float("nan"), "n_clips": 0,
                         "mean_target_len": 0.0, "mean_pred_len": 0.0}
            continue
        total_err = sum(b["errs"])
        total_len = sum(b["target_lens"])
        out[name] = {
            "cer":             total_err / max(total_len, 1),
            "n_clips":         n,
            "mean_target_len": sum(b["target_lens"]) / n,
            "mean_pred_len":   sum(b["pred_lens"]) / n,
        }
    return out


# ---------------------------------------------------------------------------
# Edit-distance ins/del/sub decomposition
# ---------------------------------------------------------------------------

def edit_decomposition(gt: str, pred: str) -> tuple[int, int, int]:
    """
    Wagner-Fischer DP to split edit distance into (ins, del, sub).
    Returns counts of insertions, deletions, substitutions.
    """
    m, n = len(gt), len(pred)
    # dp[i][j] = (ins, del, sub) tuple for transforming gt[:i] to pred[:j]
    dp = [[(0, 0, 0)] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = (0, i, 0)     # i deletions
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, 0)     # j insertions

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                ins  = dp[i][j - 1]
                del_ = dp[i - 1][j]
                sub  = dp[i - 1][j - 1]
                # cost = sum of each tuple's counts + 1 for current op
                cost_ins  = sum(ins)  + 1
                cost_del  = sum(del_) + 1
                cost_sub  = sum(sub)  + 1
                if cost_sub <= cost_ins and cost_sub <= cost_del:
                    dp[i][j] = (sub[0], sub[1], sub[2] + 1)
                elif cost_ins <= cost_del:
                    dp[i][j] = (ins[0] + 1, ins[1], ins[2])
                else:
                    dp[i][j] = (del_[0], del_[1] + 1, del_[2])
    return dp[m][n]


def aggregate_edit_decomposition(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, float]:
    """
    Aggregate ins/del/sub over the validation set, normalised by total
    target characters.
    """
    n_ins = n_del = n_sub = n_chars = 0
    for gt, pred in pairs:
        ins, del_, sub = edit_decomposition(gt, pred)
        n_ins  += ins
        n_del  += del_
        n_sub  += sub
        n_chars += len(gt)
    n_chars = max(n_chars, 1)
    return {
        "ins_rate":   n_ins / n_chars,
        "del_rate":   n_del / n_chars,
        "sub_rate":   n_sub / n_chars,
        "total_err":  n_ins + n_del + n_sub,
        "total_char": n_chars,
    }


# ---------------------------------------------------------------------------
# Posterior diagnostics  (run on the head's log-probs over the val set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def posterior_diagnostics(
    log_probs: torch.Tensor,
    lengths:   torch.Tensor,
    blank:     int = 0,
) -> dict[str, float]:
    """
    Compute blank probability and entropy averaged over valid timesteps.

    Parameters
    ----------
    log_probs : [B, T, V]  CTC log-probs over a batch
    lengths   : [B] int — valid T per sample
    blank     : blank-token index

    Returns dict with mean_blank_prob, mean_entropy_nats.
    """
    B, T, V = log_probs.shape
    probs = log_probs.exp()
    # Build a [B, T] mask of valid positions
    arange = torch.arange(T, device=lengths.device).unsqueeze(0)   # [1, T]
    mask = arange < lengths.unsqueeze(1)                            # [B, T]
    mask_f = mask.float()

    # Mean blank prob over valid positions
    blank_probs = probs[..., blank]                                 # [B, T]
    n_valid = mask_f.sum().clamp(min=1.0)
    mean_blank = (blank_probs * mask_f).sum() / n_valid

    # Entropy = -sum p * log p over the vocab axis
    safe_log_probs = log_probs.clamp(min=-30.0)
    entropy = -(probs * safe_log_probs).sum(dim=-1)                 # [B, T]
    mean_entropy = (entropy * mask_f).sum() / n_valid

    return {
        "mean_blank_prob":   float(mean_blank.item()),
        "mean_entropy_nats": float(mean_entropy.item()),
        "max_entropy_nats":  float(math.log(V)),
    }


# ---------------------------------------------------------------------------
# Argmax run-length and character marginals
# ---------------------------------------------------------------------------

def argmax_run_lengths(
    argmax_seqs: list[list[int]],
) -> dict[str, float]:
    """
    Histogram of consecutive same-symbol emissions (run lengths) in the
    pre-CTC-collapse argmax sequences.  Long runs indicate stuck symbols
    or excessive blanking.
    """
    counts: Counter[int] = Counter()
    for seq in argmax_seqs:
        if not seq:
            continue
        run = 1
        prev = seq[0]
        for t in seq[1:]:
            if t == prev:
                run += 1
            else:
                counts[run] += 1
                run = 1
                prev = t
        counts[run] += 1

    if not counts:
        return {"mean_run_len": 0.0, "max_run_len": 0.0, "p_run_ge_5": 0.0}
    total = sum(counts.values())
    weighted = sum(L * n for L, n in counts.items())
    p_long = sum(n for L, n in counts.items() if L >= 5) / total
    return {
        "mean_run_len": weighted / total,
        "max_run_len":  float(max(counts.keys())),
        "p_run_ge_5":   p_long,
    }


def character_marginal(
    strings: Sequence[str],
    chars:   str,
) -> np.ndarray:
    """
    Per-character marginal probability over the given strings.
    Returns a vector of size len(chars).
    """
    counts = np.zeros(len(chars), dtype=np.float64)
    for s in strings:
        for c in s:
            i = chars.find(c)
            if i >= 0:
                counts[i] += 1
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) in nats. p and q are 1-D probability vectors."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum())


# ---------------------------------------------------------------------------
# Gradient norms per parameter group
# ---------------------------------------------------------------------------

def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    L2 norm of the gradient for each named parameter group.  Useful for
    seeing whether the temporal model, projection, or upsampler is doing
    the heavy lifting after backprop.

    Returns
    -------
    {name: l2_norm}  where name is the top-level module name
    """
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = name.split(".")[0]
        out[group] = out.get(group, 0.0) + float(p.grad.detach().pow(2).sum().item())
    for k in out:
        out[k] = math.sqrt(out[k])
    return out


# ---------------------------------------------------------------------------
# CTC feasibility assertion
# ---------------------------------------------------------------------------

def assert_ctc_feasible(
    input_lengths:  torch.Tensor,
    target_lengths: torch.Tensor,
    raise_on_fail:  bool = True,
) -> dict[str, int]:
    """
    Check that T_out_i ≥ 2 * L_i + 1 for every sample in a batch.

    Returns counts of feasible / infeasible samples.  If raise_on_fail and
    any sample fails, raises ValueError so the bad config surfaces loudly.
    """
    required = 2 * target_lengths + 1
    feasible_mask = input_lengths >= required
    n_total       = int(target_lengths.numel())
    n_feasible    = int(feasible_mask.sum().item())
    n_infeasible  = n_total - n_feasible
    if n_infeasible > 0 and raise_on_fail:
        bad = torch.nonzero(~feasible_mask).flatten().tolist()
        examples = [
            (int(input_lengths[i].item()), int(target_lengths[i].item()),
             int(required[i].item()))
            for i in bad[:5]
        ]
        raise ValueError(
            f"CTC feasibility violated for {n_infeasible}/{n_total} samples. "
            f"Need T_out ≥ 2L+1.  Examples (T_out, L, required): {examples}. "
            "Increase upsample factor or shorten labels."
        )
    return {"feasible": n_feasible, "infeasible": n_infeasible, "total": n_total}


# ---------------------------------------------------------------------------
# One-shot snapshot
# ---------------------------------------------------------------------------

def full_diagnostic_snapshot(
    pairs:                 Sequence[tuple[str, str]],
    log_probs:             Optional[torch.Tensor] = None,
    lengths:               Optional[torch.Tensor] = None,
    chars:                 str = "abcdefghijklmnopqrstuvwxyz",
    train_loss:            Optional[float] = None,
    val_loss:              Optional[float] = None,
    blank:                 int = 0,
) -> dict[str, object]:
    """
    Compute the entire diagnostic suite in one call.  Returns a flat dict
    suitable for json.dump.  Pass log_probs/lengths for posterior diagnostics
    (sample a batch from the val loader to do this once per epoch).
    """
    snap: dict[str, object] = {}

    # CER
    total_err = sum(editdistance.eval(gt, pred) for gt, pred in pairs)
    total_len = sum(len(gt) for gt, pred in pairs)
    snap["val_cer_overall"] = total_err / max(total_len, 1)

    # Length-bucketed
    snap["length_buckets"] = length_bucketed_cer(pairs)

    # Length ratios
    if pairs:
        mean_pred   = sum(len(p) for _, p in pairs) / len(pairs)
        mean_target = sum(len(g) for g, _ in pairs) / len(pairs)
        snap["mean_pred_len"]   = mean_pred
        snap["mean_target_len"] = mean_target
        snap["len_ratio"]       = mean_pred / max(mean_target, 1e-9)

    # ins/del/sub
    snap["edit_decomposition"] = aggregate_edit_decomposition(pairs)

    # Character marginals + KL
    pred_marg  = character_marginal([p for _, p in pairs], chars)
    label_marg = character_marginal([g for g, _ in pairs], chars)
    snap["pred_char_marginal"]    = pred_marg.tolist()
    snap["label_char_marginal"]   = label_marg.tolist()
    snap["kl_pred_vs_label_nats"] = kl_divergence(pred_marg, label_marg)

    # NLL gap
    if train_loss is not None:
        snap["train_ctc_loss"] = float(train_loss)
    if val_loss is not None:
        snap["val_ctc_loss"]   = float(val_loss)
    if train_loss is not None and val_loss is not None:
        snap["nll_gap"] = float(val_loss) - float(train_loss)

    # Posterior diagnostics
    if log_probs is not None and lengths is not None:
        snap.update(posterior_diagnostics(log_probs, lengths, blank=blank))

    return snap


def format_snapshot_line(snap: dict, prefix: str = "") -> str:
    """One-line summary of a diagnostic snapshot for log printing."""
    lb = snap.get("length_buckets", {})
    parts = [
        f"{prefix}cer={snap.get('val_cer_overall', float('nan')):.4f}",
        f"L1-4={lb.get('1-4', {}).get('cer', float('nan')):.3f}",
        f"L5-8={lb.get('5-8', {}).get('cer', float('nan')):.3f}",
        f"L9-12={lb.get('9-12', {}).get('cer', float('nan')):.3f}",
        f"L13+={lb.get('13+', {}).get('cer', float('nan')):.3f}",
        f"len_ratio={snap.get('len_ratio', float('nan')):.2f}",
        f"blank_p={snap.get('mean_blank_prob', float('nan')):.3f}",
        f"H={snap.get('mean_entropy_nats', float('nan')):.2f}",
        f"KL={snap.get('kl_pred_vs_label_nats', float('nan')):.3f}",
    ]
    if "nll_gap" in snap:
        parts.append(f"gap={snap['nll_gap']:+.3f}")
    return " | ".join(parts)

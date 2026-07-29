"""
training/stage11_train.py — Stage 11 paper-comparable headline run.

Hyperparameters are FROZEN from the Stage 9a-best ablation winners:
  * Joint CTC + attention decoder
  * lambda_ctc = 0.5
  * Attention decoder: 2 layers, d=256, h=4
  * Conformer encoder: 4 layers, d=256, h=4, kernel=15, dropout 0.2
  * AdamW lr=5e-4 wd=5e-2, OneCycleLR warmup 5%, batch 32, 80 epochs
  * Seed = 42

The novelty here is purely the DATA: full 122-signer dataset under the
paper's exact 8:1:1 train/val/test split.  We do not redo the
architecture search.

Test-set discipline: this trainer NEVER touches test_loader during
training.  Test evaluation is a separate function (`final_test_eval`)
that the orchestrator calls exactly once, after training, using the
best-val checkpoint.
"""

from __future__ import annotations

import os
import json
import time
import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import editdistance

from ..models.conformer_ctc     import ConformerCTC
from ..models.attention_decoder import AttentionDecoder, build_attention_targets
from ..datasets.vocab           import make_converter
from ..datasets.landmark_paper_split import WiTAPaperSplitDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _collate(batch, pad_idx: int):
    """
    Stage 11 collate.  Yields metadata (signer, subset, label_str)
    alongside tensors so the val/test eval can do per-{signer, subset,
    length-bucket} aggregations.
    """
    feats, labels, signers, subsets, label_strs = zip(*batch)
    feats_pad  = pad_sequence(feats,  batch_first=True, padding_value=0.0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.LongTensor([f.shape[0] for f in feats])
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return (feats_pad, labels_pad, input_lens, label_lens,
            list(signers), list(subsets), list(label_strs))


# ---------------------------------------------------------------------------
# Greedy decode helpers (shared with Stage 9a)
# ---------------------------------------------------------------------------

def _ctc_greedy(log_probs: torch.Tensor, enc_lens: torch.Tensor, blank: int):
    out = []
    argmax = log_probs.argmax(dim=-1)
    for b in range(argmax.shape[0]):
        seq = argmax[b, : int(enc_lens[b].item())].tolist()
        merged = []; prev = None
        for t in seq:
            if t != prev and t != blank:
                merged.append(t)
            prev = t
        out.append(merged)
    return out


def _ids_to_str(ids, chars):
    return "".join(chars[t-1] if 1 <= t <= len(chars) else "?" for t in ids)


# ---------------------------------------------------------------------------
# Eval-side aggregation (per-subset, per-signer, length-bucketed)
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 999)]


def _length_bucket(L: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= L <= hi:
            return f"{lo}-{hi if hi < 999 else 'inf'}"
    return "unknown"


@torch.no_grad()
def evaluate_loader(
    loader,
    *,
    encoder:   ConformerCTC,
    decoder:   AttentionDecoder,
    cfg,
    blank:     int,
    sos:       int,
    eos:       int,
    pad:       int,
    device:    str,
):
    """
    Run greedy CTC + greedy attention decode on the loader.  Returns a
    dict with aggregate + per-subset + per-signer + length-bucketed CERs,
    and the raw per-clip predictions for downstream analysis.
    """
    encoder.eval(); decoder.eval()
    pairs_all:  list[dict] = []          # one dict per clip
    sum_val_ctc = 0.0; n_val_batches = 0
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    for feats, labels, in_lens, lab_lens, signers, subsets, label_strs in loader:
        feats   = feats.to(device);   labels   = labels.to(device)
        in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)
        h, pad_mask = encoder.encode(feats, in_lens)
        log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
        v_ctc = ctc_loss_fn(log_probs.transpose(0, 1).float(),
                            labels, enc_lens, lab_lens)
        sum_val_ctc += float(v_ctc.item()); n_val_batches += 1

        ctc_preds  = _ctc_greedy(log_probs, enc_lens, blank)
        attn_preds = decoder.greedy_decode(h, pad_mask)
        for b in range(len(ctc_preds)):
            gt = label_strs[b]
            p_ctc  = _ids_to_str(ctc_preds[b],            cfg.vocab.chars)
            p_attn = _ids_to_str(attn_preds[b].tolist(),  cfg.vocab.chars)
            e_ctc  = editdistance.eval(gt, p_ctc)
            e_attn = editdistance.eval(gt, p_attn)
            e_best = min(e_ctc, e_attn)
            pred_best = p_ctc if e_ctc <= e_attn else p_attn
            pairs_all.append({
                "gt":     gt,
                "pred":   pred_best,
                "edit":   int(e_best),
                "L":      int(len(gt)),
                "signer": signers[b],
                "subset": subsets[b],
                "ctc":    p_ctc,
                "attn":   p_attn,
                "e_ctc":  int(e_ctc),
                "e_attn": int(e_attn),
            })

    val_ctc = sum_val_ctc / max(n_val_batches, 1)

    # Aggregate.
    def _cer(rows):
        n = sum(r["edit"] for r in rows)
        d = sum(r["L"]    for r in rows)
        return n / max(d, 1)

    overall_cer = _cer(pairs_all)
    per_subset: dict[str, float] = {}
    per_signer: dict[str, float] = {}
    per_bucket: dict[str, float] = {}
    by_subset: dict[str, list]  = defaultdict(list)
    by_signer: dict[str, list]  = defaultdict(list)
    by_bucket: dict[str, list]  = defaultdict(list)
    for r in pairs_all:
        by_subset[r["subset"]].append(r)
        by_signer[r["signer"]].append(r)
        by_bucket[_length_bucket(r["L"])].append(r)
    for k, v in by_subset.items(): per_subset[k] = _cer(v)
    for k, v in by_signer.items(): per_signer[k] = _cer(v)
    for k, v in by_bucket.items(): per_bucket[k] = _cer(v)

    return {
        "val_ctc_loss":      val_ctc,
        "overall_cer":       overall_cer,
        "per_subset_cer":    per_subset,
        "per_signer_cer":    per_signer,
        "per_length_cer":    per_bucket,
        "n_clips":           len(pairs_all),
        "pairs":             pairs_all,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_stage11(
    cache_root:     str,
    *,
    cfg,
    num_epochs:     int = 80,
    batch_size:     int = 32,
    lr_peak:        float = 5e-4,
    weight_decay:   float = 5e-2,
    grad_clip:      float = 1.0,
    dropout:        float = 0.2,
    d_model:        int = 256,
    n_layers:       int = 4,
    n_heads:        int = 4,
    conv_kernel:    int = 15,
    upsample:       int = 2,
    warmup_pct:     float = 0.05,
    dec_n_layers:   int = 2,            # Stage 9a ablation winner
    dec_n_heads:    int = 4,
    lambda_ctc:     float = 0.5,        # Stage 9a ablation winner
    label_smoothing: float = 0.1,
    transform=None,
    seed:           int = 42,
    checkpoint_dir: str = "/kaggle/working/checkpoints",
    log_dir:        str = "/kaggle/working/logs",
    variant:        str = "stage11",
) -> dict:
    """Single-seed Stage 11 training run.  Returns the result dict."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device     = cfg.device
    converter  = make_converter(cfg.data.lang)
    pad_idx    = cfg.vocab.pad_idx
    blank      = cfg.vocab.blank_idx
    att_V      = cfg.vocab.attn_vocab_size
    sos        = cfg.vocab.sos_idx
    eos        = cfg.vocab.eos_idx
    ctc_V      = cfg.vocab.ctc_vocab_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # ---- data ----
    train_ds = WiTAPaperSplitDataset(
        cache_root, "train", subsets=("lex", "nonlex"),
        converter=converter, transform=transform,
    )
    val_ds = WiTAPaperSplitDataset(
        cache_root, "val", subsets=("lex", "nonlex"),
        converter=converter, transform=None,
    )
    coll = lambda b: _collate(b, pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=coll, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=coll,
    )
    print(f"[stage11] train={len(train_ds)}  val={len(val_ds)}  "
          f"signers train/val = {len(train_ds.signers)}/{len(val_ds.signers)}",
          flush=True)
    print(f"[stage11] train subset counts: {train_ds.per_subset_counts()}",
          flush=True)
    print(f"[stage11] val   subset counts: {val_ds.per_subset_counts()}",
          flush=True)

    # ---- model ----
    encoder = ConformerCTC(
        input_dim       = 190,
        vocab_size      = ctc_V,
        d_model         = d_model,
        n_layers        = n_layers,
        n_heads         = n_heads,
        conv_kernel     = conv_kernel,
        dropout         = dropout,
        upsample        = upsample,
        input_layernorm = False,         # landmarks, Stage 1 v2 contract
    ).to(device)
    decoder = AttentionDecoder(
        att_vocab_size = att_V, bos_idx = sos, eos_idx = eos,
        d_model        = d_model, n_layers = dec_n_layers, n_heads = dec_n_heads,
        ff_mult        = 4, dropout = dropout,
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=lr_peak, weight_decay=weight_decay, betas=(0.9, 0.999),
    )
    total_steps = num_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_peak, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    ce  = nn.CrossEntropyLoss(ignore_index=pad_idx,
                              label_smoothing=label_smoothing)

    history: list[dict] = []
    best_overall = float("inf")
    best_epoch   = -1
    best_payload: dict = {}
    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pt")

    print(
        f"\n=== Stage 11 training ===  variant={variant}  "
        f"lambda_ctc={lambda_ctc}  dec_n_layers={dec_n_layers}  "
        f"epochs={num_epochs}  seed={seed}",
        flush=True,
    )

    for epoch in range(num_epochs):
        encoder.train(); decoder.train()
        sum_ctc = sum_attn = sum_total = 0.0
        n_batches = 0
        t0 = time.time()

        for feats, labels, in_lens, lab_lens, _, _, _ in train_loader:
            feats   = feats.to(device);   labels   = labels.to(device)
            in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

            h, pad_mask = encoder.encode(feats, in_lens)
            log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
            ctc_loss = ctc(log_probs.transpose(0, 1).float(),
                           labels, enc_lens, lab_lens)
            dec_in, dec_tg = build_attention_targets(
                labels, lab_lens, bos=sos, eos=eos, pad=pad_idx,
            )
            dec_logits = decoder(h, pad_mask, dec_in)
            attn_loss  = ce(dec_logits.reshape(-1, decoder.att_vocab_size),
                            dec_tg.reshape(-1))
            total = lambda_ctc * ctc_loss + (1 - lambda_ctc) * attn_loss
            optimizer.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step(); scheduler.step()
            sum_ctc += float(ctc_loss.item())
            sum_attn += float(attn_loss.item())
            sum_total += float(total.item())
            n_batches += 1
        train_ctc   = sum_ctc   / max(n_batches, 1)
        train_attn  = sum_attn  / max(n_batches, 1)
        train_total = sum_total / max(n_batches, 1)

        # ---- val evaluation ----
        val_out = evaluate_loader(
            val_loader, encoder=encoder, decoder=decoder, cfg=cfg,
            blank=blank, sos=sos, eos=eos, pad=pad_idx, device=device,
        )
        val_overall = val_out["overall_cer"]
        val_lex     = val_out["per_subset_cer"].get("lex",    float("nan"))
        val_non     = val_out["per_subset_cer"].get("nonlex", float("nan"))

        dt = time.time() - t0
        print(
            f"[E{epoch+1:3d}/{num_epochs}] "
            f"ctc={train_ctc:.4f}  attn={train_attn:.4f}  total={train_total:.4f}  "
            f"val_ctc={val_out['val_ctc_loss']:.4f}  "
            f"val CER overall={val_overall:.4f}  lex={val_lex:.4f}  "
            f"nonlex={val_non:.4f}  {dt:.0f}s",
            flush=True,
        )
        # Strip the bulky per-clip pairs out of history.
        snap = {k: v for k, v in val_out.items() if k != "pairs"}
        history.append({
            "epoch":           epoch + 1,
            "train_ctc":       train_ctc,
            "train_attn":      train_attn,
            "train_total":     train_total,
            **snap,
        })

        # Best on val_overall_cer ONLY.
        if val_overall < best_overall:
            best_overall = val_overall
            best_epoch   = epoch + 1
            best_payload = {
                "val_overall_cer":  val_overall,
                "val_lex_cer":      val_lex,
                "val_nonlex_cer":   val_non,
                "val_per_signer":   dict(val_out["per_signer_cer"]),
                "val_per_length":   dict(val_out["per_length_cer"]),
                "epoch":            epoch + 1,
            }
            torch.save({
                "variant":            variant,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch":              epoch,
                "best_payload":       best_payload,
                "lambda_ctc":         lambda_ctc,
                "dec_n_layers":       dec_n_layers,
                "seed":               seed,
            }, ckpt_path)
            print(f"    ★ new best val_overall_cer={val_overall:.4f}", flush=True)

    result = {
        "variant":          variant,
        "num_epochs":       num_epochs,
        "n_train_clips":    len(train_ds),
        "n_val_clips":      len(val_ds),
        "seed":             seed,
        "lambda_ctc":       lambda_ctc,
        "dec_n_layers":     dec_n_layers,
        "best_val_overall": best_overall,
        "best_epoch":       best_epoch,
        "best_payload":     best_payload,
        "checkpoint_path":  ckpt_path,
        "history":          history,
    }
    json_path = os.path.join(log_dir, f"{variant}_training.json")
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)
    return result


# ---------------------------------------------------------------------------
# Test-set evaluation — called EXACTLY ONCE by the orchestrator
# ---------------------------------------------------------------------------

def final_test_eval(
    cache_root:    str,
    checkpoint:    str,
    *,
    cfg,
    batch_size:    int = 32,
    d_model:       int = 256,
    n_layers:      int = 4,
    n_heads:       int = 4,
    conv_kernel:   int = 15,
    dropout:       float = 0.2,
    upsample:      int = 2,
    dec_n_layers:  int = 2,
    dec_n_heads:   int = 4,
    log_dir:       str = "/kaggle/working/logs",
    variant:       str = "stage11",
) -> dict:
    """
    Load the best-val checkpoint and evaluate ONCE on the test split.

    Per Stage 11 §7, this MUST run exactly once.  The orchestrator notebook
    enforces that by gating the call behind a `_already_evaluated` marker
    file in /kaggle/working/.
    """
    marker = os.path.join(log_dir, f".{variant}_test_evaluated")
    if os.path.exists(marker):
        raise RuntimeError(
            f"Test evaluation already completed.  Marker: {marker}.  "
            "Per Stage 11 §7 the test set must be evaluated exactly once.  "
            "Delete the marker manually if you intentionally need to re-run."
        )

    device     = cfg.device
    converter  = make_converter(cfg.data.lang)
    pad_idx    = cfg.vocab.pad_idx
    blank      = cfg.vocab.blank_idx
    att_V      = cfg.vocab.attn_vocab_size
    sos        = cfg.vocab.sos_idx
    eos        = cfg.vocab.eos_idx
    ctc_V      = cfg.vocab.ctc_vocab_size

    test_ds = WiTAPaperSplitDataset(
        cache_root, "test", subsets=("lex", "nonlex"),
        converter=converter, transform=None,
    )
    coll = lambda b: _collate(b, pad_idx=pad_idx)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=coll,
    )

    encoder = ConformerCTC(
        input_dim       = 190, vocab_size = ctc_V,
        d_model         = d_model, n_layers = n_layers, n_heads = n_heads,
        conv_kernel     = conv_kernel, dropout = dropout,
        upsample        = upsample, input_layernorm = False,
    ).to(device)
    decoder = AttentionDecoder(
        att_vocab_size = att_V, bos_idx = sos, eos_idx = eos,
        d_model        = d_model, n_layers = dec_n_layers, n_heads = dec_n_heads,
        ff_mult        = 4, dropout = dropout,
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["encoder_state_dict"], strict=False)
    decoder.load_state_dict(state["decoder_state_dict"], strict=False)
    print(f"[stage11] loaded {checkpoint}  best val_overall="
          f"{state.get('best_payload', {}).get('val_overall_cer', '?')}",
          flush=True)

    test_out = evaluate_loader(
        test_loader, encoder=encoder, decoder=decoder, cfg=cfg,
        blank=blank, sos=sos, eos=eos, pad=pad_idx, device=device,
    )

    test_overall = test_out["overall_cer"]
    test_lex     = test_out["per_subset_cer"].get("lex",    float("nan"))
    test_non     = test_out["per_subset_cer"].get("nonlex", float("nan"))

    # ====================================================================
    # PER §9 OF THE STAGE 11 PROMPT: WRITE THE THREE NUMBERS FIRST,
    # BEFORE LOOKING AT ANYTHING ELSE.  Marker file + JSON committed
    # immediately.
    # ====================================================================
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, f"{variant}_test_headline.json")
    headline = {
        "variant":          variant,
        "test_overall_cer": float(test_overall),
        "test_lex_cer":     float(test_lex),
        "test_nonlex_cer":  float(test_non),
        "n_test_clips":     test_out["n_clips"],
        "paper_baseline": {
            "overall": 0.2924,
            "lex":     0.281,
            "nonlex":  0.365,
        },
        "checkpoint":       checkpoint,
    }
    with open(summary_path, "w") as f:
        json.dump(headline, f, indent=2, sort_keys=True)
    with open(marker, "w") as f:
        f.write(summary_path)

    print("\n" + "=" * 64)
    print(f"  STAGE 11 TEST-SET HEADLINE (written to {summary_path})")
    print("=" * 64)
    print(f"  test_overall_cer : {test_overall:.4f}   (paper: 0.2924)")
    print(f"  test_lex_cer     : {test_lex:.4f}       (paper: 0.281)")
    print(f"  test_nonlex_cer  : {test_non:.4f}       (paper: 0.365)")
    print("=" * 64 + "\n")

    # Write the full diagnostic file separately so prose interpretation
    # only happens after the headline is locked.
    diag_path = os.path.join(log_dir, f"{variant}_test_full.json")
    with open(diag_path, "w") as f:
        json.dump({
            "headline":         headline,
            "per_signer_cer":   test_out["per_signer_cer"],
            "per_length_cer":   test_out["per_length_cer"],
            "n_clips_per_subset": {
                k: sum(1 for r in test_out["pairs"] if r["subset"] == k)
                for k in ("lex", "nonlex")
            },
        }, f, indent=2, sort_keys=True)
    return {
        "headline": headline,
        "per_signer_cer":  test_out["per_signer_cer"],
        "per_length_cer":  test_out["per_length_cer"],
        "pairs":           test_out["pairs"],
    }

"""
training/stage9_train.py — Stage 9a joint CTC + attention trainer.

Same locked Stage 1 v2 encoder + recipe (4-layer Conformer at d=256,
AdamW + OneCycleLR, batch=32, 80 epochs, seed=42).  The new piece is the
attention decoder + joint loss:

    L = lambda_ctc * L_ctc + (1 - lambda_ctc) * L_attn

with lambda_ctc = 0.3 by default (the ESPnet convention).  The attention
decoder consumes the encoder's pre-upsample features [B, T_in=32, d=256]
via cross-attention; the CTC head continues to consume the post-upsample
features [B, T_out=64, d=256].

Val CER is computed for both:
    cer_ctc  : greedy CTC decode (existing pipeline)
    cer_attn : greedy attention decode (left-to-right until EOS)
    cer_best : min(cer_ctc, cer_attn) per clip   (lower bound the joint
               beam decoder will improve on in Stage 9b)
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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import editdistance

from ..models.conformer_ctc       import ConformerCTC
from ..models.attention_decoder   import AttentionDecoder, build_attention_targets
from ..datasets.vocab             import make_converter
from ..training.diagnostics       import (
    full_diagnostic_snapshot, format_snapshot_line, assert_ctc_feasible,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset wrapper (identical to Stage 1 v3's; emits (feats, label_enc, subject))
# ---------------------------------------------------------------------------

class _LandmarkDataset(Dataset):
    def __init__(self, cache: dict, clip_indices: list[int], converter,
                 transform=None):
        self.cache     = cache
        self.indices   = clip_indices
        self.converter = converter
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feats = self.cache["feats"][idx].float()
        if self.transform is not None:
            feats = self.transform(feats)
        enc, _ = self.converter.encode(self.cache["labels"][idx])
        return feats, enc, self.cache["subjects"][idx]


def _collate(batch, pad_idx: int):
    feats, labels, subjs = zip(*batch)
    feats_pad  = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.LongTensor([f.shape[0] for f in feats])
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return feats_pad, labels_pad, input_lens, label_lens, list(subjs)


# ---------------------------------------------------------------------------
# Greedy CTC decode (shared with Stage 1 v3)
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
    return "".join(
        chars[t - 1] if 1 <= t <= len(chars) else "?" for t in ids
    )


# ---------------------------------------------------------------------------
# Per-fold trainer
# ---------------------------------------------------------------------------

def train_one_fold(
    cache:        dict,
    train_idx:    list[int],
    val_idx:      list[int],
    *,
    cfg,
    fold:         int,
    variant:      str = "stage9a",
    num_epochs:   int = 80,
    batch_size:   int = 32,
    lr_peak:      float = 5e-4,
    weight_decay: float = 5e-2,
    grad_clip:    float = 1.0,
    dropout:      float = 0.2,
    d_model:      int = 256,
    n_layers:     int = 4,
    n_heads:      int = 4,
    conv_kernel:  int = 15,
    upsample:     int = 2,
    warmup_pct:   float = 0.05,
    # Stage-9-specific knobs:
    dec_n_layers: int = 3,
    dec_n_heads:  int = 4,
    lambda_ctc:   float = 0.3,
    transform=None,
    seed:           int | None = None,
    checkpoint_dir: str = "/kaggle/working/checkpoints",
    log_dir:        str = "/kaggle/working/logs",
) -> dict:
    """One full training run on one fold.  Returns the result dict.

    If `seed` is given, the trainer re-seeds Python / NumPy / Torch right
    before model construction so the run is reproducible-ish (modulo the
    cudnn nondeterminism left in place by `cudnn.benchmark=True`).  Used
    by the Stage 9a ablation matrix for the per-seed-variance estimate.
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = cfg.device
    converter = make_converter(cfg.data.lang)
    pad_idx     = cfg.vocab.pad_idx          # CE ignore_index + dec_in filler
    blank       = cfg.vocab.blank_idx
    ctc_V       = cfg.vocab.ctc_vocab_size
    att_V       = cfg.vocab.attn_vocab_size
    sos_idx     = cfg.vocab.sos_idx
    eos_idx     = cfg.vocab.eos_idx

    if not (sos_idx < att_V and eos_idx < att_V and pad_idx < att_V):
        raise RuntimeError(
            f"VocabConfig out of sync: "
            f"sos={sos_idx} eos={eos_idx} pad={pad_idx} att_V={att_V}. "
            "All special tokens must be < att_V."
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # -- data --
    train_ds = _LandmarkDataset(cache, train_idx, converter, transform=transform)
    val_ds   = _LandmarkDataset(cache, val_idx,   converter, transform=None)
    collate_fn = lambda b: _collate(b, pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=collate_fn,
    )

    # -- model --
    encoder = ConformerCTC(
        input_dim       = cache["out_dim"],
        vocab_size      = ctc_V,
        d_model         = d_model,
        n_layers        = n_layers,
        n_heads         = n_heads,
        conv_kernel     = conv_kernel,
        dropout         = dropout,
        upsample        = upsample,
        input_layernorm = False,        # landmarks, Stage 1 v2 contract
    ).to(device)
    decoder = AttentionDecoder(
        att_vocab_size = att_V,
        bos_idx        = sos_idx,
        eos_idx        = eos_idx,
        d_model        = d_model,
        n_layers       = dec_n_layers,
        n_heads        = dec_n_heads,
        ff_mult        = 4,
        dropout        = dropout,
    ).to(device)
    bos = decoder.bos
    eos = decoder.eos
    att_pad = pad_idx                      # CE ignores this index, embed has room for it

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
    ce  = nn.CrossEntropyLoss(ignore_index=att_pad)

    history: list[dict] = []
    best_cer = float("inf"); best_epoch = -1
    best_per_signer: dict[str, float] = {}
    best_mean_pred_len_ratio: float = float("nan")
    train_nll_ever_below_05 = False
    ckpt_path = os.path.join(
        checkpoint_dir, f"stage9a_fold{fold}_{variant}_best.pt",
    )

    print(
        f"\n=== Stage 9a fold={fold}  variant={variant}  lambda_ctc={lambda_ctc}  "
        f"enc_params={encoder.num_params:,}  dec_params={decoder.num_params:,}  "
        f"train={len(train_idx)}  val={len(val_idx)} ===", flush=True,
    )

    for epoch in range(num_epochs):
        encoder.train(); decoder.train()
        sum_ctc = 0.0; sum_att = 0.0; sum_total = 0.0
        n_train_batches = 0
        t0 = time.time()

        for feats, labels, in_lens, lab_lens, _ in train_loader:
            feats   = feats.to(device);   labels   = labels.to(device)
            in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

            # 1) Encoder forward, CTC head.
            h, pad_mask = encoder.encode(feats, in_lens)        # [B, T_in, d]
            log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
            assert_ctc_feasible(enc_lens.cpu(), lab_lens.cpu(),
                                raise_on_fail=True)
            ctc_loss = ctc(log_probs.transpose(0, 1).float(),
                           labels, enc_lens, lab_lens)

            # 2) Attention decoder forward (teacher-forced).
            dec_in, dec_tg = build_attention_targets(
                labels, lab_lens, bos=bos, eos=eos, pad=att_pad,
            )
            # Synchronous bounds check on epoch-1 batch-1 — CUDA gather-OOB
            # otherwise surfaces hours later as an async assertion failure.
            if epoch == 0 and n_train_batches == 0:
                _max_in = int(dec_in.max().item())
                _max_tg = int(dec_tg.max().item())
                if _max_in >= decoder.att_vocab_size or _max_tg >= decoder.att_vocab_size:
                    raise RuntimeError(
                        f"Attention-decoder index out of range: "
                        f"max(dec_in)={_max_in}, max(dec_tg)={_max_tg}, "
                        f"att_vocab_size={decoder.att_vocab_size}."
                    )
            dec_logits = decoder(h, pad_mask, dec_in)            # [B, L, att_V]
            attn_loss  = ce(
                dec_logits.reshape(-1, decoder.att_vocab_size),
                dec_tg.reshape(-1),
            )

            total = lambda_ctc * ctc_loss + (1.0 - lambda_ctc) * attn_loss
            optimizer.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step(); scheduler.step()

            sum_ctc   += float(ctc_loss.item())
            sum_att   += float(attn_loss.item())
            sum_total += float(total.item())
            n_train_batches += 1

        train_ctc   = sum_ctc   / max(n_train_batches, 1)
        train_attn  = sum_att   / max(n_train_batches, 1)
        train_total = sum_total / max(n_train_batches, 1)
        if train_ctc < 0.5:
            train_nll_ever_below_05 = True

        # -- val --
        encoder.eval(); decoder.eval()
        pairs_ctc:  list[tuple[str, str]] = []
        pairs_attn: list[tuple[str, str]] = []
        pairs_best: list[tuple[str, str]] = []
        sum_val_ctc = 0.0; n_val_batches = 0
        last_lp = None; last_lens = None
        per_subj_err_b: dict[str, int] = defaultdict(int)
        per_subj_len_b: dict[str, int] = defaultdict(int)

        with torch.no_grad():
            for feats, labels, in_lens, lab_lens, subjs in val_loader:
                feats   = feats.to(device);   labels   = labels.to(device)
                in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

                h, pad_mask = encoder.encode(feats, in_lens)
                log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
                v_ctc = ctc(log_probs.transpose(0, 1).float(),
                            labels, enc_lens, lab_lens)
                sum_val_ctc += float(v_ctc.item()); n_val_batches += 1

                # Greedy CTC predictions.
                ctc_preds = _ctc_greedy(log_probs, enc_lens, blank)
                # Greedy attention predictions.
                attn_preds = decoder.greedy_decode(h, pad_mask)

                for b in range(len(ctc_preds)):
                    gt = converter.decode(
                        labels[b, : int(lab_lens[b].item())].int().cpu(),
                        torch.IntTensor([int(lab_lens[b].item())]),
                    )
                    p_ctc  = _ids_to_str(ctc_preds[b],            cfg.vocab.chars)
                    p_attn = _ids_to_str(attn_preds[b].tolist(),  cfg.vocab.chars)
                    pairs_ctc.append((gt, p_ctc))
                    pairs_attn.append((gt, p_attn))
                    err_ctc  = editdistance.eval(gt, p_ctc)
                    err_attn = editdistance.eval(gt, p_attn)
                    err_best = min(err_ctc, err_attn)
                    p_best   = p_ctc if err_ctc <= err_attn else p_attn
                    pairs_best.append((gt, p_best))
                    per_subj_err_b[subjs[b]] += int(err_best)
                    per_subj_len_b[subjs[b]] += int(len(gt))
                last_lp = log_probs; last_lens = enc_lens

        val_ctc = sum_val_ctc / max(n_val_batches, 1)

        cer_ctc  = sum(editdistance.eval(g, p) for g, p in pairs_ctc)  / max(sum(len(g) for g,_ in pairs_ctc),  1)
        cer_attn = sum(editdistance.eval(g, p) for g, p in pairs_attn) / max(sum(len(g) for g,_ in pairs_attn), 1)
        cer_best = sum(editdistance.eval(g, p) for g, p in pairs_best) / max(sum(len(g) for g,_ in pairs_best), 1)

        # Length-collapse diagnostic: ratio of summed predicted chars to
        # summed reference chars on the best-decoder predictions.
        #   ~1.0 = balanced;  < 0.5 = under-predicting;  > 2.0 = over-predicting.
        sum_pred = sum(len(p) for _, p in pairs_best)
        sum_gt   = sum(len(g) for g, _ in pairs_best)
        mean_pred_len_ratio = sum_pred / max(sum_gt, 1)

        per_signer_cer = {
            s: per_subj_err_b[s] / max(per_subj_len_b[s], 1)
            for s in per_subj_err_b
        }
        snap = full_diagnostic_snapshot(
            pairs=pairs_best, log_probs=last_lp, lengths=last_lens,
            chars=cfg.vocab.chars,
            train_loss=train_ctc, val_loss=val_ctc,
            blank=blank,
        )
        snap["cer_ctc"]  = float(cer_ctc)
        snap["cer_attn"] = float(cer_attn)
        snap["cer_best"] = float(cer_best)
        snap["train_attn_loss"]  = float(train_attn)
        snap["train_total_loss"] = float(train_total)
        snap["mean_pred_len_ratio"] = float(mean_pred_len_ratio)
        snap["per_signer_val_cer"] = per_signer_cer
        # Use the joint (best-of) CER as the headline.
        snap["val_cer_overall"] = float(cer_best)

        history.append({
            "epoch": epoch + 1,
            **{k: v for k, v in snap.items() if not isinstance(v, (list, dict))},
            "per_signer_val_cer": per_signer_cer,
        })

        dt = time.time() - t0
        print(
            f"[F{fold} {variant}] Ep {epoch+1:3d}/{num_epochs}  "
            f"ctc={train_ctc:.4f}  attn={train_attn:.4f}  total={train_total:.4f}  "
            f"val_ctc={val_ctc:.4f}  CER ctc/attn/best={cer_ctc:.4f}/{cer_attn:.4f}/{cer_best:.4f}  "
            f"{dt:.0f}s",
            flush=True,
        )

        if cer_best < best_cer:
            best_cer = cer_best
            best_epoch = epoch + 1
            best_per_signer = dict(per_signer_cer)
            best_mean_pred_len_ratio = float(mean_pred_len_ratio)
            torch.save({
                "fold": fold, "variant": variant,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch": epoch, "val_cer": best_cer, "snapshot": snap,
                "lambda_ctc": lambda_ctc,
            }, ckpt_path)
            print(f"    ★ new best CER={best_cer:.4f}", flush=True)

    result = {
        "fold":                       fold,
        "variant":                    variant,
        "num_epochs":                 num_epochs,
        "n_train_clips":              len(train_idx),
        "n_val_clips":                len(val_idx),
        "best_val_cer":               best_cer,
        "best_epoch":                 best_epoch,
        "train_nll_ever_below_05":    bool(train_nll_ever_below_05),
        "final_train_ctc_nll":        train_ctc,
        "final_train_attn_nll":       train_attn,
        "lambda_ctc":                 lambda_ctc,
        "seed":                       seed,
        "best_mean_pred_len_ratio":   best_mean_pred_len_ratio,
        "best_per_signer_val_cer":    best_per_signer,
        "history":                    history,
        "checkpoint_path":            ckpt_path,
    }
    json_path = os.path.join(
        log_dir, f"stage9a_fold{fold}_{variant}_history.json",
    )
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)
    return result

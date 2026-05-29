"""
training/stage3_train.py — Stage 3 single-fold trainer.

DANN was falsified in Stage 1 v3, so this trainer has NO signer head.  It
runs the locked Stage 1 v2 / 1 v3 recipe (AdamW + OneCycleLR + grad clip,
80 epochs, dropout 0.2, weight_decay 5e-2, lr 5e-4, batch 32, seed 42) on
the Stage 3 feature cache produced by datasets/dinov2_feature_cache.py.

Differences from Stage 1 v3 (per plan §12 contract for Stage 3)
---------------------------------------------------------------
  * Input dim is 3*D+1 (1153 for DINOv2-S/14 at 336x336) instead of 190.
  * ConformerCTC is instantiated with input_layernorm=True so the raw
    DINOv2-statistics input is normalised before the linear projection.
  * Augmentation: LandmarkAugment with p_spatial_jitter=0.0 (temporal-only)
    because the input is a patch embedding, not a coordinate vector.
  * No signer-adversarial head, no GRL, no alpha.

Everything else (optimiser, schedule, ConvTranspose1d upsample to T_out=64,
seed, batch size, num_epochs) is identical to Stage 1 v2.

Stage-0 diagnostics (length-bucketed CER, NLL gap, blank prob, KL, edit
decomposition) are preserved via diagnostics.full_diagnostic_snapshot.
"""

from __future__ import annotations

import os
import json
import time
import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import editdistance

from ..models.conformer_ctc import ConformerCTC
from ..datasets.vocab import make_converter
from ..training.diagnostics import (
    full_diagnostic_snapshot, format_snapshot_line, assert_ctc_feasible,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _Stage3Dataset(Dataset):
    """One fold of the Stage 3 cache.  Emits (feats, label_enc, subject)."""

    def __init__(self, cache: dict, clip_indices: list[int], converter,
                 transform=None):
        self.cache     = cache
        self.indices   = clip_indices
        self.converter = converter
        self.transform = transform

    def __len__(self) -> int:
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
# Greedy CTC decode (matches dann_train._decode_argmax for consistency)
# ---------------------------------------------------------------------------

def _decode_argmax(log_probs: torch.Tensor, enc_lens: torch.Tensor, blank: int):
    out = []
    argmax = log_probs.argmax(dim=-1)
    for b in range(argmax.shape[0]):
        seq = argmax[b, : int(enc_lens[b].item())].tolist()
        merged: list[int] = []; prev: Optional[int] = None
        for t in seq:
            if t != prev and t != blank:
                merged.append(t)
            prev = t
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Per-fold trainer
# ---------------------------------------------------------------------------

def train_one_fold(
    cache:           dict,
    train_idx:       list[int],
    val_idx:         list[int],
    *,
    cfg,
    fold:            int,
    variant:         str = "stage3",          # 'stage3' | 'stage3_multijoint'
    num_epochs:      int = 80,
    batch_size:      int = 32,
    lr_peak:         float = 5e-4,
    weight_decay:    float = 5e-2,
    grad_clip:       float = 1.0,
    dropout:         float = 0.2,
    d_model:         int = 256,
    n_layers:        int = 4,
    n_heads:         int = 4,
    conv_kernel:     int = 15,
    upsample:        int = 2,
    warmup_pct:      float = 0.05,
    transform=None,
    checkpoint_dir:  str = "/kaggle/working/checkpoints",
    log_dir:         str = "/kaggle/working/logs",
) -> dict:
    """
    Run one Stage 3 fold.  Returns the same result-dict shape as Stage 1 v3
    so the notebook aggregator and reporting templates can be reused.

    Pass/fail per the post-Stage-1-v3 prompt §2 Task B:
      * train_ctc_loss must drop below 0.5 within `num_epochs` epochs.
        If it stalls above 0.5, the fingertip-pool design is feature-
        insufficient and Stage 4 (fusion) will not save you — bail out.
      * Headline: best_val_cer <= 0.860.
    """
    device = cfg.device
    converter = make_converter(cfg.data.lang)
    pad_idx = cfg.vocab.pad_idx
    blank   = cfg.vocab.blank_idx

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # -- data --
    train_ds = _Stage3Dataset(cache, train_idx, converter, transform=transform)
    val_ds   = _Stage3Dataset(cache, val_idx,   converter, transform=None)
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
    model = ConformerCTC(
        input_dim       = cache["out_dim"],
        vocab_size      = cfg.vocab.ctc_vocab_size,
        d_model         = d_model,
        n_layers        = n_layers,
        n_heads         = n_heads,
        conv_kernel     = conv_kernel,
        dropout         = dropout,
        upsample        = upsample,
        input_layernorm = True,                 # Stage 3 design contract §5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_peak, weight_decay=weight_decay, betas=(0.9, 0.999),
    )
    total_steps = num_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_peak, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    history: list[dict] = []
    best_cer = float("inf"); best_epoch = -1
    best_per_signer: dict[str, float] = {}
    train_nll_ever_below_05 = False           # gating diagnostic
    ckpt_path = os.path.join(
        checkpoint_dir, f"stage3_fold{fold}_{variant}_best.pt",
    )

    print(
        f"\n=== Stage 3 fold={fold}  variant={variant}  "
        f"in_dim={cache['out_dim']}  train_clips={len(train_idx)}  "
        f"val_clips={len(val_idx)}  params={model.num_params:,} ===",
        flush=True,
    )

    for epoch in range(num_epochs):
        model.train()
        sum_ctc = 0.0; n_train_batches = 0
        t0 = time.time()

        for feats, labels, in_lens, lab_lens, _ in train_loader:
            feats   = feats.to(device);   labels   = labels.to(device)
            in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

            log_probs, enc_lens = model(feats, in_lens)
            assert_ctc_feasible(enc_lens.cpu(), lab_lens.cpu(),
                                raise_on_fail=True)
            loss = ctc(log_probs.transpose(0, 1).float(),
                       labels, enc_lens, lab_lens)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step(); scheduler.step()

            sum_ctc += float(loss.item())
            n_train_batches += 1

        train_loss = sum_ctc / max(n_train_batches, 1)
        if train_loss < 0.5:
            train_nll_ever_below_05 = True

        # -- val --
        model.eval()
        pairs: list[tuple[str, str]] = []
        sum_val_loss = 0.0; n_val_batches = 0
        last_lp = None; last_lens = None
        per_subj_err: dict[str, int] = defaultdict(int)
        per_subj_len: dict[str, int] = defaultdict(int)

        with torch.no_grad():
            for feats, labels, in_lens, lab_lens, subjs in val_loader:
                feats   = feats.to(device);   labels   = labels.to(device)
                in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

                log_probs, enc_lens = model(feats, in_lens)
                v_loss = ctc(log_probs.transpose(0, 1).float(),
                             labels, enc_lens, lab_lens)
                sum_val_loss += float(v_loss.item()); n_val_batches += 1
                preds = _decode_argmax(log_probs, enc_lens, blank)
                for b in range(len(preds)):
                    gt = converter.decode(
                        labels[b, : int(lab_lens[b].item())].int().cpu(),
                        torch.IntTensor([int(lab_lens[b].item())]),
                    )
                    pred_str = "".join(
                        cfg.vocab.chars[t-1] if 1 <= t <= len(cfg.vocab.chars) else "?"
                        for t in preds[b]
                    )
                    pairs.append((gt, pred_str))
                    err = editdistance.eval(gt, pred_str)
                    per_subj_err[subjs[b]] += int(err)
                    per_subj_len[subjs[b]] += int(len(gt))
                last_lp = log_probs; last_lens = enc_lens
        val_loss = sum_val_loss / max(n_val_batches, 1)

        per_signer_cer = {
            s: per_subj_err[s] / max(per_subj_len[s], 1)
            for s in per_subj_err
        }

        snap = full_diagnostic_snapshot(
            pairs=pairs, log_probs=last_lp, lengths=last_lens,
            chars=cfg.vocab.chars,
            train_loss=train_loss, val_loss=val_loss,
            blank=blank,
        )
        snap["per_signer_val_cer"] = per_signer_cer

        history.append({
            "epoch": epoch + 1,
            **{k: v for k, v in snap.items() if not isinstance(v, (list, dict))},
            "per_signer_val_cer": per_signer_cer,
        })

        dt = time.time() - t0
        print(
            f"[F{fold} {variant}] Ep {epoch+1:3d}/{num_epochs}  "
            f"ctc={train_loss:.4f}  val={val_loss:.4f}  "
            f"{format_snapshot_line(snap)}  {dt:.0f}s",
            flush=True,
        )

        if snap["val_cer_overall"] < best_cer:
            best_cer        = snap["val_cer_overall"]
            best_epoch      = epoch + 1
            best_per_signer = dict(per_signer_cer)
            torch.save({
                "fold": fold, "variant": variant,
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_cer": best_cer, "snapshot": snap,
                "fingerprint": cache.get("fingerprint", {}),
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
        "final_train_nll":            train_loss,
        "best_per_signer_val_cer":    best_per_signer,
        "history":                    history,
        "checkpoint_path":            ckpt_path,
        "in_dim":                     cache["out_dim"],
        "fingerprint":                cache.get("fingerprint", {}),
    }
    json_path = os.path.join(
        log_dir, f"stage3_fold{fold}_{variant}_history.json",
    )
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)

    if not train_nll_ever_below_05:
        print(
            f"[F{fold} {variant}] !!! train NLL never reached < 0.5 "
            f"(final={train_loss:.4f}).  Per Stage 3 contract, this design "
            f"is feature-insufficient — Stage 4 fusion will not help.",
            flush=True,
        )

    return result

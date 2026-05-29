"""
training/stage4_train.py — Stage 4 dual-stream fusion trainer.

Two fusion modes (see configs/stage4_earlyfusion.yaml,
configs/stage4_latefusion.yaml):

  mode='early': single ConformerCTC over cat([landmark_feat, dino_feat]).
                Reuses ConformerCTC with input_dim=1343 and
                input_layernorm=True.

  mode='late':  LateFusionConformerCTC with two parallel 3-layer Conformers
                at d=128 each, concatenated to a 256-d fused stream.

Both caches must be index-aligned (built from the same `samples` stream,
same seed, same Korean/English filter).  The trainer enforces this by
checking len() and the per-clip subjects list.
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
from ..models.dual_stream_head import LateFusionConformerCTC
from ..datasets.vocab import make_converter
from ..training.diagnostics import (
    full_diagnostic_snapshot, format_snapshot_line, assert_ctc_feasible,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dual-cache dataset
# ---------------------------------------------------------------------------

def _verify_caches_aligned(cache_l: dict, cache_d: dict) -> None:
    if len(cache_l["feats"]) != len(cache_d["feats"]):
        raise RuntimeError(
            f"Stage 4 caches have different lengths: "
            f"landmark={len(cache_l['feats'])}  dino={len(cache_d['feats'])}"
        )
    if cache_l["subjects"] != cache_d["subjects"]:
        bad = sum(1 for a, b in zip(cache_l["subjects"], cache_d["subjects"]) if a != b)
        raise RuntimeError(
            f"Stage 4 caches' per-clip subjects disagree on {bad} clips. "
            "Both caches must be built from the same `samples` stream."
        )
    if cache_l["labels"] != cache_d["labels"]:
        bad = sum(1 for a, b in zip(cache_l["labels"], cache_d["labels"]) if a != b)
        raise RuntimeError(
            f"Stage 4 caches' per-clip labels disagree on {bad} clips."
        )


class _DualCacheDataset(Dataset):
    """Emits (landmark_feat, dino_feat, label_enc, subject) per __getitem__."""

    def __init__(
        self,
        cache_l:      dict,
        cache_d:      dict,
        clip_indices: list[int],
        converter,
        transform_l = None,
        transform_d = None,
    ):
        _verify_caches_aligned(cache_l, cache_d)
        self.cache_l = cache_l
        self.cache_d = cache_d
        self.indices = clip_indices
        self.converter = converter
        self.transform_l = transform_l
        self.transform_d = transform_d

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feats_l = self.cache_l["feats"][idx].float()
        feats_d = self.cache_d["feats"][idx].float()
        if self.transform_l is not None:
            feats_l = self.transform_l(feats_l)
        if self.transform_d is not None:
            feats_d = self.transform_d(feats_d)
        # Both streams use T_native=32 so lengths agree.  Crop to min just in case.
        T = min(feats_l.shape[0], feats_d.shape[0])
        feats_l = feats_l[:T]; feats_d = feats_d[:T]
        enc, _ = self.converter.encode(self.cache_l["labels"][idx])
        return feats_l, feats_d, enc, self.cache_l["subjects"][idx]


def _collate(batch, pad_idx: int):
    feats_l, feats_d, labels, subjs = zip(*batch)
    feats_l_pad = pad_sequence(feats_l, batch_first=True, padding_value=0.0)
    feats_d_pad = pad_sequence(feats_d, batch_first=True, padding_value=0.0)
    labels_pad  = pad_sequence(labels,  batch_first=True, padding_value=pad_idx)
    input_lens  = torch.LongTensor([f.shape[0] for f in feats_l])
    label_lens  = torch.LongTensor([l.shape[0] for l in labels])
    return feats_l_pad, feats_d_pad, labels_pad, input_lens, label_lens, list(subjs)


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
    cache_landmark: dict,
    cache_dino:     dict,
    train_idx:      list[int],
    val_idx:        list[int],
    *,
    cfg,
    fold:           int,
    fusion_mode:    str,                # 'early' | 'late'
    variant:        str = None,         # logging-only label
    num_epochs:     int = 80,
    batch_size:     int = 32,
    lr_peak:        float = 5e-4,
    weight_decay:   float = 5e-2,
    grad_clip:      float = 1.0,
    dropout:        float = 0.2,
    d_model:        int = 256,           # early fusion shared d
    d_per_stream:   int = 128,           # late fusion per-stream d
    n_layers:       int = 4,             # early-fusion Conformer depth
    n_layers_per_stream: int = 3,        # late-fusion per-stream depth
    n_heads:        int = 4,
    conv_kernel:    int = 15,
    upsample:       int = 2,
    warmup_pct:     float = 0.05,
    transform_landmark = None,
    transform_dino     = None,
    checkpoint_dir: str = "/kaggle/working/checkpoints",
    log_dir:        str = "/kaggle/working/logs",
) -> dict:
    if fusion_mode not in ("early", "late"):
        raise ValueError(f"fusion_mode must be 'early' or 'late', got {fusion_mode}")
    if variant is None:
        variant = f"stage4_{fusion_mode}"

    device = cfg.device
    converter = make_converter(cfg.data.lang)
    pad_idx = cfg.vocab.pad_idx
    blank   = cfg.vocab.blank_idx

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    landmark_dim = cache_landmark["out_dim"]
    dino_dim     = cache_dino["out_dim"]

    train_ds = _DualCacheDataset(
        cache_landmark, cache_dino, train_idx, converter,
        transform_l=transform_landmark, transform_d=transform_dino,
    )
    val_ds = _DualCacheDataset(
        cache_landmark, cache_dino, val_idx, converter,
        transform_l=None, transform_d=None,
    )
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
    if fusion_mode == "early":
        in_dim = landmark_dim + dino_dim
        model = ConformerCTC(
            input_dim       = in_dim,
            vocab_size      = cfg.vocab.ctc_vocab_size,
            d_model         = d_model,
            n_layers        = n_layers,
            n_heads         = n_heads,
            conv_kernel     = conv_kernel,
            dropout         = dropout,
            upsample        = upsample,
            input_layernorm = True,
        ).to(device)
    else:
        model = LateFusionConformerCTC(
            landmark_dim         = landmark_dim,
            dino_dim             = dino_dim,
            vocab_size           = cfg.vocab.ctc_vocab_size,
            d_per_stream         = d_per_stream,
            n_layers_per_stream  = n_layers_per_stream,
            n_heads              = n_heads,
            conv_kernel          = conv_kernel,
            dropout              = dropout,
            upsample             = upsample,
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
    ckpt_path = os.path.join(
        checkpoint_dir, f"stage4_fold{fold}_{variant}_best.pt",
    )

    print(
        f"\n=== Stage 4 ({fusion_mode}) fold={fold}  "
        f"landmark_dim={landmark_dim}  dino_dim={dino_dim}  "
        f"params={model.num_params:,}  train={len(train_idx)}  val={len(val_idx)} ===",
        flush=True,
    )

    for epoch in range(num_epochs):
        model.train()
        sum_ctc = 0.0; n_train_batches = 0
        t0 = time.time()

        for feats_l, feats_d, labels, in_lens, lab_lens, _ in train_loader:
            feats_l  = feats_l.to(device);  feats_d  = feats_d.to(device)
            labels   = labels.to(device);   in_lens  = in_lens.to(device)
            lab_lens = lab_lens.to(device)

            if fusion_mode == "early":
                feats = torch.cat([feats_l, feats_d], dim=-1)
                log_probs, enc_lens = model(feats, in_lens)
            else:
                log_probs, enc_lens = model(feats_l, feats_d, in_lens)
            assert_ctc_feasible(enc_lens.cpu(), lab_lens.cpu(), raise_on_fail=True)
            loss = ctc(log_probs.transpose(0, 1).float(),
                       labels, enc_lens, lab_lens)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step(); scheduler.step()
            sum_ctc += float(loss.item()); n_train_batches += 1

        train_loss = sum_ctc / max(n_train_batches, 1)

        # -- val --
        model.eval()
        pairs: list[tuple[str, str]] = []
        sum_val_loss = 0.0; n_val_batches = 0
        last_lp = None; last_lens = None
        per_subj_err: dict[str, int] = defaultdict(int)
        per_subj_len: dict[str, int] = defaultdict(int)

        with torch.no_grad():
            for feats_l, feats_d, labels, in_lens, lab_lens, subjs in val_loader:
                feats_l  = feats_l.to(device);  feats_d  = feats_d.to(device)
                labels   = labels.to(device);   in_lens  = in_lens.to(device)
                lab_lens = lab_lens.to(device)
                if fusion_mode == "early":
                    feats = torch.cat([feats_l, feats_d], dim=-1)
                    log_probs, enc_lens = model(feats, in_lens)
                else:
                    log_probs, enc_lens = model(feats_l, feats_d, in_lens)
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
                "fold": fold, "variant": variant, "fusion_mode": fusion_mode,
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_cer": best_cer, "snapshot": snap,
            }, ckpt_path)
            print(f"    ★ new best CER={best_cer:.4f}", flush=True)

    result = {
        "fold":                    fold,
        "variant":                 variant,
        "fusion_mode":             fusion_mode,
        "num_epochs":              num_epochs,
        "n_train_clips":           len(train_idx),
        "n_val_clips":             len(val_idx),
        "best_val_cer":            best_cer,
        "best_epoch":              best_epoch,
        "best_per_signer_val_cer": best_per_signer,
        "history":                 history,
        "checkpoint_path":         ckpt_path,
        "landmark_dim":            landmark_dim,
        "dino_dim":                dino_dim,
    }
    json_path = os.path.join(
        log_dir, f"stage4_fold{fold}_{variant}_history.json",
    )
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)
    return result

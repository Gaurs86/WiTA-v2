"""
training/dann_train.py — Single-fold trainer for Stage 1 v3 DANN runs.

One call to `train_one_fold(...)` runs one (variant, fold) combination.
The notebook orchestrates the 15-call sweep and aggregates.

Contract (per Stage 1 v3 plan §3):
  * Stage 1 v2's ConformerCTC config — UNCHANGED
  * Stage 1 v2's optimiser, scheduler, augmentation, seed — UNCHANGED
  * The single new thing: a SignerAdversary attached after the Conformer
    encoder (pre-upsample), trained jointly with the CTC objective.

This module owns:
  - the joint training loop (CTC + adversarial CE)
  - λ scheduling via lambda_schedule()
  - per-epoch diagnostic logging (Stage-0 suite + DANN-specific signals)
  - per-signer val CER tracking (for the per-fold outlier dot plot)
  - best-checkpoint saving keyed by val CER

It does NOT:
  - choose the fold (caller responsibility)
  - aggregate folds (the notebook does that)
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
from ..models.signer_adversary import SignerAdversary
from ..models.grl import lambda_schedule
from ..datasets.vocab import make_converter
from ..training.diagnostics import (
    full_diagnostic_snapshot, format_snapshot_line, assert_ctc_feasible,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset wrapper that emits (feats, label_enc, signer_local_idx)
# ---------------------------------------------------------------------------

class _SkeletonDatasetWithSigner(Dataset):
    """Wraps the skeleton cache for one fold; emits the local signer label."""

    def __init__(
        self,
        cache:            dict,
        clip_indices:     list[int],
        signer_to_local:  dict[str, int],
        converter,
        transform=None,
        emit_signer:      bool = True,
        unknown_signer_label: int = -100,   # PyTorch CE ignore_index
    ):
        self.cache            = cache
        self.indices          = clip_indices
        self.signer_to_local  = signer_to_local
        self.converter        = converter
        self.transform        = transform
        self.emit_signer      = emit_signer
        self.unknown_label    = unknown_signer_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feats = self.cache["feats"][idx].float()
        if self.transform is not None:
            feats = self.transform(feats)
        label_str = self.cache["labels"][idx]
        subj      = self.cache["subjects"][idx]
        enc, _ = self.converter.encode(label_str)

        signer_lbl = (
            self.signer_to_local.get(subj, self.unknown_label)
            if self.emit_signer else 0
        )
        return feats, enc, int(signer_lbl), subj


def _collate(batch, pad_idx: int):
    feats, labels, signer_lbls, subjects = zip(*batch)
    feats_pad  = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens  = torch.LongTensor([f.shape[0] for f in feats])
    label_lens  = torch.LongTensor([l.shape[0] for l in labels])
    signer_lbls = torch.LongTensor(list(signer_lbls))
    return feats_pad, labels_pad, input_lens, label_lens, signer_lbls, list(subjects)


# ---------------------------------------------------------------------------
# Per-fold trainer
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


def train_one_fold(
    cache:           dict,
    train_idx:       list[int],
    val_idx:         list[int],
    signer_to_local: dict[str, int],
    *,
    cfg,                                      # WiTA Config object
    fold:            int,
    variant:         str,                     # 'no_dann' | 'dann_a1' | 'dann_a03'
    alpha:           float,                   # 0.0 disables DANN
    gamma_grl:       float,
    num_epochs:      int,
    batch_size:      int,
    lr_peak:         float,
    weight_decay:    float,
    grad_clip:       float,
    dropout:         float,
    d_model:         int,
    n_layers:        int,
    n_heads:         int,
    conv_kernel:     int,
    upsample:        int,
    warmup_pct:      float,
    transform=None,
    checkpoint_dir:  str = "/kaggle/working/checkpoints",
    log_dir:         str = "/kaggle/working/logs",
) -> dict:
    """
    One full training run on one fold.  Returns a result dict with the
    best-epoch metrics, full per-epoch history, and per-signer val CER.

    `alpha=0.0` disables the signer head entirely (variant A).
    """
    device = cfg.device
    converter = make_converter(cfg.data.lang)
    pad_idx = cfg.vocab.pad_idx
    blank   = cfg.vocab.blank_idx
    n_signers = len(signer_to_local)
    use_dann = alpha > 0.0

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # -- data --
    train_ds = _SkeletonDatasetWithSigner(
        cache, train_idx, signer_to_local, converter,
        transform=transform, emit_signer=True,
    )
    # Val signers are NOT in signer_to_local — emit a sentinel and ignore.
    val_ds = _SkeletonDatasetWithSigner(
        cache, val_idx, signer_to_local, converter,
        transform=None, emit_signer=True, unknown_signer_label=-100,
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
    model = ConformerCTC(
        input_dim   = cache["out_dim"],
        vocab_size  = cfg.vocab.ctc_vocab_size,
        d_model     = d_model,
        n_layers    = n_layers,
        n_heads     = n_heads,
        conv_kernel = conv_kernel,
        dropout     = dropout,
        upsample    = upsample,
    ).to(device)

    signer_head: Optional[SignerAdversary] = None
    if use_dann:
        signer_head = SignerAdversary(
            d_model   = d_model,
            n_signers = n_signers,
            hidden    = 128,
            dropout   = 0.3,
        ).to(device)

    # Single optimiser over both modules (Adam handles it cleanly).
    params = list(model.parameters())
    if signer_head is not None:
        params += list(signer_head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=lr_peak, weight_decay=weight_decay, betas=(0.9, 0.999),
    )
    total_steps = num_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_peak, total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    ce  = nn.CrossEntropyLoss(ignore_index=-100)

    # -- training loop --
    history: list[dict] = []
    best_cer = float("inf")
    best_epoch = -1
    best_per_signer: dict[str, float] = {}
    ckpt_path = os.path.join(
        checkpoint_dir, f"stage1v3_fold{fold}_{variant}_best.pt",
    )

    global_step = 0
    print(
        f"\n=== fold={fold}  variant={variant}  alpha={alpha}  "
        f"n_signers={n_signers}  train_clips={len(train_idx)}  "
        f"val_clips={len(val_idx)} ===", flush=True,
    )

    for epoch in range(num_epochs):
        model.train()
        if signer_head is not None:
            signer_head.train()

        sum_ctc = 0.0; sum_sgn = 0.0; sum_sgn_acc = 0.0
        n_train_batches = 0
        t0 = time.time()
        last_lambda = 0.0

        for feats, labels, in_lens, lab_lens, sgn_lbls, _ in train_loader:
            feats   = feats.to(device);   labels   = labels.to(device)
            in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)
            sgn_lbls = sgn_lbls.to(device)

            enc_out, pad_mask = model.encode(feats, in_lens)
            log_probs, enc_lens = model.decode_ctc(enc_out, in_lens)
            assert_ctc_feasible(enc_lens.cpu(), lab_lens.cpu(),
                                raise_on_fail=True)
            ctc_loss = ctc(log_probs.transpose(0, 1).float(),
                            labels, enc_lens, lab_lens)

            total_loss = ctc_loss
            if use_dann and signer_head is not None:
                lam = lambda_schedule(global_step, total_steps, gamma=gamma_grl)
                last_lambda = lam
                sgn_logits = signer_head(enc_out, lambda_=lam, mask=pad_mask)
                sgn_loss   = ce(sgn_logits, sgn_lbls)
                total_loss = ctc_loss + alpha * sgn_loss
                sum_sgn += float(sgn_loss.item())
                sum_sgn_acc += float(
                    (sgn_logits.argmax(dim=-1) == sgn_lbls).float().mean().item()
                )

            optimizer.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step(); scheduler.step()

            sum_ctc += float(ctc_loss.item())
            n_train_batches += 1
            global_step += 1

        train_ctc_loss = sum_ctc / max(n_train_batches, 1)
        train_sgn_loss = sum_sgn / max(n_train_batches, 1) if use_dann else 0.0
        train_sgn_acc  = sum_sgn_acc / max(n_train_batches, 1) if use_dann else 0.0

        # -- val --
        model.eval()
        if signer_head is not None:
            signer_head.eval()
        pairs: list[tuple[str, str]] = []
        sum_val_loss = 0.0; n_val_batches = 0
        last_lp = None; last_lens = None
        per_subj_err: dict[str, int]  = defaultdict(int)
        per_subj_len: dict[str, int]  = defaultdict(int)

        with torch.no_grad():
            for feats, labels, in_lens, lab_lens, sgn_lbls, subjs in val_loader:
                feats   = feats.to(device);   labels   = labels.to(device)
                in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)
                enc_out, _ = model.encode(feats, in_lens)
                log_probs, enc_lens = model.decode_ctc(enc_out, in_lens)
                v_loss = ctc(log_probs.transpose(0,1).float(),
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

        # per-signer CER (val set only)
        per_signer_cer = {
            s: per_subj_err[s] / max(per_subj_len[s], 1)
            for s in per_subj_err
        }

        snap = full_diagnostic_snapshot(
            pairs=pairs, log_probs=last_lp, lengths=last_lens,
            chars=cfg.vocab.chars,
            train_loss=train_ctc_loss, val_loss=val_loss,
            blank=blank,
        )
        snap["lambda_current"]    = float(last_lambda)
        snap["train_sgn_loss"]    = float(train_sgn_loss)
        snap["train_sgn_acc"]     = float(train_sgn_acc)
        snap["per_signer_val_cer"] = per_signer_cer

        history.append({
            "epoch": epoch + 1,
            **{k: v for k, v in snap.items() if not isinstance(v, (list, dict))},
            "per_signer_val_cer": per_signer_cer,
        })

        dt = time.time() - t0
        extra = ""
        if use_dann:
            extra = (f" λ={last_lambda:.2f} sgn_loss={train_sgn_loss:.3f} "
                     f"sgn_acc={train_sgn_acc:.2f}")
        print(
            f"[F{fold} {variant}] Ep {epoch+1:3d}/{num_epochs}  "
            f"ctc={train_ctc_loss:.4f}  val={val_loss:.4f}  "
            f"{format_snapshot_line(snap)}{extra}  {dt:.0f}s",
            flush=True,
        )

        if snap["val_cer_overall"] < best_cer:
            best_cer = snap["val_cer_overall"]
            best_epoch = epoch + 1
            best_per_signer = dict(per_signer_cer)
            torch.save({
                "fold": fold, "variant": variant,
                "model_state_dict": model.state_dict(),
                "signer_head_state_dict":
                    signer_head.state_dict() if signer_head is not None else None,
                "epoch": epoch, "val_cer": best_cer, "snapshot": snap,
            }, ckpt_path)
            print(f"    ★ new best CER={best_cer:.4f}", flush=True)

    result = {
        "fold":             fold,
        "variant":          variant,
        "alpha":            alpha,
        "gamma_grl":        gamma_grl,
        "num_epochs":       num_epochs,
        "n_signers_train":  n_signers,
        "n_train_clips":    len(train_idx),
        "n_val_clips":      len(val_idx),
        "best_val_cer":     best_cer,
        "best_epoch":       best_epoch,
        "best_per_signer_val_cer": best_per_signer,
        "history":          history,
        "checkpoint_path":  ckpt_path,
    }
    # Persist a per-fold JSON for safety
    json_path = os.path.join(
        log_dir, f"stage1v3_fold{fold}_{variant}_history.json",
    )
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)
    # full history separately (it can be large)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2)

    return result

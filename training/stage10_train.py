"""
training/stage10_train.py — Stage 10: end-to-end DINOv2 last-2-blocks
unfreeze + 7x7 fingertip-window bell pool.

This is the 1-fold gate test for "does fine-tuning DINOv2 around the
fingertip fundamentally change the appearance-vs-kinematics story?"

Design
------
* DINOv2-S/14 at 336x336, FIRST 10 BLOCKS FROZEN, LAST 2 BLOCKS TRAINABLE.
* For each frame: extract 24x24 patch grid -> identify the patch
  containing the fingertip (joint 8) -> bell-weighted pool over the
  7x7 window (K=3) centered there.
* Per-frame feature D=384 from DINOv2-S; +-1 temporal context concat
  -> 3D=1152; visibility gate appends a 0/1 bit -> 1153-dim.
* Same Conformer + joint CTC + attention decoder as Stage 9a.
* Two optimizer groups: DINOv2 unfrozen blocks at LR 5e-6, head at LR 5e-4.

Memory + compute
----------------
End-to-end gradients through DINOv2 require raw frames in the loop, not
the pre-pooled Stage 3 cache.  To fit in Kaggle T4's ~13GB RAM:

  * Train on a 500-clip random subset of fold-0 train (~18% of full set).
  * Validate on the FULL 720-clip fold-0 val set (no subsetting; the
    gate metric is the per-signer val CER).
  * Cropped 336x336 uint8 frames at T_native=32:
      32 frames * 336 * 336 * 3 = 10.8 MB / clip
    1220 clips * 10.8 MB ≈ 13 GB.  Borderline; bumps notebook
    BATCH_SIZE down to 8 if necessary.

This is a 1-fold gate.  If the verdict is positive (>=0.20 CER drop vs
Stage 9a fold 0 = 0.5681), the next step is a full 5-fold sweep on a
Colab Pro A100 with the unsampled training set.
"""

from __future__ import annotations

import io
import os
import json
import math
import time
import logging
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import editdistance

from ..models.encoders.dinov2_encoder import DINOv2Encoder, default_normalize
from ..models.encoders.dinov2_fingertip_extractor import (
    bell_weights_window, fingertip_pool_window, INDEX_FINGER_TIP,
)
from ..models.conformer_ctc       import ConformerCTC
from ..models.attention_decoder   import AttentionDecoder, build_attention_targets
from ..datasets.skeleton_cache    import LandmarkExtractor
from ..datasets.dinov2_fingertip_cache import (
    _detect_bbox_from_landmarks, _fingertip_in_cropped,
    _crop_resize, _resample_uniform,
)
from ..datasets.vocab             import make_converter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-RAM frame cache
# ---------------------------------------------------------------------------

class _Stage10ClipCache:
    """
    Pre-extracts cropped+resized uint8 frames + fingertip xy (cropped-frame
    normalised) + visibility per clip.  Held in RAM so the training loop
    avoids re-decoding JPEG every batch.

    Per clip storage (T_native=32, image_size=336):
        frames    : np.uint8  [T, H, W, 3]    10.8 MB
        tip_xy    : np.float32[T, 2]            256 B
        vis       : np.float32[T]               128 B
        label     : str
        subject   : str
    """

    def __init__(self, image_size: int = 336, T_native: int = 32,
                 padding_ratio: float = 0.3):
        self.image_size    = image_size
        self.T_native      = T_native
        self.padding_ratio = padding_ratio
        self.frames:   list[np.ndarray] = []
        self.tip_xy:   list[np.ndarray] = []
        self.vis:      list[np.ndarray] = []
        self.labels:   list[str]        = []
        self.subjects: list[str]        = []

    def add_clip(self, frame_bytes: list[bytes],
                 label: str, subject: str,
                 extractor: LandmarkExtractor) -> bool:
        """Process one clip and append.  Returns False if extraction failed."""
        if not frame_bytes:
            return False
        try:
            pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
        except Exception:
            return False

        # MediaPipe landmarks per frame.
        lms: list[Optional[np.ndarray]] = [extractor.detect(f) for f in pil]
        # Union bbox.
        bbox = _detect_bbox_from_landmarks(
            lms, pil[0].size, padding_ratio=self.padding_ratio,
        )

        # Crop + resize.
        crops = [_crop_resize(f, bbox, self.image_size) for f in pil]
        frames_native = np.stack(
            [np.asarray(c, dtype=np.uint8) for c in crops], axis=0,
        )                                                       # [T_raw, H, W, 3]

        # Per-frame fingertip in cropped-frame normalised coords.
        T_raw = len(pil)
        tip_xy_native = np.zeros((T_raw, 2), dtype=np.float32)
        vis_native    = np.zeros(T_raw,      dtype=np.float32)
        last_valid_xy: Optional[tuple[float, float]] = None
        for t, lm in enumerate(lms):
            if lm is None:
                # fall back to last valid; else centre.
                xy = last_valid_xy if last_valid_xy is not None else (0.5, 0.5)
                vis_native[t] = 0.0
            else:
                xy = _fingertip_in_cropped(
                    (float(lm[INDEX_FINGER_TIP, 0]),
                     float(lm[INDEX_FINGER_TIP, 1])),
                    bbox, pil[t].size,
                )
                last_valid_xy = xy
                vis_native[t] = 1.0
            tip_xy_native[t, 0] = xy[0]
            tip_xy_native[t, 1] = xy[1]

        # Resample to T_native.  Use nearest-neighbour temporal indexing
        # for frames (uint8) and linear for tip/vis.
        T = self.T_native
        idx = np.linspace(0, T_raw - 1, T)
        idx_int = np.round(idx).astype(np.int64).clip(0, T_raw - 1)
        frames_T = frames_native[idx_int]                       # [T, H, W, 3]

        idx_lo  = np.floor(idx).astype(np.int64).clip(0, T_raw - 1)
        idx_hi  = np.minimum(idx_lo + 1, T_raw - 1)
        frac    = (idx - idx_lo)[:, None]
        tip_xy_T = (1 - frac) * tip_xy_native[idx_lo] + frac * tip_xy_native[idx_hi]
        vis_T    = ((1 - frac.squeeze(-1)) * vis_native[idx_lo]
                    + frac.squeeze(-1) * vis_native[idx_hi])
        vis_T    = (vis_T >= 0.5).astype(np.float32)

        self.frames.append(frames_T)
        self.tip_xy.append(tip_xy_T.astype(np.float32))
        self.vis.append(vis_T.astype(np.float32))
        self.labels.append(label)
        self.subjects.append(subject)
        return True

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def memory_mb(self) -> float:
        b = sum(f.nbytes + xy.nbytes + v.nbytes
                for f, xy, v in zip(self.frames, self.tip_xy, self.vis))
        return b / 1e6


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _Stage10Dataset(Dataset):
    """Wraps a _Stage10ClipCache for one fold's train or val subset."""

    def __init__(self, clip_cache: _Stage10ClipCache, indices: list[int],
                 converter, augment: bool = False):
        self.cache     = clip_cache
        self.indices   = indices
        self.converter = converter
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # Convert uint8 -> float32 [0,1], CHW, ImageNet normalised.
        frames = torch.from_numpy(self.cache.frames[idx]).float() / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()        # [T, 3, H, W]
        frames = default_normalize(frames)                       # in-place safe
        tip_xy = torch.from_numpy(self.cache.tip_xy[idx])        # [T, 2]
        vis    = torch.from_numpy(self.cache.vis[idx])           # [T]
        enc, _ = self.converter.encode(self.cache.labels[idx])
        return frames, tip_xy, vis, enc, self.cache.subjects[idx]


def _collate(batch, pad_idx: int):
    frames, tip_xy, vis, labels, subjs = zip(*batch)
    # All clips share T_native, so frames/tip_xy/vis stack cleanly.
    frames = torch.stack(frames, dim=0)              # [B, T, 3, H, W]
    tip_xy = torch.stack(tip_xy, dim=0)              # [B, T, 2]
    vis    = torch.stack(vis,    dim=0)              # [B, T]
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.LongTensor([f.shape[0] for f in frames])
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return frames, tip_xy, vis, labels_pad, input_lens, label_lens, list(subjs)


# ---------------------------------------------------------------------------
# Stage 10 forward module: DINOv2 -> windowed pool -> temporal context + vis
# ---------------------------------------------------------------------------

class Stage10FeatureModule(nn.Module):
    """
    Composite: DINOv2 (with last-N unfreeze) + per-frame windowed bell pool
    around fingertip + +-1 temporal context concat + visibility gate.
    """

    def __init__(
        self,
        encoder:           DINOv2Encoder,
        k:                 int = 3,
        bell_sigma:        float = 1.5,
        seg_chunk:         int = 16,
        temporal_context:  bool = True,
        visibility_gate:   bool = True,
    ):
        super().__init__()
        self.encoder         = encoder
        self.k               = k
        self.seg_chunk       = seg_chunk
        self.temporal_context = temporal_context
        self.visibility_gate = visibility_gate
        self.register_buffer(
            "bell", bell_weights_window(k, sigma=bell_sigma),
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        D = self.encoder.out_dim
        base = D * (3 if self.temporal_context else 1)
        return base + (1 if self.visibility_gate else 0)

    def forward(
        self,
        frames: torch.Tensor,        # [B, T, 3, H, W] normalised
        tip_xy: torch.Tensor,        # [B, T, 2]
        vis:    torch.Tensor,        # [B, T]
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        G = self.encoder.grid_size
        D = self.encoder.out_dim

        # 1) Flatten and run DINOv2 patches.
        flat = frames.reshape(B * T, C, H, W)
        # Honour seg_chunk to bound activation memory.
        outs: list[torch.Tensor] = []
        cs = self.seg_chunk
        for s in range(0, flat.size(0), cs):
            outs.append(self.encoder.forward_patches(flat[s: s + cs]))
        patches = torch.cat(outs, dim=0)                # [B*T, G*G, D]
        patches_2d = patches.view(B, T, G, G, D)

        # 2) Per-frame windowed bell pool.
        per_t = torch.empty(B, T, D, dtype=patches.dtype, device=patches.device)
        for b in range(B):
            for t in range(T):
                per_t[b, t] = fingertip_pool_window(
                    patches_2d[b, t], tip_xy[b, t],
                    k=self.k, weights=self.bell,
                )

        # 3) Optional +-1 temporal context.
        if self.temporal_context:
            prev = torch.cat([per_t[:, :1], per_t[:, :-1]], dim=1)
            nxt  = torch.cat([per_t[:, 1:], per_t[:, -1:]], dim=1)
            ctx  = torch.cat([prev, per_t, nxt], dim=-1)  # [B, T, 3D]
        else:
            ctx = per_t

        # 4) Visibility gate.
        if self.visibility_gate:
            ctx = ctx * vis.unsqueeze(-1)
            return torch.cat([ctx, vis.unsqueeze(-1)], dim=-1)
        return ctx


# ---------------------------------------------------------------------------
# Greedy CTC decode (shared with Stage 9a)
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
# Per-fold (1-fold!) trainer
# ---------------------------------------------------------------------------

def train_stage10(
    train_clip_cache: _Stage10ClipCache,
    val_clip_cache:   _Stage10ClipCache,
    *,
    cfg,
    fold:             int = 0,
    variant:          str = "stage10_d2u2_w3",
    num_epochs:       int = 60,
    batch_size:       int = 16,
    grad_clip:        float = 1.0,
    dropout:          float = 0.2,
    d_model:          int = 256,
    n_layers:         int = 4,
    n_heads:          int = 4,
    conv_kernel:      int = 15,
    upsample:         int = 2,
    warmup_pct:       float = 0.05,
    dec_n_layers:     int = 3,
    dec_n_heads:      int = 4,
    lambda_ctc:       float = 0.3,
    # Stage 10 specific knobs:
    unfreeze_last_n:  int = 2,
    window_k:         int = 3,
    bell_sigma:       float = 1.5,
    lr_dinov2:        float = 5e-6,
    lr_head:          float = 5e-4,
    weight_decay:     float = 5e-2,
    seg_chunk:        int = 16,
    seed:             int = 42,
    checkpoint_dir:   str = "/kaggle/working/checkpoints",
    log_dir:          str = "/kaggle/working/logs",
) -> dict:
    """
    1-fold gate trainer for Stage 10.

    train_clip_cache : pre-extracted fold-0 train subset (in RAM).
    val_clip_cache   : pre-extracted fold-0 val (FULL 720 clips).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device      = cfg.device
    converter   = make_converter(cfg.data.lang)
    pad_idx     = cfg.vocab.pad_idx
    blank       = cfg.vocab.blank_idx
    att_V       = cfg.vocab.attn_vocab_size
    sos_idx     = cfg.vocab.sos_idx
    eos_idx     = cfg.vocab.eos_idx

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)

    # -- data --
    train_ds = _Stage10Dataset(train_clip_cache,
                               list(range(len(train_clip_cache))),
                               converter, augment=False)
    val_ds   = _Stage10Dataset(val_clip_cache,
                               list(range(len(val_clip_cache))),
                               converter, augment=False)
    coll = lambda b: _collate(b, pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=coll, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=coll,
    )

    # -- model --
    dinov2 = DINOv2Encoder(
        model_name="facebook/dinov2-small",
        image_size=336,
        pool="mean_patch",         # unused on patch path
        unfreeze_last_n=unfreeze_last_n,
    ).to(device)
    feature_module = Stage10FeatureModule(
        encoder=dinov2, k=window_k, bell_sigma=bell_sigma,
        seg_chunk=seg_chunk,
        temporal_context=True, visibility_gate=True,
    ).to(device)
    in_dim = feature_module.out_dim         # 3*384 + 1 = 1153

    encoder = ConformerCTC(
        input_dim       = in_dim,
        vocab_size      = cfg.vocab.ctc_vocab_size,
        d_model         = d_model,
        n_layers        = n_layers,
        n_heads         = n_heads,
        conv_kernel     = conv_kernel,
        dropout         = dropout,
        upsample        = upsample,
        input_layernorm = True,             # Stage 3 contract for non-landmark input
    ).to(device)
    decoder = AttentionDecoder(
        att_vocab_size = att_V, bos_idx = sos_idx, eos_idx = eos_idx,
        d_model        = d_model, n_layers = dec_n_layers, n_heads = dec_n_heads,
        ff_mult        = 4, dropout = dropout,
    ).to(device)

    # -- two LR groups: DINOv2 unfrozen blocks at low LR, head at full LR --
    dinov2_trainable = [p for p in dinov2.parameters() if p.requires_grad]
    head_params = (
        list(encoder.parameters()) + list(decoder.parameters())
    )
    param_groups = []
    if dinov2_trainable:
        param_groups.append({"params": dinov2_trainable, "lr": lr_dinov2,
                             "name": "dinov2_unfrozen"})
    param_groups.append({"params": head_params, "lr": lr_head,
                         "name": "head"})
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=weight_decay, betas=(0.9, 0.999),
    )
    total_steps = num_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_dinov2, lr_head] if dinov2_trainable else lr_head,
        total_steps=total_steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    ce  = nn.CrossEntropyLoss(ignore_index=pad_idx)

    history: list[dict] = []
    best_cer = float("inf"); best_epoch = -1
    best_per_signer: dict[str, float] = {}
    ckpt_path = os.path.join(
        checkpoint_dir, f"stage10_fold{fold}_{variant}_best.pt",
    )
    print(
        f"\n=== Stage 10 fold={fold}  variant={variant}  "
        f"unfreeze_last_n={unfreeze_last_n}  window_k={window_k}  "
        f"in_dim={in_dim}  train={len(train_clip_cache)}  "
        f"val={len(val_clip_cache)} ===",
        flush=True,
    )
    n_dinov2_trainable = sum(p.numel() for p in dinov2_trainable)
    n_head             = sum(p.numel() for p in head_params)
    print(f"  trainable params: dinov2={n_dinov2_trainable:,}  head={n_head:,}",
          flush=True)

    for epoch in range(num_epochs):
        feature_module.train(); encoder.train(); decoder.train()
        sum_ctc = sum_attn = sum_total = 0.0
        n_batches = 0
        t0 = time.time()

        for frames, tip_xy, vis, labels, in_lens, lab_lens, _ in train_loader:
            frames   = frames.to(device);   tip_xy  = tip_xy.to(device)
            vis      = vis.to(device);      labels  = labels.to(device)
            in_lens  = in_lens.to(device);  lab_lens = lab_lens.to(device)

            feats = feature_module(frames, tip_xy, vis)      # [B, T, 1153]
            h, pad_mask = encoder.encode(feats, in_lens)
            log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
            ctc_loss = ctc(log_probs.transpose(0, 1).float(),
                           labels, enc_lens, lab_lens)
            dec_in, dec_tg = build_attention_targets(
                labels, lab_lens, bos=sos_idx, eos=eos_idx, pad=pad_idx,
            )
            dec_logits = decoder(h, pad_mask, dec_in)
            attn_loss  = ce(dec_logits.reshape(-1, decoder.att_vocab_size),
                            dec_tg.reshape(-1))
            total = lambda_ctc * ctc_loss + (1 - lambda_ctc) * attn_loss
            optimizer.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], grad_clip,
            )
            optimizer.step(); scheduler.step()
            sum_ctc   += float(ctc_loss.item())
            sum_attn  += float(attn_loss.item())
            sum_total += float(total.item())
            n_batches += 1
        train_ctc, train_attn, train_total = (
            sum_ctc / max(n_batches, 1),
            sum_attn / max(n_batches, 1),
            sum_total / max(n_batches, 1),
        )

        # -- val (greedy CTC + greedy attention, best-of-both per clip) --
        feature_module.eval(); encoder.eval(); decoder.eval()
        pairs_best: list[tuple[str, str]] = []
        per_subj_err: dict[str, int] = defaultdict(int)
        per_subj_len: dict[str, int] = defaultdict(int)
        with torch.no_grad():
            for frames, tip_xy, vis, labels, in_lens, lab_lens, subjs in val_loader:
                frames   = frames.to(device);   tip_xy  = tip_xy.to(device)
                vis      = vis.to(device);      labels  = labels.to(device)
                in_lens  = in_lens.to(device);  lab_lens = lab_lens.to(device)
                feats = feature_module(frames, tip_xy, vis)
                h, pad_mask = encoder.encode(feats, in_lens)
                log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
                ctc_preds = _ctc_greedy(log_probs, enc_lens, blank)
                attn_preds = decoder.greedy_decode(h, pad_mask)
                for b in range(len(ctc_preds)):
                    gt = converter.decode(
                        labels[b, : int(lab_lens[b].item())].int().cpu(),
                        torch.IntTensor([int(lab_lens[b].item())]),
                    )
                    p_ctc  = _ids_to_str(ctc_preds[b], cfg.vocab.chars)
                    p_attn = _ids_to_str(attn_preds[b].tolist(), cfg.vocab.chars)
                    e_ctc  = editdistance.eval(gt, p_ctc)
                    e_attn = editdistance.eval(gt, p_attn)
                    e_best = min(e_ctc, e_attn)
                    pairs_best.append((gt, p_ctc if e_ctc <= e_attn else p_attn))
                    per_subj_err[subjs[b]] += int(e_best)
                    per_subj_len[subjs[b]] += int(len(gt))

        total_err = sum(editdistance.eval(g, p) for g, p in pairs_best)
        total_len = max(sum(len(g) for g, _ in pairs_best), 1)
        cer_best = total_err / total_len
        per_signer_cer = {s: per_subj_err[s] / max(per_subj_len[s], 1)
                          for s in per_subj_err}

        dt = time.time() - t0
        print(
            f"[F{fold} {variant}] Ep {epoch+1:3d}/{num_epochs}  "
            f"ctc={train_ctc:.4f} attn={train_attn:.4f} total={train_total:.4f}  "
            f"val CER={cer_best:.4f}  {dt:.0f}s",
            flush=True,
        )

        history.append({
            "epoch":            epoch + 1,
            "train_ctc":        train_ctc,
            "train_attn":       train_attn,
            "train_total":      train_total,
            "val_cer_best":     cer_best,
            "per_signer_val_cer": per_signer_cer,
        })
        if cer_best < best_cer:
            best_cer = cer_best
            best_epoch = epoch + 1
            best_per_signer = dict(per_signer_cer)
            torch.save({
                "fold": fold, "variant": variant,
                "dinov2_state_dict":  dinov2.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch": epoch, "val_cer": best_cer,
                "lambda_ctc": lambda_ctc, "window_k": window_k,
                "unfreeze_last_n": unfreeze_last_n,
            }, ckpt_path)
            print(f"    ★ new best CER={best_cer:.4f}", flush=True)

    result = {
        "fold":                       fold,
        "variant":                    variant,
        "num_epochs":                 num_epochs,
        "n_train_clips":              len(train_clip_cache),
        "n_val_clips":                len(val_clip_cache),
        "best_val_cer":               best_cer,
        "best_epoch":                 best_epoch,
        "final_train_ctc_nll":        train_ctc,
        "final_train_attn_nll":       train_attn,
        "lambda_ctc":                 lambda_ctc,
        "window_k":                   window_k,
        "unfreeze_last_n":            unfreeze_last_n,
        "lr_dinov2":                  lr_dinov2,
        "lr_head":                    lr_head,
        "seed":                       seed,
        "best_per_signer_val_cer":    best_per_signer,
        "history":                    history,
        "checkpoint_path":            ckpt_path,
    }
    json_path = os.path.join(
        log_dir, f"stage10_fold{fold}_{variant}_history.json",
    )
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)
    return result

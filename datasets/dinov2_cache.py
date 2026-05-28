"""
datasets/dinov2_cache.py — DINOv2 per-frame feature extraction + cache.

Stage 2 of the iterative ablation plan.

Pipeline per clip
-----------------
  PNG bytes
    → PIL.Image
    → MediaPipe hand crop (reuse existing HandCropper)  [optional]
    → resize 224×224
    → ImageNet normalize
    → DINOv2 per-frame forward → [384]
    → resample sequence to T_native frames → [T_native, 384]

Cache layout — IDENTICAL keys to skeleton_cache.py so the SkeletonDataset
can plug in directly with no changes:
  {
    "feats":     list[Tensor [T_native, 384]]  fp16 on CPU
    "labels":    list[str]
    "subjects":  list[str]
    "lengths":   LongTensor [N]   (T_native for all)
    "out_dim":   int              (384)
    "n_native":  int              (T_native)
    "model_name": str
    "image_size": int
    "pool":      str              ("mean_patch")
    "hand_crop": bool             (whether HandCropper was used)
  }
"""

from __future__ import annotations

import io
import os
import time
import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def _pil_to_dinov2_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    """PIL → [3, H, W] float32, ImageNet-normalized."""
    from ..models.encoders.dinov2_encoder import default_normalize
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    t = TF.to_tensor(img)
    return default_normalize(t)


def _resample_uniform(arr: torch.Tensor, T_target: int) -> torch.Tensor:
    """
    Linear-interpolation resample of [T_in, D] tensor to [T_target, D].
    Same behavior as skeleton_cache._resample_uniform.
    """
    T_in = arr.shape[0]
    if T_in == T_target:
        return arr
    if T_in < 2:
        return arr.expand(T_target, *arr.shape[1:]).clone()
    src_idx = torch.linspace(0, T_in - 1, T_target)
    src_lo  = src_idx.floor().long()
    src_hi  = (src_lo + 1).clamp(max=T_in - 1)
    frac    = (src_idx - src_lo.float()).unsqueeze(-1)
    return arr[src_lo] * (1 - frac) + arr[src_hi] * frac


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_dinov2_features(
    samples:      list[tuple],
    encoder,                      # DINOv2Encoder instance
    out_path:     str,
    T_native:     int = 32,
    hand_cropper = None,          # optional HandCropper instance
    seg_chunk:    int = 16,       # frames per DINOv2 forward
    device:       str | torch.device = "cuda",
    dtype:        torch.dtype = torch.float16,
) -> dict:
    """
    Build a DINOv2 feature cache compatible with the existing SkeletonDataset.

    Parameters
    ----------
    samples      : list of (frame_bytes, label, subject_id) — from
                   subject_splits.stream_and_index_with_subjects()
    encoder      : DINOv2Encoder (frozen).
    out_path     : .pt cache path.
    T_native     : uniform sequence length to resample every clip to.
    hand_cropper : optional HandCropper.  If provided, every clip is
                   spatially cropped to the hand bbox BEFORE DINOv2.
    seg_chunk    : DINOv2 batch size in frames.
    """
    encoder = encoder.to(device).eval()
    image_size = encoder.image_size
    D = encoder.out_dim

    if hand_cropper is not None and hand_cropper.target_size != image_size:
        logger.warning(
            "hand_cropper.target_size=%d != encoder.image_size=%d; overriding to match.",
            hand_cropper.target_size, image_size,
        )
        hand_cropper.target_size = image_size

    feats_list:    list[torch.Tensor] = []
    labels_list:   list[str]          = []
    subjects_list: list[str]          = []
    lengths_list:  list[int]          = []

    total = len(samples)
    log_every = max(1, total // 100)
    t0 = time.time()
    skipped = 0

    print(
        f"[dinov2_cache] Encoding {total} clips with {encoder.model_name} "
        f"on {device} (image_size={image_size}, T_native={T_native}, "
        f"hand_crop={hand_cropper is not None}, dtype={dtype})",
        flush=True,
    )

    for ci, item in enumerate(samples):
        if len(item) == 3:
            frame_bytes, label, subject = item
        else:
            frame_bytes, label = item
            subject = "UNKNOWN"

        if not frame_bytes:
            skipped += 1
            continue

        try:
            pil_frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
        except Exception as e:
            logger.warning("Skipping clip %d (%s): decode error %s", ci, subject, e)
            skipped += 1
            continue

        # Optional hand-region crop (reuse Stage-2 spatial focus).
        if hand_cropper is not None:
            pil_frames = hand_cropper.crop_clip(pil_frames)
            # crop_clip already resized to image_size.

        # Normalize per-frame
        frame_tensors = [_pil_to_dinov2_tensor(f, image_size) for f in pil_frames]
        clip = torch.stack(frame_tensors, dim=0).to(device, non_blocking=True)

        # Per-frame DINOv2 forward
        feats = encoder.forward_clip(clip, chunk_size=seg_chunk)   # [T_raw, 384]

        # Resample to T_native
        feats_resampled = _resample_uniform(feats.cpu(), T_native)  # [T_native, 384]

        feats_resampled = feats_resampled.to(dtype).contiguous()
        feats_list.append(feats_resampled)
        labels_list.append(label)
        subjects_list.append(subject)
        lengths_list.append(T_native)

        if (ci + 1) % log_every == 0 or (ci + 1) == total:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total - (ci + 1)) / max(rate, 1e-3)
            extra = ""
            if hand_cropper is not None:
                s = hand_cropper.stats()
                extra = f"  hand_det={s['frame_detect_rate']*100:.1f}%"
            print(
                f"[dinov2_cache] {ci + 1}/{total}  "
                f"({100*(ci+1)/total:5.1f}%)  "
                f"{rate:.1f} clips/s  ETA {eta/60:5.1f} min  "
                f"skipped={skipped}{extra}",
                flush=True,
            )

    cache = {
        "feats":      feats_list,
        "labels":     labels_list,
        "subjects":   subjects_list,
        "lengths":    torch.tensor(lengths_list, dtype=torch.long),
        "out_dim":    D,
        "n_native":   T_native,
        "model_name": encoder.model_name,
        "image_size": image_size,
        "pool":       encoder.pool,
        "hand_crop":  hand_cropper is not None,
    }
    if hand_cropper is not None:
        cache["hand_crop_stats"] = hand_cropper.stats()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    print(f"[dinov2_cache] Saved {len(feats_list)} clips → {out_path}  "
          f"({mb:.1f} MB, dtype={dtype})", flush=True)
    return cache

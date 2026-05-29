"""
datasets/dinov2_fingertip_cache.py — Stage 3 cache: DINOv2 patches mean-pooled
over a 2×2 grid centered on the MediaPipe fingertip.

Why this is the Stage 3 fix
---------------------------
Stage 2 (mean-pool over all 256 patches) got CER 0.86 — substantially worse
than Stage 1 landmarks (0.69).  The diagnosis: mean-pool destroys spatial
focus.  256 patches each encode a different ~14×14-pixel region of the
hand crop; averaging them dilutes the writing-region signal with patches
covering wrist, palm, background.

Per the iterative-ablation plan, Stage 3 swaps mean-pool for a 2×2 patch
window centered on the fingertip:
  1. Detect MediaPipe hand landmarks per frame  → 21 (x,y,z)
  2. Take INDEX_FINGER_TIP (joint 8) for fingertip position
  3. Compute the hand bbox (union over frames, padded) and crop the frame
  4. Transform fingertip coords to the cropped-frame reference
  5. Map cropped fingertip (x,y) ∈ [0,1] to a (row, col) in the 16×16 DINOv2
     token grid
  6. Take the 2×2 patch grid whose top-left contains the fingertip and
     mean-pool those 4 tokens → 384-d
  7. Resample sequence to T_native = 32

Fallback: if MediaPipe finds no hand in a frame, use the previous valid
fingertip; if no previous, use the center of the grid.

Cache layout: same keys as skeleton_cache.py and dinov2_cache.py, so the
SkeletonDataset / FeatureDataset in the existing notebook just works.
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

from .skeleton_cache import LandmarkExtractor, N_JOINTS

logger = logging.getLogger(__name__)


# MediaPipe hand landmark indices (https://developers.google.com/mediapipe)
INDEX_FINGER_TIP = 8


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _detect_bbox_from_landmarks(
    lms_all_frames: list[Optional[np.ndarray]],
    frame_size:     tuple[int, int],
    padding_ratio:  float = 0.3,
) -> tuple[int, int, int, int]:
    """
    Union bbox over all valid landmark detections, padded and squared up.
    Mirrors HandCropper's logic.

    Returns (x0, y0, x1, y1) in pixel coords of the original frame.
    """
    W, H = frame_size
    valid = [lm for lm in lms_all_frames if lm is not None]
    if not valid:
        # No hand seen at all — center square fallback.
        side = min(W, H)
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        return (x0, y0, x0 + side, y0 + side)

    xs_all = np.concatenate([lm[:, 0] for lm in valid])
    ys_all = np.concatenate([lm[:, 1] for lm in valid])
    x0 = max(0.0, xs_all.min()) * W
    y0 = max(0.0, ys_all.min()) * H
    x1 = min(1.0, xs_all.max()) * W
    y1 = min(1.0, ys_all.max()) * H

    # Pad
    bw, bh = (x1 - x0), (y1 - y0)
    px = bw * padding_ratio
    py = bh * padding_ratio
    x0 -= px; x1 += px; y0 -= py; y1 += py

    # Square-up around center
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0)
    half = side * 0.5
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    # Clip to image
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(W, int(x1))
    y1 = min(H, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        side = min(W, H)
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        return (x0, y0, x0 + side, y0 + side)
    return (x0, y0, x1, y1)


def _fingertip_in_cropped(
    fingertip_norm: tuple[float, float],
    bbox:           tuple[int, int, int, int],
    frame_size:     tuple[int, int],
) -> tuple[float, float]:
    """
    Transform a fingertip position from full-frame normalized coords
    [0, 1] to cropped-frame normalized coords [0, 1].
    """
    W, H = frame_size
    fx, fy = fingertip_norm
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = (fx * W - x0) / bw
    cy = (fy * H - y0) / bh
    # Clamp to [0, 1] — if the fingertip happens to fall outside the crop
    # (rare due to padding) we just take the nearest edge patch.
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    return (cx, cy)


def _fingertip_patch_indices(
    fingertip_xy: tuple[float, float],
    grid_size:    int,
    window:       int = 2,
) -> list[int]:
    """
    Map cropped fingertip (x, y) ∈ [0, 1] to a `window × window` block of
    patch indices in the 16×16 token grid.  The 2×2 block is the one whose
    top-left corner is just above-and-left of the fingertip position.

    Token index in the un-CLSed sequence: row * grid_size + col.

    Examples (grid=16, window=2)
    ----------------------------
    fingertip at (0.5, 0.5) → r0=7, c0=7 → patches (7,7), (7,8), (8,7), (8,8)
    fingertip at (0.0, 0.0) → r0=0, c0=0 → top-left 2×2
    """
    if window < 1:
        window = 1
    pr = fingertip_xy[1] * grid_size       # row coord in patch units
    pc = fingertip_xy[0] * grid_size       # col coord in patch units

    # Place the 2×2 block so the fingertip falls inside it.
    half = window / 2.0
    r0 = int(round(pr - half))
    c0 = int(round(pc - half))
    r0 = max(0, min(grid_size - window, r0))
    c0 = max(0, min(grid_size - window, c0))

    indices: list[int] = []
    for dr in range(window):
        for dc in range(window):
            indices.append((r0 + dr) * grid_size + (c0 + dc))
    return indices


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def _crop_resize(
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    image_size: int,
) -> Image.Image:
    cropped = img.crop(bbox)
    if cropped.size != (image_size, image_size):
        cropped = cropped.resize((image_size, image_size), Image.BILINEAR)
    return cropped


def _pil_to_dinov2_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    from ..models.encoders.dinov2_encoder import default_normalize
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    t = TF.to_tensor(img)
    return default_normalize(t)


def _resample_uniform(arr: torch.Tensor, T_target: int) -> torch.Tensor:
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
def extract_dinov2_fingertip_features(
    samples:       list[tuple],
    encoder,                          # DINOv2Encoder
    out_path:      str,
    T_native:      int = 32,
    padding_ratio: float = 0.3,
    window:        int = 2,           # 2×2 patch pool
    seg_chunk:     int = 16,
    device:        str | torch.device = "cuda",
    dtype:         torch.dtype = torch.float16,
) -> dict:
    """
    Build a Stage-3 feature cache.

    Per-clip pipeline
    -----------------
      1. Per-frame MediaPipe → 21 landmarks
      2. Union bbox over valid landmarks
      3. Crop+resize all frames to encoder.image_size
      4. DINOv2 forward → [T, N_patches, D] patch tokens
      5. For each frame: get fingertip-in-cropped-coords (use previous
         valid if missing; else center), map to 2×2 patch indices,
         mean-pool → [D]
      6. Resample sequence to T_native, store as fp16
    """
    encoder = encoder.to(device).eval()
    image_size = encoder.image_size
    grid       = encoder.grid_size               # 16 for 224 / 14
    D          = encoder.out_dim

    lm_extractor = LandmarkExtractor(max_num_hands=1)

    feats_list:    list[torch.Tensor] = []
    labels_list:   list[str]          = []
    subjects_list: list[str]          = []
    lengths_list:  list[int]          = []
    detect_frames_total    = 0
    detect_frames_detected = 0
    fallback_count         = 0

    total = len(samples)
    log_every = max(1, total // 100)
    t0 = time.time()
    skipped = 0

    print(
        f"[dinov2_fingertip_cache] Encoding {total} clips with {encoder.model_name} "
        f"on {device} (image_size={image_size}, grid={grid}, window={window}, "
        f"T_native={T_native}, dtype={dtype})",
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

        # 1. Per-frame landmark detection on FULL frame
        landmarks_per_frame: list[Optional[np.ndarray]] = [
            lm_extractor.detect(f) for f in pil_frames
        ]
        n_det = sum(1 for lm in landmarks_per_frame if lm is not None)
        detect_frames_total    += len(pil_frames)
        detect_frames_detected += n_det

        # 2. Union bbox
        bbox = _detect_bbox_from_landmarks(
            landmarks_per_frame, pil_frames[0].size,
            padding_ratio=padding_ratio,
        )

        # 3. Crop+resize all frames; build tensor batch
        cropped_pils = [_crop_resize(f, bbox, image_size) for f in pil_frames]
        tensors = torch.stack(
            [_pil_to_dinov2_tensor(c, image_size) for c in cropped_pils],
            dim=0,
        ).to(device, non_blocking=True)                  # [T_raw, 3, H, W]

        # 4. DINOv2 patch tokens
        patches = encoder.forward_patches_clip(tensors, chunk_size=seg_chunk)
        # patches: [T_raw, N_patches, D]

        # 5. Per-frame fingertip-region pool
        last_valid_xy: Optional[tuple[float, float]] = None
        per_frame_feats: list[torch.Tensor] = []
        for t, lm in enumerate(landmarks_per_frame):
            if lm is not None:
                fingertip_norm = (float(lm[INDEX_FINGER_TIP, 0]),
                                  float(lm[INDEX_FINGER_TIP, 1]))
                xy_crop = _fingertip_in_cropped(
                    fingertip_norm, bbox, pil_frames[t].size,
                )
                last_valid_xy = xy_crop
            elif last_valid_xy is not None:
                xy_crop = last_valid_xy
                fallback_count += 1
            else:
                xy_crop = (0.5, 0.5)        # center fallback
                fallback_count += 1

            idx_list = _fingertip_patch_indices(xy_crop, grid_size=grid, window=window)
            selected = patches[t, idx_list]                 # [window^2, D]
            pooled = selected.mean(dim=0)                   # [D]
            per_frame_feats.append(pooled)

        clip_feats = torch.stack(per_frame_feats, dim=0).cpu()   # [T_raw, D]

        # 6. Resample sequence to T_native
        clip_feats = _resample_uniform(clip_feats, T_native)     # [T_native, D]
        clip_feats = clip_feats.to(dtype).contiguous()

        feats_list.append(clip_feats)
        labels_list.append(label)
        subjects_list.append(subject)
        lengths_list.append(T_native)

        if (ci + 1) % log_every == 0 or (ci + 1) == total:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total - (ci + 1)) / max(rate, 1e-3)
            det_rate = detect_frames_detected / max(detect_frames_total, 1) * 100
            print(
                f"[dinov2_fingertip_cache] {ci + 1}/{total}  "
                f"({100*(ci+1)/total:5.1f}%)  "
                f"{rate:.1f} clips/s  ETA {eta/60:5.1f} min  "
                f"detect_rate={det_rate:.1f}%  fallback_frames={fallback_count}  "
                f"skipped={skipped}",
                flush=True,
            )

    lm_extractor.close()

    cache = {
        "feats":      feats_list,
        "labels":     labels_list,
        "subjects":   subjects_list,
        "lengths":    torch.tensor(lengths_list, dtype=torch.long),
        "out_dim":    D,
        "n_native":   T_native,
        "model_name": encoder.model_name,
        "image_size": image_size,
        "pool":       f"fingertip_{window}x{window}",
        "hand_crop":  True,
        "frame_detect_rate":
            detect_frames_detected / max(detect_frames_total, 1),
        "fallback_frames_total": fallback_count,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    print(
        f"[dinov2_fingertip_cache] Saved {len(feats_list)} clips → {out_path}  "
        f"({mb:.1f} MB, dtype={dtype}, detect_rate="
        f"{detect_frames_detected/max(detect_frames_total,1)*100:.1f}%, "
        f"fallback_frames={fallback_count})",
        flush=True,
    )
    return cache

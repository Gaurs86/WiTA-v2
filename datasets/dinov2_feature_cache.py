"""
datasets/dinov2_feature_cache.py — Stage 3 (strengthened) feature cache.

Per-clip pipeline
-----------------
  1. Decode raw frames from the streamed clip bytes.
  2. Run MediaPipe HandLandmarker (same params as Stage 1) -> 21 (x,y,z)
     per frame, or None if not detected.
  3. Union-bbox over valid detections; pad and square it up.
  4. Crop-resize every frame to encoder.image_size x encoder.image_size
     (336x336 in the headline Stage 3 config).
  5. DINOv2 forward -> [T_raw, G*G, D] patch tokens (G = image_size /
     patch_size; D = 384 for dinov2-s).
  6. Per frame: transform fingertip (joint 8) to cropped-frame coords,
     run bell-weighted 3x3 patch pool.  Frames without detections get a
     zero vector AND visibility flag 0.
  7. Build temporal-context concat [t-1, t, t+1] -> 3*D channels.
  8. Append visibility column -> 3*D + 1 channels (visibility gate).
  9. Resample sequence to T_native = 32 -> store as fp16.

Cache layout (compatible with the existing CachedFeatureDataset)
----------------------------------------------------------------
  feats:       list[Tensor [T_native, 3*D+1]] fp16
  labels:      list[str]
  subjects:    list[str]
  lengths:     LongTensor [N]
  out_dim:     int                          3*D + 1
  n_native:    int
  fingerprint: dict {
      model_name, image_size, grid_size, patch_size,
      pool: "fingertip_3x3_bell",
      temporal_context: True,
      visibility_gate:  True,
      multi_joint:      False,
  }
  per_clip_visibility:  list[float]         visibility rate per clip
  frame_detect_rate:    float               global rate

Refuse to load when fingerprint disagrees with the requested config.
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

from .skeleton_cache import LandmarkExtractor
from .dinov2_fingertip_cache import (
    _detect_bbox_from_landmarks,
    _fingertip_in_cropped,
    _crop_resize,
    _pil_to_dinov2_tensor,
    _resample_uniform,
)
from ..models.encoders.dinov2_fingertip_extractor import (
    build_clip_features_stage3,
    expected_out_dim,
    INDEX_FINGER_TIP,
    ALL_FINGERTIPS,
)

logger = logging.getLogger(__name__)


def make_fingerprint(
    encoder,
    *,
    temporal_context: bool,
    visibility_gate:  bool,
    multi_joint:      bool,
    pool:             str = "fingertip_3x3_bell",
) -> dict:
    return {
        "model_name":       encoder.model_name,
        "image_size":       encoder.image_size,
        "grid_size":        encoder.grid_size,
        "patch_size":       encoder.patch_size,
        "pool":             pool,
        "temporal_context": temporal_context,
        "visibility_gate":  visibility_gate,
        "multi_joint":      multi_joint,
    }


def check_fingerprint_match(cache: dict, expected: dict) -> None:
    got = cache.get("fingerprint", {})
    mismatches = {k: (got.get(k), expected[k]) for k in expected
                  if got.get(k) != expected[k]}
    if mismatches:
        raise RuntimeError(
            "Stage 3 cache fingerprint mismatch — refusing to load.\n"
            f"  cache.fingerprint = {got}\n"
            f"  requested         = {expected}\n"
            f"  mismatches        = {mismatches}"
        )


@torch.no_grad()
def extract_dinov2_feature_cache(
    samples:          list[tuple],
    encoder,                              # DINOv2Encoder, already on device
    out_path:         str,
    *,
    T_native:         int = 32,
    padding_ratio:    float = 0.3,
    temporal_context: bool = True,
    visibility_gate:  bool = True,
    multi_joint:      bool = False,
    bell_sigma:       float = 1.0,
    seg_chunk:        int = 16,
    device:           str | torch.device = "cuda",
    dtype:            torch.dtype = torch.float16,
) -> dict:
    """
    Build the Stage-3 cache.  See module docstring for the pipeline.

    multi_joint=True swaps the single-fingertip pool for a 5-fingertip
    concatenation (THUMB / INDEX / MIDDLE / RING / PINKY).
    """
    encoder = encoder.to(device).eval()
    image_size = encoder.image_size
    grid       = encoder.grid_size
    D          = encoder.out_dim
    out_dim    = expected_out_dim(
        D, temporal_context=temporal_context,
        visibility_gate=visibility_gate, multi_joint=multi_joint,
    )

    fp = make_fingerprint(
        encoder,
        temporal_context=temporal_context,
        visibility_gate=visibility_gate,
        multi_joint=multi_joint,
    )

    lm_extractor = LandmarkExtractor(max_num_hands=1)

    feats_list:    list[torch.Tensor] = []
    labels_list:   list[str]          = []
    subjects_list: list[str]          = []
    lengths_list:  list[int]          = []
    per_clip_vis:  list[float]        = []
    detect_frames_total    = 0
    detect_frames_detected = 0
    skipped                = 0

    total     = len(samples)
    log_every = max(1, total // 100)
    t0        = time.time()

    print(
        f"[dinov2_feature_cache] Stage-3 cache: {total} clips, "
        f"{encoder.model_name} at {image_size}x{image_size} -> "
        f"grid {grid}x{grid}, D={D}, out_dim={out_dim}, T_native={T_native}, "
        f"temporal_context={temporal_context}, visibility_gate={visibility_gate}, "
        f"multi_joint={multi_joint}",
        flush=True,
    )

    for ci, item in enumerate(samples):
        if len(item) == 3:
            frame_bytes, label, subject = item
        else:
            frame_bytes, label = item
            subject = "UNKNOWN"
        if not frame_bytes:
            skipped += 1; continue

        try:
            pil_frames = [Image.open(io.BytesIO(b)).convert("RGB")
                          for b in frame_bytes]
        except Exception as e:
            logger.warning("Skipping clip %d (%s): decode error %s",
                           ci, subject, e)
            skipped += 1; continue

        landmarks_per_frame: list[Optional[np.ndarray]] = [
            lm_extractor.detect(f) for f in pil_frames
        ]
        n_det = sum(1 for lm in landmarks_per_frame if lm is not None)
        detect_frames_total    += len(pil_frames)
        detect_frames_detected += n_det

        bbox = _detect_bbox_from_landmarks(
            landmarks_per_frame, pil_frames[0].size,
            padding_ratio=padding_ratio,
        )

        cropped_pils = [_crop_resize(f, bbox, image_size) for f in pil_frames]
        tensors = torch.stack(
            [_pil_to_dinov2_tensor(c, image_size) for c in cropped_pils],
            dim=0,
        ).to(device, non_blocking=True)                       # [T_raw, 3, H, W]

        patches = encoder.forward_patches_clip(tensors, chunk_size=seg_chunk)
        # patches: [T_raw, G*G, D]

        # Build per-frame tip lists.
        tip_xy_per_t: list[Optional[tuple[float, float]]] = []
        multi_joint_xy_per_t: list[list[Optional[tuple[float, float]]]] = []
        for t, lm in enumerate(landmarks_per_frame):
            if lm is None:
                tip_xy_per_t.append(None)
                multi_joint_xy_per_t.append([None] * 5)
            else:
                xy_idx = _fingertip_in_cropped(
                    (float(lm[INDEX_FINGER_TIP, 0]), float(lm[INDEX_FINGER_TIP, 1])),
                    bbox, pil_frames[t].size,
                )
                tip_xy_per_t.append(xy_idx)
                joint_list = []
                for j in ALL_FINGERTIPS:
                    joint_list.append(_fingertip_in_cropped(
                        (float(lm[j, 0]), float(lm[j, 1])),
                        bbox, pil_frames[t].size,
                    ))
                multi_joint_xy_per_t.append(joint_list)

        # Run the Stage-3 builder.
        clip_feats = build_clip_features_stage3(
            patches.float(),
            tip_xy_per_t,
            grid_size=grid,
            visibility_gate=visibility_gate,
            temporal_context=temporal_context,
            bell_sigma=bell_sigma,
            multi_joint_tip_xy=(multi_joint_xy_per_t if multi_joint else None),
        ).cpu()                                                # [T_raw, out_dim]

        # Resample to T_native.
        clip_feats = _resample_uniform(clip_feats, T_native).to(dtype).contiguous()
        feats_list.append(clip_feats)
        labels_list.append(label)
        subjects_list.append(subject)
        lengths_list.append(T_native)
        per_clip_vis.append(n_det / max(len(pil_frames), 1))

        if (ci + 1) % log_every == 0 or (ci + 1) == total:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total - (ci + 1)) / max(rate, 1e-3)
            det_rate = detect_frames_detected / max(detect_frames_total, 1) * 100
            print(
                f"[dinov2_feature_cache] {ci+1}/{total}  "
                f"({100*(ci+1)/total:5.1f}%)  "
                f"{rate:.2f} clips/s  ETA {eta/60:5.1f} min  "
                f"detect_rate={det_rate:.1f}%  skipped={skipped}",
                flush=True,
            )

    lm_extractor.close()

    cache = {
        "feats":               feats_list,
        "labels":              labels_list,
        "subjects":            subjects_list,
        "lengths":             torch.tensor(lengths_list, dtype=torch.long),
        "out_dim":             out_dim,
        "n_native":            T_native,
        "fingerprint":         fp,
        "per_clip_visibility": per_clip_vis,
        "frame_detect_rate":
            detect_frames_detected / max(detect_frames_total, 1),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    print(
        f"[dinov2_feature_cache] Saved {len(feats_list)} clips -> {out_path}  "
        f"({mb:.1f} MB, dtype={dtype}, detect_rate="
        f"{detect_frames_detected/max(detect_frames_total,1)*100:.1f}%)",
        flush=True,
    )
    return cache

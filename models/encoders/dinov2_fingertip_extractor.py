"""
models/encoders/dinov2_fingertip_extractor.py — Stage 3 (strengthened design).

Per-frame feature recipe (post-Stage-1-v3 prompt §2 Task B):
  * Input crop resized to 336x336 (DINOv2-S/14 -> 24x24 patch grid).
  * Locate the MediaPipe INDEX_FINGER_TIP patch on the 24x24 grid.
  * Take the 3x3 patch window centered on it, weighted by a cosine-bell
    kernel that peaks at the centre (sums to 1).  Edge-clamped and
    weight-renormalised when the tip lies within 1 patch of a border.
  * Concatenate the bell-pool from frames (t-1, t, t+1) along the feature
    axis -> 3*D channels.  First/last frames repeat their own feature.
  * Append a per-frame visibility bit (1.0 if MediaPipe detected the hand
    on that frame, else 0.0).  When visibility=0, the 3*D temporal-context
    block is zeroed out so the downstream Conformer can learn to skip
    dropouts rather than treat zero-vector frames as real content.

Final per-frame feature shape: [3*D + 1] = [1153] for DINOv2-S (D=384).

Cache fingerprint
-----------------
The cache key includes encoder name, resolution, grid size, pool type,
temporal-context flag, and visibility-gate flag.  Any mismatch refuses to
load — see datasets/dinov2_feature_cache.py.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# MediaPipe joint indices we care about.
INDEX_FINGER_TIP    = 8
THUMB_TIP           = 4
MIDDLE_FINGER_TIP   = 12
RING_FINGER_TIP     = 16
PINKY_TIP           = 20
ALL_FINGERTIPS      = (THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP,
                       RING_FINGER_TIP, PINKY_TIP)


# ---------------------------------------------------------------------------
# Bell weights for the 3x3 fingertip window
# ---------------------------------------------------------------------------

def bell_weights_3x3(sigma: float = 1.0,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Cosine/Gaussian bell weights on a 3x3 grid, peaked at the centre.
    Sums to 1.  Used by `fingertip_pool_3x3`.

    Returns a Tensor of shape [3, 3].
    """
    coords = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    w = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


# ---------------------------------------------------------------------------
# 3x3 patch pool with edge clamping and weight renormalisation
# ---------------------------------------------------------------------------

def fingertip_pool_3x3(
    patch_tokens: torch.Tensor,           # [G, G, D]
    tip_xy_norm:  tuple[float, float] | torch.Tensor,
    *,
    weights:      Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Bell-weighted 3x3 pool around the patch that contains the fingertip.

    Parameters
    ----------
    patch_tokens : [G, G, D] DINOv2 patch features (CLS already excluded
                   and reshaped into a 2-D grid).
    tip_xy_norm  : (x, y) in [0,1] image-normalised coords.  x maps to
                   columns, y to rows.
    weights      : optional 3x3 weight tensor; defaults to bell_weights_3x3().

    Returns
    -------
    pooled : [D] feature for this frame.
    """
    if patch_tokens.dim() != 3:
        raise ValueError(f"patch_tokens must be [G,G,D], got {patch_tokens.shape}")
    G, G2, D = patch_tokens.shape
    if G != G2:
        raise ValueError(f"patch_tokens must be square in space; got {patch_tokens.shape}")

    device = patch_tokens.device
    dtype  = patch_tokens.dtype
    if weights is None:
        weights = bell_weights_3x3(dtype=dtype).to(device)
    else:
        weights = weights.to(device=device, dtype=dtype)

    # Convert tip to patch indices.
    if isinstance(tip_xy_norm, torch.Tensor):
        tx, ty = float(tip_xy_norm[0].item()), float(tip_xy_norm[1].item())
    else:
        tx, ty = float(tip_xy_norm[0]), float(tip_xy_norm[1])
    tx = min(max(tx, 0.0), 1.0)
    ty = min(max(ty, 0.0), 1.0)
    # Floor so a tip at 0.5*G lands in patch 0.5*G-1's row when at exactly the
    # patch boundary; clamp to [0, G-1].
    c = int(min(max(int(math.floor(tx * G)), 0), G - 1))
    r = int(min(max(int(math.floor(ty * G)), 0), G - 1))

    # 3x3 window with edge clamping.
    r0, r1 = max(r - 1, 0), min(r + 2, G)
    c0, c1 = max(c - 1, 0), min(c + 2, G)
    # Slice into the 3x3 weight grid that survives clamping.
    wr0 = r0 - (r - 1)
    wr1 = 3 - ((r + 2) - r1)
    wc0 = c0 - (c - 1)
    wc1 = 3 - ((c + 2) - c1)

    w = weights[wr0:wr1, wc0:wc1]
    w = w / w.sum().clamp(min=1e-8)
    region = patch_tokens[r0:r1, c0:c1, :]          # [<=3, <=3, D]
    return (region * w.unsqueeze(-1)).sum(dim=(0, 1))


# ---------------------------------------------------------------------------
# Per-clip pipeline: patch grid -> [T, 3D+1] with vis gate
# ---------------------------------------------------------------------------

def build_clip_features_stage3(
    patches:        torch.Tensor,         # [T, G*G, D] DINOv2 patch tokens
    tip_xy_per_t:   list[tuple[float, float] | None],   # length T
    grid_size:      int,
    *,
    visibility_gate: bool = True,
    temporal_context: bool = True,
    bell_sigma:      float = 1.0,
    multi_joint_tip_xy: Optional[list[list[Optional[tuple[float, float]]]]] = None,
) -> torch.Tensor:
    """
    Run the Stage-3 per-clip pipeline.

    Parameters
    ----------
    patches : [T, G*G, D] patch tokens (CLS excluded).
    tip_xy_per_t : per-frame index-fingertip (x, y) normalised to the
                   CROPPED frame coordinate space, or None if MediaPipe
                   missed that frame.  None is also where we apply the
                   visibility gate.
    grid_size : G (24 for 336/14 dinov2-s).
    visibility_gate : if True, zero out the temporal-context block and
                      append a 0/1 visibility column.  Net out_dim = 3*D+1.
                      If False, do not append the column; out_dim = 3*D.
    temporal_context : if True, concat [t-1, t, t+1]; else just t.
                       (Headline Stage 3 uses True.)
    multi_joint_tip_xy : optional per-frame list of [tip0, tip1, ... tip4]
                         (all five fingertips for the multi-joint ablation).
                         When provided, replaces the single fingertip pool
                         with a concatenation of 5 fingertip pools (5*D
                         channels per frame, before temporal context).

    Returns
    -------
    feats : [T, out_dim] float32 tensor.
            out_dim = (5 if multi_joint else 1) * D * (3 if context else 1)
                    + (1 if visibility_gate else 0)
    """
    if patches.dim() != 3:
        raise ValueError(f"patches must be [T, P, D], got {patches.shape}")
    T, P, D = patches.shape
    if P != grid_size * grid_size:
        raise ValueError(
            f"patch count {P} != grid_size^2 = {grid_size*grid_size}"
        )

    weights = bell_weights_3x3(sigma=bell_sigma, dtype=patches.dtype).to(patches.device)
    patches_2d = patches.view(T, grid_size, grid_size, D)

    # 1) Per-frame fingertip pool (or multi-joint pool).  Frames with no
    #    detection get a zero vector.
    if multi_joint_tip_xy is None:
        per_t = []
        vis_flags = []
        last_valid_xy: Optional[tuple[float, float]] = None
        for t in range(T):
            xy = tip_xy_per_t[t]
            if xy is not None:
                pooled = fingertip_pool_3x3(patches_2d[t], xy, weights=weights)
                vis_flags.append(1.0)
                last_valid_xy = xy
            else:
                vis_flags.append(0.0)
                if visibility_gate or last_valid_xy is None:
                    pooled = torch.zeros(D, dtype=patches.dtype, device=patches.device)
                else:
                    pooled = fingertip_pool_3x3(patches_2d[t], last_valid_xy, weights=weights)
            per_t.append(pooled)
        per_frame = torch.stack(per_t, dim=0)               # [T, D]
        vis_t     = torch.tensor(vis_flags, dtype=patches.dtype, device=patches.device)
    else:
        # Multi-joint: 5 fingertips.  Joints with None get zeroed.
        if len(multi_joint_tip_xy) != T:
            raise ValueError("multi_joint_tip_xy must have length T")
        per_t = []
        vis_flags = []
        for t in range(T):
            joint_list = multi_joint_tip_xy[t] or [None] * 5
            any_valid = False
            blocks = []
            for j in joint_list:
                if j is not None:
                    blocks.append(fingertip_pool_3x3(patches_2d[t], j, weights=weights))
                    any_valid = True
                else:
                    blocks.append(torch.zeros(D, dtype=patches.dtype, device=patches.device))
            per_t.append(torch.cat(blocks, dim=0))          # [5*D]
            vis_flags.append(1.0 if any_valid else 0.0)
        per_frame = torch.stack(per_t, dim=0)               # [T, 5*D]
        vis_t     = torch.tensor(vis_flags, dtype=patches.dtype, device=patches.device)

    # 2) Optional temporal context (concat t-1, t, t+1; repeat-pad ends).
    if temporal_context:
        prev = torch.cat([per_frame[:1], per_frame[:-1]], dim=0)
        nxt  = torch.cat([per_frame[1:], per_frame[-1:]], dim=0)
        ctx  = torch.cat([prev, per_frame, nxt], dim=-1)    # [T, 3*Dpf]
    else:
        ctx  = per_frame

    # 3) Visibility gate.
    if visibility_gate:
        ctx = ctx * vis_t.unsqueeze(-1)
        feats = torch.cat([ctx, vis_t.unsqueeze(-1)], dim=-1)
    else:
        feats = ctx
    return feats


# ---------------------------------------------------------------------------
# Output-dim helper (cache fingerprint construction)
# ---------------------------------------------------------------------------

def expected_out_dim(
    d_model_per_frame: int,         # 384 for dinov2-s
    *,
    temporal_context:  bool = True,
    visibility_gate:   bool = True,
    multi_joint:       bool = False,
) -> int:
    base = d_model_per_frame * (5 if multi_joint else 1)
    base = base * (3 if temporal_context else 1)
    return base + (1 if visibility_gate else 0)

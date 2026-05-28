"""
datasets/skeleton_augment.py — Augmentation pipeline for landmark sequences.

Operates on [T, 190]-shaped tensors produced by datasets/skeleton_cache.py.

The plan's prescribed set
-------------------------
  1) temporal warp (±15%)
  2) spatial jitter on (x, y) of ±2% of crop size
  3) random temporal crop 60-100% of the clip in time
  4) mirror with low probability  — DELIBERATELY OMITTED

Why no mirroring
----------------
Both the original WiTA paper (Kim et al. 2023) and the TR-AWR paper
(Tan et al. 2023) explicitly warn against horizontal-flip augmentation
because it inverts the writing-direction-of-motion and corrupts the
ground-truth label.  E.g. 'cat' written left-to-right becomes 'tac' under
x-mirror.  Same caveat applies to landmark coordinates: flipping x
swaps which direction the fingertip travels, breaking the label.

Composition policy
------------------
All three augmentations are applied INDEPENDENTLY per clip per __getitem__,
each gated by its own probability.  Defaults match the plan's intent:
~80% temporal warp, ~80% spatial jitter, ~50% temporal crop.

Channel layout assumption
-------------------------
Per build_clip_features in skeleton_cache.py:
  channels 0..62   : position    (21 joints × [x, y, z])
  channels 63..125 : velocity    (21 joints × [dx, dy, dz])
  channels 126..188: acceleration (21 joints × [ddx, ddy, ddz])
  channel  189     : per-frame visibility (1 if hand detected, 0 if fallback)
Total = 190.

Spatial jitter perturbs all (x, y, z) channels uniformly; velocity and
acceleration are NOT re-derived from the jittered positions because the
noise is small (sigma << expected fingertip displacement) and recomputing
would invalidate the caching contract.
"""

from __future__ import annotations
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def temporal_warp(feats: torch.Tensor, max_warp: float = 0.15) -> torch.Tensor:
    """
    Resample [T, D] to [T_warped, D] with T_warped in [(1-w)*T, (1+w)*T],
    then resample back to T.  Effectively changes the implied writing speed.
    """
    if max_warp <= 0:
        return feats
    T, D = feats.shape
    warp = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * max_warp   # in [1-w, 1+w]
    T_w = max(2, int(round(T * warp)))

    # Resample once to T_w, then back to T.  Combined effect is a non-linear
    # time stretch — still preserves trajectory but varies cadence.
    src_idx = torch.linspace(0, T - 1, T_w)
    src_lo  = src_idx.floor().long()
    src_hi  = (src_lo + 1).clamp(max=T - 1)
    frac    = (src_idx - src_lo.float()).unsqueeze(-1)
    interm  = feats[src_lo] * (1 - frac) + feats[src_hi] * frac      # [T_w, D]

    src_idx2 = torch.linspace(0, T_w - 1, T)
    src_lo2  = src_idx2.floor().long()
    src_hi2  = (src_lo2 + 1).clamp(max=T_w - 1)
    frac2    = (src_idx2 - src_lo2.float()).unsqueeze(-1)
    return interm[src_lo2] * (1 - frac2) + interm[src_hi2] * frac2   # [T, D]


def spatial_jitter(
    feats:    torch.Tensor,
    sigma:    float = 0.02,
    apply_to_vel_acc: bool = False,
) -> torch.Tensor:
    """
    Add Gaussian noise to position channels (and optionally velocity / accel).
    sigma is in normalized image coords ([0, 1]), so 0.02 ≈ 2% of crop side.

    Visibility channel (last column) is left untouched.
    """
    if sigma <= 0:
        return feats
    T, D = feats.shape
    out = feats.clone()
    pos = out[:, 0:63]
    out[:, 0:63] = pos + torch.randn_like(pos) * sigma
    if apply_to_vel_acc:
        vel = out[:, 63:126]
        acc = out[:, 126:189]
        out[:, 63:126]  = vel + torch.randn_like(vel) * sigma * 0.5
        out[:, 126:189] = acc + torch.randn_like(acc) * sigma * 0.25
    return out


def temporal_crop_resize(
    feats:     torch.Tensor,
    min_ratio: float = 0.60,
    max_ratio: float = 1.00,
) -> torch.Tensor:
    """
    Pick a random contiguous sub-segment whose length is in
    [min_ratio*T, max_ratio*T], then linearly interpolate back to T frames.

    Acts like a random temporal zoom — simulates varied writing speed AND
    truncation of the start/end of the gesture.
    """
    T, D = feats.shape
    if min_ratio >= 1.0:
        return feats
    ratio = min_ratio + torch.rand(1).item() * (max_ratio - min_ratio)
    seg_len = max(4, int(round(T * ratio)))
    if seg_len >= T:
        return feats
    start = torch.randint(0, T - seg_len + 1, (1,)).item()
    seg   = feats[start : start + seg_len]                    # [seg_len, D]

    src_idx = torch.linspace(0, seg_len - 1, T)
    src_lo  = src_idx.floor().long()
    src_hi  = (src_lo + 1).clamp(max=seg_len - 1)
    frac    = (src_idx - src_lo.float()).unsqueeze(-1)
    return seg[src_lo] * (1 - frac) + seg[src_hi] * frac     # [T, D]


# ---------------------------------------------------------------------------
# Composed augmenter
# ---------------------------------------------------------------------------

class LandmarkAugment:
    """
    Composable per-clip augmenter for the SkeletonDataset.

    Parameters
    ----------
    p_temporal_warp   : probability of applying temporal_warp
    temporal_warp_max : max stretch fraction (± of T)
    p_spatial_jitter  : probability of applying spatial_jitter
    spatial_sigma     : sigma of position noise in normalized coords
    p_temporal_crop   : probability of applying temporal_crop_resize
    temporal_crop_min : minimum ratio of T kept by the crop
    temporal_crop_max : maximum ratio of T kept by the crop
    seed              : optional fixed RNG seed (otherwise uses global RNG)

    Use
    ---
    >>> aug = LandmarkAugment()
    >>> train_ds = SkeletonDataset(cache, train_idx, converter, transform=aug)
    >>> val_ds   = SkeletonDataset(cache, val_idx, converter, transform=None)
    """

    def __init__(
        self,
        p_temporal_warp:   float = 0.80,
        temporal_warp_max: float = 0.15,
        p_spatial_jitter:  float = 0.80,
        spatial_sigma:     float = 0.02,
        p_temporal_crop:   float = 0.50,
        temporal_crop_min: float = 0.60,
        temporal_crop_max: float = 1.00,
        seed:              Optional[int] = None,
    ):
        self.p_warp   = p_temporal_warp
        self.warp_max = temporal_warp_max
        self.p_jitter = p_spatial_jitter
        self.sigma    = spatial_sigma
        self.p_crop   = p_temporal_crop
        self.crop_min = temporal_crop_min
        self.crop_max = temporal_crop_max
        # Per-worker RNG; if seed given, use it.  Otherwise leave to global.
        self._gen = torch.Generator() if seed is None else \
                    torch.Generator().manual_seed(seed)

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_warp:
            feats = temporal_warp(feats, max_warp=self.warp_max)
        if torch.rand(1).item() < self.p_jitter:
            feats = spatial_jitter(feats, sigma=self.sigma)
        if torch.rand(1).item() < self.p_crop:
            feats = temporal_crop_resize(
                feats, min_ratio=self.crop_min, max_ratio=self.crop_max,
            )
        return feats

    def __repr__(self) -> str:
        return (
            f"LandmarkAugment(p_warp={self.p_warp}, warp_max={self.warp_max}, "
            f"p_jitter={self.p_jitter}, sigma={self.sigma}, "
            f"p_crop={self.p_crop}, crop_range=[{self.crop_min},{self.crop_max}])"
        )

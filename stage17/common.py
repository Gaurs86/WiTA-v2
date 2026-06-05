"""
stage17/common.py — shared helpers for the trajectory-image + VLM pipeline.

REUSE (no new MediaPipe run): the Stage 11 landmark cache
(datasets/landmark_cache_122.py) writes one .npz per clip with key
`feature` of shape [T_native, 190].  The first 63 columns are the RAW 21
MediaPipe hand landmarks (x, y, z), flattened joint-major:

    feature[:, j*3 : j*3+3] = landmark j  (x, y, z), x/y in [0,1] image coords

The index fingertip is MediaPipe landmark 8 -> columns (24, 25, 26).  So:
    fingertip_xy = feature[:, 24:26]            # the (x,y) path to render
    all 21 joints = feature[:, :63].reshape(T,21,3)[..., :2]
and the per-clip hand-detection rate is the npz key `detected` (data quality).

Cache layout (same as landmark_cache_122):
    <cache_root>/<split>/<subset>/<SIGNER>__<clip_id>.npz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Reuse the standard-CTC char tools + CER from stage16 (single source of truth).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import CharConverter, cer_pair, gt_string   # noqa: E402,F401

INDEX_FINGER_TIP = 8        # MediaPipe Hands landmark id for the index fingertip
N_JOINTS = 21


def fingertip_cols(joint: int = INDEX_FINGER_TIP):
    """(x_col, y_col) into the 190-dim feature for a given landmark id."""
    return joint * 3, joint * 3 + 1


def load_clip_npz(npz_path: str) -> dict:
    """Load a Stage 11 per-clip npz -> dict with feature/label/signer/etc."""
    d = np.load(npz_path, allow_pickle=True)
    feat = d["feature"].astype(np.float32)            # [T, 190]

    def _s(key, default=""):
        if key not in d:
            return default
        v = d[key]
        try:
            return v.item() if getattr(v, "ndim", 1) == 0 else str(v)
        except Exception:
            return str(v)

    return {
        "feature": feat,
        "label":   str(_s("label")).strip().lower(),
        "signer":  str(_s("signer")),
        "subset":  str(_s("subset")),
        "clip_id": str(_s("clip_id")),
        "detected": float(d["detected"]) if "detected" in d else float("nan"),
    }


def fingertip_xy(feature: np.ndarray, joint: int = INDEX_FINGER_TIP) -> np.ndarray:
    """[T,2] fingertip (x,y) path in [0,1] image coords."""
    cx, cy = fingertip_cols(joint)
    return feature[:, [cx, cy]].astype(np.float32)


def all_landmarks_xy(feature: np.ndarray) -> np.ndarray:
    """[T,21,2] all-joint (x,y) — for richer renders if the fingertip path
    alone proves too sparse."""
    return feature[:, : N_JOINTS * 3].reshape(-1, N_JOINTS, 3)[..., :2].astype(np.float32)


def iter_clips(cache_root: str, split: str, subsets=("lex", "nonlex")) -> list[dict]:
    """List per-clip npz files for a split (lazy: label is loaded on demand)."""
    root = Path(cache_root)
    out = []
    for sub in subsets:
        d = root / split / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.npz")):
            stem = p.stem                          # <SIGNER>__<clip_id>
            signer = stem.split("__", 1)[0]
            out.append({"npz": str(p), "signer": signer, "subset": sub, "stem": stem})
    return out


def find_landmark_cache(default="/kaggle/working/landmark_cache_122") -> str:
    """Locate the landmark cache root (working dir, then any /kaggle/input mount)."""
    if os.path.isdir(os.path.join(default, "train")):
        return default
    base = "/kaggle/input"
    if os.path.isdir(base):
        for c in os.listdir(base):
            for cand in (os.path.join(base, c),
                         os.path.join(base, c, "landmark_cache_122")):
                if os.path.isdir(os.path.join(cand, "train")):
                    return cand
    return default

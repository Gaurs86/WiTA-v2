"""
stage18/normalize_cache.py — per-clip feature normalization (spec item 2).

The Stage 11 cache stores RAW MediaPipe [0,1] coords (absolute frame position),
which varies across signers and hurts cross-signer generalization (the 0.085->
0.652 per-signer spread).  This re-processes the cache so positions are
translation/scale-INVARIANT:
  * centre the (x,y) of all 21 joints by the per-clip mean,
  * scale by the per-clip std,
  * scale velocity / acceleration (x,y) by the same factor (consistent, since
    they are position differences); centring doesn't affect differences.
z and the visibility channel are left as-is.  input_dim stays 190 (no model
change), so it's a clean single-variable comparison vs Stage 11.

Applied to ALL splits (train/val/test) identically -> no train/eval mismatch and
no leakage (normalization is per-clip, derived only from that clip).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_JOINTS = 21


def normalize_feature(feat: np.ndarray) -> np.ndarray:
    """[T,190] raw -> [T,190] with per-clip-normalized (x,y) on pos/vel/acc."""
    out = feat.astype(np.float32).copy()
    T = out.shape[0]
    pos = out[:, :63].reshape(T, N_JOINTS, 3)
    xy = pos[:, :, :2]
    mean = xy.reshape(-1, 2).mean(0)                       # [2]
    scale = float(xy.reshape(-1, 2).std(0).mean())
    scale = scale if scale > 1e-6 else 1.0
    pos[:, :, :2] = (xy - mean) / scale
    out[:, :63] = pos.reshape(T, 63)
    for base in (63, 126):                                 # velocity, acceleration
        if base + 63 > out.shape[1] - 1:
            break
        blk = out[:, base:base + 63].reshape(T, N_JOINTS, 3)
        blk[:, :, :2] = blk[:, :, :2] / scale              # differences: scale only
        out[:, base:base + 63] = blk.reshape(T, 63)
    return out


def reprocess_cache(src_root: str, dst_root: str) -> int:
    """Write a normalized copy of the cache, preserving the split/subset layout
    and all npz fields (label/signer/subset/clip_id/detected)."""
    from stage17.common import iter_clips
    n = 0
    for split in ("train", "val", "test"):
        clips = iter_clips(src_root, split, ("lex", "nonlex"))
        print(f"  [{split}] {len(clips)} clips...", flush=True)
        for c in clips:
            d = np.load(c["npz"], allow_pickle=True)
            feat = normalize_feature(d["feature"].astype(np.float32))
            outdir = os.path.join(dst_root, split, c["subset"])
            os.makedirs(outdir, exist_ok=True)
            np.savez(os.path.join(outdir, os.path.basename(c["npz"])),
                     feature=feat.astype(np.float16), label=d["label"],
                     signer=d["signer"], subset=d["subset"],
                     clip_id=d["clip_id"], detected=d["detected"])
            n += 1
            if n % 1000 == 0:
                print(f"    reprocessed {n} clips", flush=True)
        print(f"  [{split}] done (running total {n})", flush=True)
    print(f"[normalize_cache] wrote {n} normalized clips -> {dst_root}", flush=True)
    return n


def _selftest():
    rng = np.random.RandomState(0)
    T = 32
    feat = np.zeros((T, 190), np.float32)
    pos = np.cumsum(rng.randn(T, 21, 3) * 0.02, 0) + 0.5    # absolute-ish positions
    vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(pos); acc[1:] = vel[1:] - vel[:-1]
    feat[:, :63] = pos.reshape(T, 63); feat[:, 63:126] = vel.reshape(T, 63)
    feat[:, 126:189] = acc.reshape(T, 63); feat[:, 189] = 1.0
    out = normalize_feature(feat)
    xy = out[:, :63].reshape(T, 21, 3)[:, :, :2].reshape(-1, 2)
    assert out.shape == feat.shape
    assert abs(xy.mean()) < 1e-5, f"not centred: {xy.mean()}"
    assert abs(xy.std(0).mean() - 1.0) < 1e-3, f"not unit-scaled: {xy.std(0).mean()}"
    assert np.allclose(out[:, 189], feat[:, 189]), "visibility changed"
    # consistency: normalized vel of joint8 == diff of normalized pos of joint8
    po = out[:, :63].reshape(T, 21, 3); vo = out[:, 63:126].reshape(T, 21, 3)
    assert np.allclose(vo[1:, 8, :2], po[1:, 8, :2] - po[:-1, 8, :2], atol=1e-4), "vel inconsistent"
    print("[normalize_cache] self-test OK: centred, unit-scaled, vis preserved, vel consistent")


if __name__ == "__main__":
    _selftest()

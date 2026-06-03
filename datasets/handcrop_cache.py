"""
datasets/handcrop_cache.py — Stage 12 prereq: 224x224 hand-crop video cache.

Reuses the paper-split per-signer directory layout that landmark_cache_122
already handles (gt.txt per signer subdir, numbered clip subdirs of jpegs).
For each clip:
  1. Read all frame jpegs.
  2. Run MediaPipe HandLandmarker per frame -> 21 (x,y) per detected frame.
  3. Compute union bbox over all valid detections, 1.3x pad, square it,
     clamp to image bounds.
  4. Sample T=16 frames uniformly from the original frame range.
  5. Crop each sampled frame with the union bbox; resize to 224x224 uint8.
  6. Save as <SIGNER>__<clip_id>.npz with key 'video' [T=16, 224, 224, 3].

Disk cost: 16 * 224 * 224 * 3 = 2.4 MB per clip uint8.  ~10300 clips
across all splits = ~25 GB total.  Use np.savez (not compressed) so
loading is fast at training time.

Multiprocessing fast path: extract_dir_handcrops_parallel() mirrors the
landmark_cache_122 pattern -- 4 workers, each lazy-initialises its own
MediaPipe LandmarkExtractor.  Per-clip work is ~0.5-1 s on T4 CPU.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .skeleton_cache  import LandmarkExtractor
from .landmark_cache_122 import (
    _signer_id_from_parent, _safe_clip_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_union_bbox_pixels(
    lms_per_frame: list[Optional[np.ndarray]],
    frame_h: int,
    frame_w: int,
    pad_factor: float = 1.3,
) -> tuple[int, int, int, int]:
    """
    Union bounding box (x0, y0, x1, y1) in pixel coords across all valid
    detections.  Padded by `pad_factor` and made square (longest side).
    Clamped to image bounds.  Falls back to a centred square when no
    valid landmarks were detected.
    """
    valid = [lm for lm in lms_per_frame if lm is not None]
    if not valid:
        side = min(frame_h, frame_w)
        x0 = (frame_w - side) // 2; y0 = (frame_h - side) // 2
        return (x0, y0, x0 + side, y0 + side)
    xs = np.concatenate([lm[:, 0] for lm in valid]) * frame_w
    ys = np.concatenate([lm[:, 1] for lm in valid]) * frame_h
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0) * pad_factor
    half = side * 0.5
    x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(frame_w, int(x1)); y1 = min(frame_h, int(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        # Degenerate bbox -> fall back to centred square.
        side = min(frame_h, frame_w)
        x0 = (frame_w - side) // 2; y0 = (frame_h - side) // 2
        return (x0, y0, x0 + side, y0 + side)
    return (x0, y0, x1, y1)


def _build_handcrop_video(
    frame_paths: list[Path],
    extractor:   LandmarkExtractor,
    *,
    T:           int = 16,
    crop_size:   int = 224,
    pad_factor:  float = 1.3,
) -> tuple[np.ndarray, dict]:
    """
    Build the [T, crop_size, crop_size, 3] uint8 hand-crop video.

    SPEED CONTRACT
    --------------
    MediaPipe runs ONLY on the T uniformly-sampled frames we actually
    crop, not on every input frame.  WiTA clips average ~64 input
    frames; detecting on all of them was the dominant cost and gave
    no additional information for the union bbox.
    """
    from PIL import Image

    n_in = len(frame_paths)
    if n_in == 0:
        raise ValueError("Empty frame list.")
    # Plan indices first; decode + MediaPipe only on those.
    idx = np.linspace(0, n_in - 1, T).round().astype(int)
    idx = np.clip(idx, 0, n_in - 1)
    sampled_paths = [frame_paths[int(i)] for i in idx]

    pil_frames = [Image.open(p).convert("RGB") for p in sampled_paths]
    H, W = pil_frames[0].size[1], pil_frames[0].size[0]   # PIL is (W, H)

    lms   = [extractor.detect(f) for f in pil_frames]
    n_det = sum(1 for lm in lms if lm is not None)
    bbox  = _compute_union_bbox_pixels(lms, H, W, pad_factor=pad_factor)

    x0, y0, x1, y1 = bbox
    out = np.zeros((T, crop_size, crop_size, 3), dtype=np.uint8)
    for t, pil in enumerate(pil_frames):
        arr  = np.asarray(pil)                                # [H, W, 3] RGB uint8
        crop = arr[y0:y1, x0:x1]
        out[t] = cv2.resize(crop, (crop_size, crop_size),
                            interpolation=cv2.INTER_LINEAR)
    return out, {
        "n_frames_seen":     n_in,
        "n_frames_sampled":  T,
        "n_frames_detected": n_det,
        "detect_rate":       n_det / max(T, 1),
        "bbox":              [int(b) for b in bbox],
    }


# ---------------------------------------------------------------------------
# Multiprocessing extractor
# ---------------------------------------------------------------------------

_worker_extractor = None      # type: ignore[var-annotated]


def _init_worker():
    global _worker_extractor
    if _worker_extractor is None:
        _worker_extractor = LandmarkExtractor()


def _process_one_handcrop(work_item: tuple) -> dict:
    """
    Worker: read jpegs, run MediaPipe + crop pipeline, save npz.
    """
    (clip_dir, label, signer_id, clip_id, out_npz, subset,
     T, crop_size, pad_factor, overwrite) = work_item

    if os.path.exists(out_npz) and not overwrite:
        return {"existed": True}
    try:
        clip_dir = Path(clip_dir)
        frame_paths = sorted(
            [p for p in clip_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not frame_paths:
            return {"skipped": True}

        global _worker_extractor
        if _worker_extractor is None:
            _init_worker()

        video, stats = _build_handcrop_video(
            frame_paths, _worker_extractor,
            T=T, crop_size=crop_size, pad_factor=pad_factor,
        )
        # COMPRESSED — Kaggle /kaggle/working/ is 20 GB; uint8 hand-crop
        # video compresses 3-5x with deflate (background pixels dominate).
        # Uncompressed (~2.4 MB/clip × 10K) overflowed at train/lex 6517.
        # Load-time decompression cost is negligible vs the GPU forward.
        np.savez_compressed(
            out_npz,
            video=video,                              # [T, crop_size, crop_size, 3] uint8
            label=label, signer=signer_id, subset=subset,
            clip_id=clip_id,
            detected=np.float32(stats["detect_rate"]),
            bbox=np.array(stats["bbox"], dtype=np.int32),
        )
        return {
            "written":           True,
            "n_frames_seen":     stats["n_frames_seen"],
            "n_frames_detected": stats["n_frames_detected"],
        }
    except Exception as e:
        return {"skipped": True, "err": f"{type(e).__name__}: {e}"}


def _collect_handcrop_work_items(
    dir_path:   str,
    out_dir:    str,
    split:      str,
    subset:     str,
    *,
    T:          int,
    crop_size:  int,
    pad_factor: float,
    overwrite:  bool,
) -> tuple[list[tuple], set[str]]:
    out_subset = Path(out_dir) / split / subset
    out_subset.mkdir(parents=True, exist_ok=True)
    work: list[tuple] = []
    signers: set[str] = set()
    dir_p = Path(dir_path)
    for gt_full in dir_p.rglob("gt.txt"):
        parent = gt_full.parent
        parent_rel = str(parent.relative_to(dir_p))
        try:
            signer_id = _signer_id_from_parent(parent_rel)
        except ValueError as e:
            logger.warning("  skip %s: %s", parent_rel, e); continue
        signers.add(signer_id)
        try:
            with open(gt_full, "r", encoding="utf-8", errors="replace") as f:
                labels = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            logger.warning("  read %s failed: %s", gt_full, e); continue
        for clip_idx, label in enumerate(labels):
            clip_id  = _safe_clip_id(parent_rel, clip_idx)
            out_npz  = str(out_subset / f"{signer_id}__{clip_id}.npz")
            clip_dir = str(parent / str(clip_idx))
            work.append((
                clip_dir, label, signer_id, clip_id, out_npz, subset,
                T, crop_size, pad_factor, overwrite,
            ))
    return work, signers


def extract_dir_handcrops_parallel(
    dir_path:        str,
    out_dir:         str,
    split:           str,
    subset:          str,
    *,
    n_workers:       int = 4,
    T:               int = 16,
    crop_size:       int = 224,
    pad_factor:      float = 1.3,
    overwrite:       bool = False,
    log_every:       int = 5,
    heartbeat_sec:   float = 30.0,
) -> dict:
    """
    Parallel hand-crop extraction.

    Progress contract
    -----------------
    * `log_every=5` and `chunksize=1`: completions are reported one-by-one
      and a status line lands every 5 finished clips.  This gives a clear
      ETA inside the first ~30 s of work.
    * `heartbeat_sec=30`: if no completions arrive for 30 seconds we print
      a heartbeat line including the elapsed time.  Lets you distinguish
      "workers initialising" from "workers deadlocked".
    """
    import multiprocessing as mp
    import shutil
    assert split  in {"train", "val", "test"}
    assert subset in {"lex", "nonlex"}

    # Disk guard: bail fast if free space is too low to even start.
    # Compressed clips are ~600-800 KB; demand at least 1.5 GB headroom.
    free = shutil.disk_usage(out_dir).free
    if free < 1_500_000_000:
        raise RuntimeError(
            f"[handcrop_cache] only {free/1e9:.2f} GB free under {out_dir} -- "
            f"refusing to start.  Wipe the partial cache or move to a bigger "
            f"output dir before re-running."
        )

    print(f"[handcrop_cache] parallel ({n_workers} workers) "
          f"{dir_path} -> {split}/{subset}  T={T} crop={crop_size}  "
          f"free={free/1e9:.1f} GB", flush=True)
    work, signers = _collect_handcrop_work_items(
        dir_path=dir_path, out_dir=out_dir, split=split, subset=subset,
        T=T, crop_size=crop_size, pad_factor=pad_factor, overwrite=overwrite,
    )
    print(f"  {len(work)} clips queued ({len(signers)} signers)  "
          f"chunksize=1  log_every={log_every}  heartbeat={heartbeat_sec:.0f}s",
          flush=True)
    if not work:
        return {"split": split, "subset": subset, "n_total": 0,
                "n_written": 0, "n_existing": 0, "n_skipped": 0,
                "n_signers": len(signers), "signers": sorted(signers),
                "detect_rate": float('nan'),
                "frames_detected": 0, "frames_total": 0}

    ctx = mp.get_context("fork")
    n_total = n_written = n_existing = n_skipped = 0
    fd_total = fd_det = 0
    t0 = time.time()
    last_event = t0
    last_hb    = t0
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        # imap_unordered with timeout-polling so we can emit heartbeats when
        # the queue is silent (workers still spinning up, or stalled).
        it = pool.imap_unordered(_process_one_handcrop, work, chunksize=1)
        remaining = len(work)
        while remaining > 0:
            try:
                # `next(it, timeout=...)` isn't supported; use the iterator
                # protocol via a small private helper on IMapIterator.
                r = it.next(timeout=heartbeat_sec)
            except mp.TimeoutError:
                now = time.time()
                print(f"  [{split}/{subset}] heartbeat: no completion in "
                      f"{now - last_event:.0f}s  (elapsed total {now - t0:.0f}s, "
                      f"done={n_total}/{len(work)})", flush=True)
                last_hb = now
                continue
            now = time.time()
            last_event = now
            n_total  += 1
            remaining -= 1
            if r.get("existed"): n_existing += 1
            elif r.get("written"):
                n_written += 1
                fd_total += r.get("n_frames_seen",     0)
                fd_det   += r.get("n_frames_detected", 0)
            else:
                n_skipped += 1
                if r.get("err"):
                    # Surface the first skip from each worker — silent
                    # exceptions were the 2000-s stall in the first run.
                    print(f"    skip: {r['err']}", flush=True)
            if n_total % log_every == 0 or n_total == len(work):
                elapsed = now - t0
                rate    = n_total / max(elapsed, 1e-3)
                eta_min = (len(work) - n_total) / max(rate, 1e-3) / 60.0
                drate   = fd_det / max(fd_total, 1) * 100
                print(f"  [{split}/{subset}] {n_total}/{len(work)}  "
                      f"({rate:.1f} clips/s)  ETA {eta_min:5.1f} min  "
                      f"written={n_written}  reused={n_existing}  "
                      f"skipped={n_skipped}  detect={drate:.1f}%", flush=True)

    print(f"[handcrop_cache] done  {split}/{subset}: written={n_written}  "
          f"reused={n_existing}  skipped={n_skipped}  "
          f"signers={len(signers)}  elapsed={time.time()-t0:.0f}s", flush=True)
    return {
        "split": split, "subset": subset,
        "n_total": n_total, "n_written": n_written,
        "n_existing": n_existing, "n_skipped": n_skipped,
        "n_signers": len(signers), "signers": sorted(signers),
        "detect_rate": fd_det / max(fd_total, 1),
        "frames_detected": fd_det, "frames_total": fd_total,
    }

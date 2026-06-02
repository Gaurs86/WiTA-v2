"""
datasets/landmark_cache_122.py — per-clip landmark cache for the 122-signer
paper-split dataset.

Unlike the 38-signer HuggingFace pipeline (which builds ONE .pt with all
clips bundled), this writes ONE .npz per clip into the layout:

    landmark_cache_122/
        train/{lex, nonlex}/<SIGNER>__<clip_id>.npz
        val/{lex, nonlex}/<SIGNER>__<clip_id>.npz
        test/{lex, nonlex}/<SIGNER>__<clip_id>.npz

so the WiTAPaperSplitDataset can lazily load clips per-batch and downstream
analysis (per-signer CER, per-subset CER) is one filename parse away.

Zip layout assumption — same as the HF format the existing `_index_zip`
already handles: each zip contains one or more per-signer subdirectories,
each subdirectory has a `gt.txt` with one label per line, and frame jpegs
live under `<signer_subdir>/<clip_idx>/<frame>.jpg`.

Per-clip npz schema
-------------------
    feature   : [T_native, 190] float16 (matches the existing fp16 cache)
    label     : str
    signer    : str          # e.g. 'CYB'
    subset    : str          # 'lex' or 'nonlex'
    clip_id   : str          # filename-safe id within the signer subdir
    detected  : float        # frame_detect_rate for this clip
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from .skeleton_cache  import LandmarkExtractor, build_clip_features
from .subject_splits  import subject_id_from_zip
from .dataset         import _parse_gt, _read_frames_from_zip

logger = logging.getLogger(__name__)


_SIGNER_PREFIX_RE = re.compile(r"^([A-Za-z]{2,4})_")


def _signer_id_from_parent(parent: str) -> str:
    """
    Extract the signer ID prefix from a `<SIGNER>_<gender>_<age>_<lang>_<type>`
    subdirectory.  Reuses the same convention as the HF zip filenames.
    """
    base = os.path.basename(parent.rstrip("/"))
    m = _SIGNER_PREFIX_RE.match(base)
    if not m:
        raise ValueError(
            f"Could not extract signer ID from subdir {base!r}.  "
            "Expected layout: <SIGNER>_<gender>_<age>_<lang>_<type>/"
        )
    return m.group(1).upper()


def _safe_clip_id(parent: str, clip_idx: int) -> str:
    """Filename-safe clip ID: '<parent_basename>_<clip_idx>'."""
    base = os.path.basename(parent.rstrip("/"))
    return f"{base}_{clip_idx}"


def extract_zip_per_clip_landmarks(
    zip_path:        str,
    out_dir:         str,
    split:           str,                        # 'train' | 'val' | 'test'
    subset:          str,                        # 'lex'   | 'nonlex'
    *,
    lang:            str = "english",
    max_frames:      int = 64,
    T_native:        int = 32,
    extractor:       LandmarkExtractor | None = None,
    overwrite:       bool = False,
    log_every:       int = 100,
) -> dict:
    """
    Walk one paper-split zip and write per-clip .npz files under
    `out_dir/<split>/<subset>/`.

    Returns a stats dict: total clips processed, skipped, signer-count,
    global detect_rate.

    `overwrite=False` skips clips whose npz already exists — robust to
    Kaggle session disconnects mid-extraction.
    """
    assert split  in {"train", "val", "test"}, split
    assert subset in {"lex", "nonlex"}, subset

    out_subset = Path(out_dir) / split / subset
    out_subset.mkdir(parents=True, exist_ok=True)

    own_extractor = False
    if extractor is None:
        extractor = LandmarkExtractor()
        own_extractor = True

    n_total = n_written = n_skipped = n_existing = 0
    detect_frames_total = detect_frames_detected = 0
    seen_signers: set[str] = set()

    logger.info("[landmark_cache_122] processing %s -> %s/%s",
                zip_path, split, subset)

    with zipfile.ZipFile(zip_path, "r") as zf:
        gt_files = [n for n in zf.namelist() if n.endswith("gt.txt")]
        logger.info("  found %d gt.txt files (per-signer subdirs)", len(gt_files))

        for gi, gt_path in enumerate(gt_files):
            parent = gt_path.rsplit("/gt.txt", 1)[0]
            try:
                signer_id = _signer_id_from_parent(parent)
            except ValueError as e:
                logger.warning("  skip %s: %s", parent, e)
                continue
            seen_signers.add(signer_id)

            try:
                labels = _parse_gt(zf, gt_path, lang)
            except Exception as e:
                logger.warning("  could not read %s: %s", gt_path, e)
                continue

            for clip_idx, label in enumerate(labels):
                n_total += 1
                clip_id = _safe_clip_id(parent, clip_idx)
                out_npz = out_subset / f"{signer_id}__{clip_id}.npz"
                if out_npz.exists() and not overwrite:
                    n_existing += 1
                    continue

                seq_prefix = f"{parent}/{clip_idx}/"
                frames = _read_frames_from_zip(zf, seq_prefix, max_frames)
                if frames is None:
                    n_skipped += 1
                    continue

                try:
                    feats, stats = build_clip_features(
                        frames, extractor, T_native=T_native,
                    )
                except Exception as e:
                    logger.warning("  clip %s failed: %s", clip_id, e)
                    n_skipped += 1
                    continue

                detect_frames_total    += stats["n_frames_seen"]
                detect_frames_detected += stats["n_frames_detected"]

                np.savez_compressed(
                    out_npz,
                    feature=feats.astype(np.float16),
                    label=label,
                    signer=signer_id,
                    subset=subset,
                    clip_id=clip_id,
                    detected=np.float32(stats["detect_rate"]),
                )
                n_written += 1

                if (n_total) % log_every == 0:
                    print(
                        f"  [{split}/{subset}] {n_total} clips  "
                        f"(written {n_written}, existing {n_existing}, "
                        f"skipped {n_skipped})  "
                        f"detect={detect_frames_detected / max(detect_frames_total, 1) * 100:.1f}%",
                        flush=True,
                    )

    if own_extractor:
        extractor.close()

    print(
        f"[landmark_cache_122] done  {split}/{subset}:  total={n_total}  "
        f"written={n_written}  reused={n_existing}  skipped={n_skipped}  "
        f"signers={len(seen_signers)}  "
        f"detect_rate={detect_frames_detected / max(detect_frames_total, 1) * 100:.2f}%",
        flush=True,
    )
    return {
        "split":             split,
        "subset":            subset,
        "n_total":           n_total,
        "n_written":         n_written,
        "n_existing":        n_existing,
        "n_skipped":         n_skipped,
        "n_signers":         len(seen_signers),
        "signers":           sorted(seen_signers),
        "detect_rate":       detect_frames_detected / max(detect_frames_total, 1),
        "frames_detected":   detect_frames_detected,
        "frames_total":      detect_frames_total,
    }


def extract_dir_per_clip_landmarks(
    dir_path:        str,
    out_dir:         str,
    split:           str,
    subset:          str,
    *,
    lang:            str = "english",
    max_frames:      int = 64,
    T_native:        int = 32,
    extractor:       LandmarkExtractor | None = None,
    overwrite:       bool = False,
    log_every:       int = 100,
) -> dict:
    """
    Same as `extract_zip_per_clip_landmarks` but reads from a directory
    that's already been extracted (Kaggle auto-unpacks .zip datasets at
    mount time).  The directory is expected to contain per-signer
    subdirectories with `gt.txt` and per-clip numbered subfolders of
    frames -- the same layout the HF zips have.
    """
    from PIL import Image
    import io

    assert split  in {"train", "val", "test"}, split
    assert subset in {"lex", "nonlex"}, subset

    out_subset = Path(out_dir) / split / subset
    out_subset.mkdir(parents=True, exist_ok=True)

    own_extractor = False
    if extractor is None:
        extractor = LandmarkExtractor()
        own_extractor = True

    n_total = n_written = n_skipped = n_existing = 0
    detect_frames_total = detect_frames_detected = 0
    seen_signers: set[str] = set()

    dir_path = Path(dir_path)
    logger.info("[landmark_cache_122] processing dir %s -> %s/%s",
                dir_path, split, subset)

    # Walk for every gt.txt under dir_path.
    gt_files = sorted(dir_path.rglob("gt.txt"))
    logger.info("  found %d gt.txt files (per-signer subdirs)", len(gt_files))

    for gi, gt_full in enumerate(gt_files):
        parent_dir = gt_full.parent
        parent_rel = str(parent_dir.relative_to(dir_path))
        try:
            signer_id = _signer_id_from_parent(parent_rel)
        except ValueError as e:
            logger.warning("  skip %s: %s", parent_rel, e)
            continue
        seen_signers.add(signer_id)

        # Parse labels (text lines).  English filter mirrors _parse_gt.
        try:
            with open(gt_full, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            labels = lines           # one per clip subdir
        except Exception as e:
            logger.warning("  could not read %s: %s", gt_full, e)
            continue

        for clip_idx, label in enumerate(labels):
            n_total += 1
            clip_id = _safe_clip_id(parent_rel, clip_idx)
            out_npz = out_subset / f"{signer_id}__{clip_id}.npz"
            if out_npz.exists() and not overwrite:
                n_existing += 1
                continue

            clip_dir = parent_dir / str(clip_idx)
            if not clip_dir.is_dir():
                n_skipped += 1
                continue

            # Read frame bytes (jpegs sorted).
            frame_paths = sorted(
                [p for p in clip_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
            if max_frames and len(frame_paths) > max_frames:
                # Uniform-subsample like _read_frames_from_zip.
                idx = np.linspace(0, len(frame_paths) - 1, max_frames).round().astype(int)
                frame_paths = [frame_paths[int(i)] for i in idx]
            if not frame_paths:
                n_skipped += 1
                continue
            frames = [p.read_bytes() for p in frame_paths]

            try:
                feats, stats = build_clip_features(
                    frames, extractor, T_native=T_native,
                )
            except Exception as e:
                logger.warning("  clip %s failed: %s", clip_id, e)
                n_skipped += 1
                continue

            detect_frames_total    += stats["n_frames_seen"]
            detect_frames_detected += stats["n_frames_detected"]

            np.savez_compressed(
                out_npz,
                feature=feats.astype(np.float16),
                label=label,
                signer=signer_id,
                subset=subset,
                clip_id=clip_id,
                detected=np.float32(stats["detect_rate"]),
            )
            n_written += 1

            if (n_total) % log_every == 0:
                print(
                    f"  [{split}/{subset}] {n_total} clips  "
                    f"(written {n_written}, existing {n_existing}, "
                    f"skipped {n_skipped})  "
                    f"detect={detect_frames_detected / max(detect_frames_total, 1) * 100:.1f}%",
                    flush=True,
                )

    if own_extractor:
        extractor.close()

    print(
        f"[landmark_cache_122] done  {split}/{subset}:  total={n_total}  "
        f"written={n_written}  reused={n_existing}  skipped={n_skipped}  "
        f"signers={len(seen_signers)}  "
        f"detect_rate={detect_frames_detected / max(detect_frames_total, 1) * 100:.2f}%",
        flush=True,
    )
    return {
        "split":             split,
        "subset":            subset,
        "n_total":           n_total,
        "n_written":         n_written,
        "n_existing":        n_existing,
        "n_skipped":         n_skipped,
        "n_signers":         len(seen_signers),
        "signers":           sorted(seen_signers),
        "detect_rate":       detect_frames_detected / max(detect_frames_total, 1),
        "frames_detected":   detect_frames_detected,
        "frames_total":      detect_frames_total,
    }


# ---------------------------------------------------------------------------
# Multiprocessing fast path
# ---------------------------------------------------------------------------

# Each worker holds ONE LandmarkExtractor in a module-level global.
# fork() default on Linux means the children get the parent's loaded
# Python state cheaply; the extractor is built lazily via init_worker().

_worker_extractor = None      # type: ignore[var-annotated]


def _init_worker():
    """Pool initializer: build a fresh LandmarkExtractor per worker."""
    global _worker_extractor
    if _worker_extractor is None:
        _worker_extractor = LandmarkExtractor()


def _process_one_dir_clip(work_item: tuple) -> dict:
    """
    Worker side: read frames from disk, run MediaPipe + feature build,
    write the per-clip .npz.  Returns small stats dict.

    work_item:
      (clip_dir_path, label, signer_id, clip_id, out_npz_path,
       subset, max_frames, T_native, overwrite)
    """
    (clip_dir, label, signer_id, clip_id, out_npz,
     subset, max_frames, T_native, overwrite) = work_item

    if os.path.exists(out_npz) and not overwrite:
        return {"existed": True}

    try:
        # Sort frames by filename then take up to max_frames uniformly.
        from pathlib import Path
        clip_dir = Path(clip_dir)
        frame_paths = sorted(
            [p for p in clip_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not frame_paths:
            return {"skipped": True}
        if max_frames and len(frame_paths) > max_frames:
            idx = np.linspace(0, len(frame_paths) - 1, max_frames).round().astype(int)
            frame_paths = [frame_paths[int(i)] for i in idx]
        frames = [p.read_bytes() for p in frame_paths]

        global _worker_extractor
        if _worker_extractor is None:
            _init_worker()

        feats, stats = build_clip_features(
            frames, _worker_extractor, T_native=T_native,
        )

        # Write uncompressed (np.savez): ~5x faster than savez_compressed
        # for the per-clip overhead, ~150 MB total disk vs ~50 MB compressed.
        # Still well under Kaggle's 20 GB /kaggle/working limit.
        np.savez(
            out_npz,
            feature=feats.astype(np.float16),
            label=label,
            signer=signer_id,
            subset=subset,
            clip_id=clip_id,
            detected=np.float32(stats["detect_rate"]),
        )
        return {
            "written":          True,
            "n_frames_seen":    stats["n_frames_seen"],
            "n_frames_detected": stats["n_frames_detected"],
        }
    except Exception as e:
        return {"skipped": True, "err": f"{type(e).__name__}: {e}"}


def _collect_dir_work_items(
    dir_path:  str,
    out_dir:   str,
    split:     str,
    subset:    str,
    max_frames: int,
    T_native:   int,
    overwrite:  bool,
) -> tuple[list[tuple], set[str]]:
    """Walk the directory once, return the list of per-clip work items."""
    out_subset = Path(out_dir) / split / subset
    out_subset.mkdir(parents=True, exist_ok=True)

    work_items: list[tuple] = []
    seen_signers: set[str] = set()
    dir_path_p = Path(dir_path)

    for gt_full in dir_path_p.rglob("gt.txt"):
        parent_dir = gt_full.parent
        parent_rel = str(parent_dir.relative_to(dir_path_p))
        try:
            signer_id = _signer_id_from_parent(parent_rel)
        except ValueError as e:
            logger.warning("  skip %s: %s", parent_rel, e)
            continue
        seen_signers.add(signer_id)
        try:
            with open(gt_full, "r", encoding="utf-8", errors="replace") as f:
                labels = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            logger.warning("  could not read %s: %s", gt_full, e)
            continue

        for clip_idx, label in enumerate(labels):
            clip_id   = _safe_clip_id(parent_rel, clip_idx)
            out_npz   = str(out_subset / f"{signer_id}__{clip_id}.npz")
            clip_dir  = str(parent_dir / str(clip_idx))
            work_items.append((
                clip_dir, label, signer_id, clip_id, out_npz,
                subset, max_frames, T_native, overwrite,
            ))
    return work_items, seen_signers


def extract_dir_per_clip_landmarks_parallel(
    dir_path:        str,
    out_dir:         str,
    split:           str,
    subset:          str,
    *,
    n_workers:       int = 4,
    max_frames:      int = 64,
    T_native:        int = 32,
    overwrite:       bool = False,
    log_every:       int = 100,
) -> dict:
    """
    Multi-process directory extractor.  ~4x faster than the serial
    version on Kaggle's 4-core CPU.

    Returns a stats dict in the same shape as the serial function.
    """
    import multiprocessing as mp

    assert split  in {"train", "val", "test"}, split
    assert subset in {"lex", "nonlex"}, subset

    print(f"[landmark_cache_122] parallel ({n_workers} workers) "
          f"processing {dir_path} -> {split}/{subset}", flush=True)

    work_items, seen_signers = _collect_dir_work_items(
        dir_path=dir_path, out_dir=out_dir, split=split, subset=subset,
        max_frames=max_frames, T_native=T_native, overwrite=overwrite,
    )
    print(f"  {len(work_items)} clips queued ({len(seen_signers)} signers)",
          flush=True)
    if not work_items:
        return {
            "split": split, "subset": subset,
            "n_total": 0, "n_written": 0, "n_existing": 0, "n_skipped": 0,
            "n_signers": len(seen_signers), "signers": sorted(seen_signers),
            "detect_rate": float("nan"),
            "frames_detected": 0, "frames_total": 0,
        }

    # Use spawn-safe pool with the worker initializer.
    ctx = mp.get_context("fork")        # Linux default; fastest on Kaggle
    n_total = n_written = n_existing = n_skipped = 0
    detect_frames_total = detect_frames_detected = 0
    t0 = time.time()
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_one_dir_clip,
                                                  work_items, chunksize=8)):
            n_total += 1
            if r.get("existed"):
                n_existing += 1
            elif r.get("written"):
                n_written += 1
                detect_frames_total    += r["n_frames_seen"]
                detect_frames_detected += r["n_frames_detected"]
            else:
                n_skipped += 1
            if n_total % log_every == 0 or n_total == len(work_items):
                elapsed = time.time() - t0
                rate    = n_total / max(elapsed, 1e-3)
                eta_min = (len(work_items) - n_total) / max(rate, 1e-3) / 60.0
                drate   = (detect_frames_detected
                           / max(detect_frames_total, 1) * 100.0)
                print(
                    f"  [{split}/{subset}] {n_total}/{len(work_items)}  "
                    f"({rate:.1f} clips/s)  ETA {eta_min:5.1f} min  "
                    f"written={n_written}  reused={n_existing}  "
                    f"skipped={n_skipped}  detect={drate:.1f}%",
                    flush=True,
                )

    print(
        f"[landmark_cache_122] done  {split}/{subset}:  total={n_total}  "
        f"written={n_written}  reused={n_existing}  skipped={n_skipped}  "
        f"signers={len(seen_signers)}  "
        f"detect_rate={detect_frames_detected / max(detect_frames_total, 1) * 100:.2f}%  "
        f"elapsed={time.time()-t0:.0f}s",
        flush=True,
    )
    return {
        "split":             split,
        "subset":            subset,
        "n_total":           n_total,
        "n_written":         n_written,
        "n_existing":        n_existing,
        "n_skipped":         n_skipped,
        "n_signers":         len(seen_signers),
        "signers":           sorted(seen_signers),
        "detect_rate":       detect_frames_detected / max(detect_frames_total, 1),
        "frames_detected":   detect_frames_detected,
        "frames_total":      detect_frames_total,
    }


def extract_temp_then_process(
    zip_path:        str,
    out_dir:         str,
    split:           str,
    subset:          str,
    *,
    temp_dir:        str = "/kaggle/temp/wita_extract",
    **kwargs,
) -> dict:
    """
    For very large zips (the paper-split releases can be ~10 GB each),
    extract to `temp_dir` first, then re-pack a temporary zip with the
    extracted contents... actually the cleanest approach is to NOT extract
    and just stream from the zip directly — `extract_zip_per_clip_landmarks`
    already does that.  This wrapper exists only to clean `temp_dir` if a
    caller used the documented two-step pattern.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    try:
        return extract_zip_per_clip_landmarks(
            zip_path=zip_path, out_dir=out_dir,
            split=split, subset=subset, **kwargs,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

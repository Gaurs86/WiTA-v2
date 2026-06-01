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

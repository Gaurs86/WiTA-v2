"""
datasets/subject_splits.py — Subject-disjoint train/val splitter for WiTA.

Why this matters
----------------
The original `make_dataloaders` splits clips randomly with a per-clip RNG.
Because WiTA has many clips per subject (75 lexical + 15 non-lexical per
subject), random splitting almost guarantees the same signer's writing
appears in both train and val.  CER then partially measures "same-signer
generalisation" rather than "new-signer generalisation," inflating
optimism in the reported numbers.

This module provides:
  - subject_id_from_zip(name) -> 'CYB' from 'CYB_Female_20_eng_freq_word.zip'
  - stream_and_index_with_subjects(cfg) -> list[(frames, label, subject_id)]
  - build_subject_split(samples, train_ratio, seed) -> (train_idx, val_idx, manifest)
  - save_split_manifest / load_split_manifest

The manifest is a JSON file that records which subjects went to train
vs val, plus per-subject clip counts.  Saving it makes the split
reproducible across runs and inspectable in the dissertation.
"""

from __future__ import annotations

import io
import os
import json
import shutil
import zipfile
import logging
import random
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subject ID extraction
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^([A-Z]{2,4})_")


def subject_id_from_zip(zip_name: str) -> str:
    """
    Extract the participant initials from a WiTA zip filename.

    Examples
    --------
    'CYB_Female_20_eng_freq_word.zip' -> 'CYB'
    'data/HJH_Male_30_eng_non_lex.zip' -> 'HJH'
    """
    base = os.path.basename(zip_name)
    m = _SUBJECT_RE.match(base)
    if not m:
        raise ValueError(
            f"Could not extract subject ID from zip name: {zip_name!r}. "
            "Expected pattern like 'XXX_Female_20_eng_freq_word.zip'."
        )
    return m.group(1)


# ---------------------------------------------------------------------------
# Streaming with subject IDs
# ---------------------------------------------------------------------------

def stream_and_index_with_subjects(
    cfg,
) -> list[tuple[list[bytes], str, str]]:
    """
    Mirror of `stream_and_index` from dataset.py, but emits subject IDs.

    Returns
    -------
    list of (frame_bytes, label, subject_id)
    """
    # Import here to avoid circular import at module load time.
    from .dataset import _index_zip
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    repo_id    = cfg.data.hf_repo_id
    lang       = cfg.data.lang
    max_zips   = cfg.data.max_zips
    dl_dir     = cfg.data.download_dir
    hf_cache   = cfg.data.hf_cache_dir
    max_frames = cfg.data.max_frames

    os.makedirs(dl_dir, exist_ok=True)

    all_files = list(list_repo_files(repo_id, repo_type="dataset"))
    lang_filters = {
        "english": lambda f: f.endswith(".zip") and "kor" not in f.lower(),
        "korean":  lambda f: f.endswith(".zip") and ("kor" in f.lower() or "korean" in f.lower()),
        "both":    lambda f: f.endswith(".zip"),
    }
    zip_files = sorted(filter(lang_filters[lang], all_files))

    if max_zips is not None:
        zip_files = zip_files[:max_zips]

    logger.info("Streaming %d ZIPs with subject-ID tracking from %s …",
                len(zip_files), repo_id)

    all_samples: list[tuple[list[bytes], str, str]] = []

    for i, zip_name in enumerate(zip_files):
        subject_id = subject_id_from_zip(zip_name)
        logger.info("[%d/%d] %s  subject=%s", i + 1, len(zip_files),
                    zip_name, subject_id)

        local = hf_hub_download(
            repo_id=repo_id, filename=zip_name, repo_type="dataset",
            local_dir=dl_dir, local_dir_use_symlinks=False,
        )
        try:
            with zipfile.ZipFile(local, "r") as zf:
                batch = _index_zip(zf, lang, max_frames)
            for frames, label in batch:
                all_samples.append((frames, label, subject_id))
            logger.info("  → %d clips  (total: %d, subject=%s)",
                        len(batch), len(all_samples), subject_id)
        except Exception as e:
            logger.error("Failed to process %s: %s", zip_name, e)
        finally:
            if os.path.exists(local):
                os.remove(local)
            if os.path.exists(hf_cache):
                shutil.rmtree(hf_cache, ignore_errors=True)
            os.makedirs(hf_cache, exist_ok=True)

        free_gb = shutil.disk_usage("/").free / (1024 ** 3)
        if free_gb < 2.0:
            raise RuntimeError(f"Disk critically low ({free_gb:.2f} GB).")

    logger.info("Indexing complete: %d total clips across %d subjects",
                len(all_samples),
                len({s for _, _, s in all_samples}))
    return all_samples


# ---------------------------------------------------------------------------
# Subject-disjoint split
# ---------------------------------------------------------------------------

def build_subject_split(
    samples:     list[tuple],
    train_ratio: float = 0.90,
    seed:        int = 42,
) -> tuple[list[int], list[int], dict]:
    """
    Partition `samples` into train and val by SUBJECT, not by clip.

    Parameters
    ----------
    samples      : list whose entries have subject_id at index 2.
                   (Compatible with both 2-tuple and 3-tuple sample lists, as
                    long as 3-tuple entries are passed.)
    train_ratio  : fraction of UNIQUE SUBJECTS to use for training.
    seed         : RNG seed for the subject permutation.

    Returns
    -------
    train_idx : list[int]   indices into `samples` belonging to train subjects
    val_idx   : list[int]   indices into `samples` belonging to val subjects
    manifest  : dict with keys
                  'train_subjects'    : list[str]
                  'val_subjects'      : list[str]
                  'n_train_clips'     : int
                  'n_val_clips'       : int
                  'clips_per_subject' : {subject_id: int}
                  'train_ratio'       : float
                  'seed'              : int
    """
    if not samples:
        raise ValueError("Empty samples list passed to build_subject_split.")
    if not (isinstance(samples[0], tuple) and len(samples[0]) >= 3):
        raise ValueError(
            "samples must be 3-tuples (frame_bytes, label, subject_id). "
            "Use stream_and_index_with_subjects(cfg) to produce these."
        )

    subj_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        subj_to_indices[s[2]].append(i)

    subjects = sorted(subj_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_train_subj = max(1, int(round(len(subjects) * train_ratio)))
    train_subjects = sorted(subjects[:n_train_subj])
    val_subjects   = sorted(subjects[n_train_subj:])

    train_idx: list[int] = []
    val_idx:   list[int] = []
    for s in train_subjects:
        train_idx.extend(subj_to_indices[s])
    for s in val_subjects:
        val_idx.extend(subj_to_indices[s])

    manifest = {
        "train_subjects":    train_subjects,
        "val_subjects":      val_subjects,
        "n_train_clips":     len(train_idx),
        "n_val_clips":       len(val_idx),
        "n_subjects_total":  len(subjects),
        "n_subjects_train":  len(train_subjects),
        "n_subjects_val":    len(val_subjects),
        "clips_per_subject": {s: len(idxs) for s, idxs in subj_to_indices.items()},
        "train_ratio":       train_ratio,
        "seed":              seed,
    }

    # Defensive sanity check: no overlap.
    assert not (set(train_subjects) & set(val_subjects)), \
        "Internal error: train/val subject overlap"

    logger.info(
        "[subject_split] %d subjects → %d train (%d clips) / %d val (%d clips)",
        len(subjects), len(train_subjects), len(train_idx),
        len(val_subjects), len(val_idx),
    )
    return train_idx, val_idx, manifest


def save_split_manifest(manifest: dict, path: str) -> None:
    """Write the split manifest as a human-readable JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("[subject_split] Saved manifest → %s", path)


def load_split_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def apply_split_from_manifest(
    samples:  list[tuple],
    manifest: dict,
) -> tuple[list[int], list[int]]:
    """
    Re-apply a saved manifest to a freshly-streamed `samples` list.

    Subjects in the manifest's train_subjects go to train, val_subjects to val.
    Subjects in `samples` not in either list are dropped with a warning
    (e.g. a future run that includes new subjects).
    """
    train_set = set(manifest["train_subjects"])
    val_set   = set(manifest["val_subjects"])

    train_idx: list[int] = []
    val_idx:   list[int] = []
    seen_unknown: set[str] = set()

    for i, s in enumerate(samples):
        subj = s[2]
        if subj in train_set:
            train_idx.append(i)
        elif subj in val_set:
            val_idx.append(i)
        else:
            seen_unknown.add(subj)

    if seen_unknown:
        logger.warning(
            "[subject_split] %d subjects in samples are not in manifest, "
            "dropping their clips: %s",
            len(seen_unknown), sorted(seen_unknown),
        )

    return train_idx, val_idx

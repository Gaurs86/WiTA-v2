"""
datasets/cv_splits.py — 5-fold subject-disjoint cross-validation for WiTA.

Plan §4.1
---------
The dataset has 39 signers.  We construct 5 folds by:
  1. Seeding a random permutation of the 39 signer IDs (seed=42).
  2. Slicing the permutation into 5 chunks of 8 / 8 / 8 / 8 / 7 signers.
  3. Each fold uses one chunk as val (held-out signers) and the other 31/32
     as train.  Each signer appears in exactly ONE val fold.

The manifest is materialised as JSON with:
  {
    "permutation": [...39 signer IDs...],   # deterministic from seed
    "folds": [
      {"fold": 0, "train_subjects": [...], "val_subjects": [...]},
      ...
      {"fold": 4, ...}
    ],
    "seed": 42,
    "n_folds": 5
  }

The per-fold local signer index map (signer_id -> 0..n_train_signers-1) is
NOT stored in the manifest because it is fold-local and depends on the
training set; it is computed by `build_fold_signer_map()` on demand.
"""

from __future__ import annotations

import os
import json
import random
import logging

logger = logging.getLogger(__name__)


def build_cv5_manifest(
    samples:  list[tuple],
    n_folds:  int = 5,
    seed:     int = 42,
) -> dict:
    """
    Build the 5-fold subject-disjoint manifest.

    Parameters
    ----------
    samples : list of (frame_bytes, label, subject_id) tuples — from
              subject_splits.stream_and_index_with_subjects(cfg).
    n_folds : default 5 (plan §4.1).
    seed    : RNG seed for the signer permutation.

    Returns
    -------
    Manifest dict (also savable via save_cv5_manifest).
    """
    if not samples or len(samples[0]) < 3:
        raise ValueError("Samples must be 3-tuples (frames, label, subject_id).")

    all_subjects = sorted({s[2] for s in samples})
    n_subj = len(all_subjects)
    if n_subj < n_folds:
        raise ValueError(
            f"Need at least n_folds={n_folds} subjects, only got {n_subj}."
        )

    rng = random.Random(seed)
    permutation = list(all_subjects)
    rng.shuffle(permutation)

    # Slice into n_folds nearly-equal chunks
    base = n_subj // n_folds
    rem  = n_subj %  n_folds
    chunks: list[list[str]] = []
    idx = 0
    for f in range(n_folds):
        size = base + (1 if f < rem else 0)
        chunks.append(sorted(permutation[idx : idx + size]))
        idx += size

    # Map subject → number of clips (for diagnostics)
    clips_per_subj: dict[str, int] = {}
    for s in samples:
        clips_per_subj[s[2]] = clips_per_subj.get(s[2], 0) + 1

    folds: list[dict] = []
    for f in range(n_folds):
        val_subjects   = chunks[f]
        train_subjects = sorted([s for s in all_subjects if s not in val_subjects])
        n_train_clips  = sum(clips_per_subj[s] for s in train_subjects)
        n_val_clips    = sum(clips_per_subj[s] for s in val_subjects)
        folds.append({
            "fold":            f,
            "train_subjects":  train_subjects,
            "val_subjects":    val_subjects,
            "n_train_subjects": len(train_subjects),
            "n_val_subjects":   len(val_subjects),
            "n_train_clips":   n_train_clips,
            "n_val_clips":     n_val_clips,
        })

    manifest = {
        "n_folds":           n_folds,
        "seed":              seed,
        "permutation":       permutation,
        "n_subjects_total":  n_subj,
        "clips_per_subject": clips_per_subj,
        "folds":             folds,
    }

    # Defensive: every signer should appear in exactly one val fold.
    val_seen: dict[str, int] = {}
    for f in folds:
        for s in f["val_subjects"]:
            val_seen[s] = val_seen.get(s, 0) + 1
    assert all(v == 1 for v in val_seen.values()), \
        "Internal error: signer in >1 val fold"
    assert sorted(val_seen.keys()) == all_subjects, \
        "Internal error: not all signers covered"

    logger.info(
        "[cv5] built %d folds over %d signers (seed=%d).  "
        "Per-fold val sizes: %s",
        n_folds, n_subj, seed,
        [f["n_val_subjects"] for f in folds],
    )
    return manifest


def save_cv5_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("[cv5] saved manifest → %s", path)


def load_cv5_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-fold helpers
# ---------------------------------------------------------------------------

def fold_indices(
    manifest:        dict,
    fold:            int,
    subject_per_idx: list[str],
) -> tuple[list[int], list[int]]:
    """
    Given a cache's per-clip subject list and a fold number, return
    (train_indices, val_indices) into that cache.
    """
    fold_info = manifest["folds"][fold]
    train_set = set(fold_info["train_subjects"])
    val_set   = set(fold_info["val_subjects"])
    train_idx = [i for i, s in enumerate(subject_per_idx) if s in train_set]
    val_idx   = [i for i, s in enumerate(subject_per_idx) if s in val_set]
    return train_idx, val_idx


def build_fold_signer_map(manifest: dict, fold: int) -> dict[str, int]:
    """
    Map training-fold signer IDs to 0..n_train_signers-1 local indices.
    The signer-adversarial head's output dim matches len(this dict).

    The map is built deterministically by sorting train_subjects so that the
    same training signer always gets the same local index within a fold.
    """
    train_subjects = sorted(manifest["folds"][fold]["train_subjects"])
    return {s: i for i, s in enumerate(train_subjects)}

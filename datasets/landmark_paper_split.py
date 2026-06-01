"""
datasets/landmark_paper_split.py — WiTA paper 8:1:1 split dataset.

Consumes the per-clip .npz layout produced by `landmark_cache_122.py`:

    landmark_cache_122/
        train/{lex, nonlex}/<SIGNER>__<clip_id>.npz
        val/{lex, nonlex}/<SIGNER>__<clip_id>.npz
        test/{lex, nonlex}/<SIGNER>__<clip_id>.npz

Three loaders are built from a single class — same signature contract as
the existing skeleton cache, but with `signer`, `subset`, `clip_id`
preserved in the batch metadata so downstream evaluation can do
per-signer / per-subset / length-bucketed CER without re-walking files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


_VALID_SPLITS  = {"train", "val", "test"}
_VALID_SUBSETS = {"lex", "nonlex"}


class WiTAPaperSplitDataset(Dataset):
    """
    Per-clip lazy-load dataset for the paper's 8:1:1 person-split.

    Parameters
    ----------
    cache_root : path containing train/, val/, test/ subdirs
    split      : 'train' | 'val' | 'test'
    subsets    : ('lex',), ('nonlex',), or both ('lex', 'nonlex').
                 Both is the default — matches the paper's overall CER.
    converter  : StrLabelConverter (lazily built from cfg if None)
    transform  : optional callable on the [T, 190] feature tensor
                 (typically LandmarkAugment for train only)
    """

    def __init__(
        self,
        cache_root: str | Path,
        split:      str,
        subsets:    Sequence[str] = ("lex", "nonlex"),
        *,
        converter=None,
        transform=None,
        lang:       str = "english",
    ):
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS}, got {split!r}")
        for s in subsets:
            if s not in _VALID_SUBSETS:
                raise ValueError(f"subset must be in {_VALID_SUBSETS}, got {s!r}")
        self.cache_root = Path(cache_root)
        self.split      = split
        self.subsets    = tuple(subsets)
        self.transform  = transform

        # Lazy converter.
        if converter is None:
            from .vocab import make_converter
            converter = make_converter(lang)
        self.converter = converter

        # Build the entry list deterministically (sorted) so seeds reproduce.
        self.entries: list[dict] = []
        for subset in self.subsets:
            d = self.cache_root / split / subset
            if not d.exists():
                raise FileNotFoundError(
                    f"Missing subset dir: {d} (expected per-clip .npz files)"
                )
            for npz_path in sorted(d.glob("*.npz")):
                # Filename convention: <SIGNER>__<clip_id>.npz
                stem = npz_path.stem
                if "__" not in stem:
                    logger.warning("Unexpected filename %s; skipping.", npz_path)
                    continue
                signer_id, clip_id = stem.split("__", 1)
                self.entries.append({
                    "path":    str(npz_path),
                    "signer":  signer_id,
                    "clip_id": clip_id,
                    "subset":  subset,
                })

        if not self.entries:
            raise RuntimeError(
                f"No clips found under {self.cache_root}/{split} for "
                f"subsets={self.subsets}"
            )
        logger.info(
            "[WiTAPaperSplitDataset] %s/%s: %d clips across %d signers",
            split, "+".join(self.subsets),
            len(self.entries),
            len({e["signer"] for e in self.entries}),
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        with np.load(e["path"], allow_pickle=False) as data:
            feat  = data["feature"]              # [T, 190] fp16 by convention
            label = str(data["label"].item())
        feats = torch.from_numpy(feat).float()   # promote to fp32 for train
        if self.transform is not None:
            feats = self.transform(feats)
        enc, _ = self.converter.encode(label)
        return feats, enc, e["signer"], e["subset"], label

    # ------------------------------------------------------------------

    @property
    def signers(self) -> list[str]:
        return sorted({e["signer"] for e in self.entries})

    def per_subset_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e["subset"]] = out.get(e["subset"], 0) + 1
        return out


# ---------------------------------------------------------------------------
# Sanity helpers (used by scripts/sanity_check_stage11.py)
# ---------------------------------------------------------------------------

def collect_signers(cache_root: str | Path, split: str) -> set[str]:
    """Return the set of distinct signer IDs under `cache_root/split`."""
    out: set[str] = set()
    d = Path(cache_root) / split
    if not d.exists():
        return out
    for sub in _VALID_SUBSETS:
        if not (d / sub).exists():
            continue
        for p in (d / sub).glob("*.npz"):
            stem = p.stem
            if "__" in stem:
                out.add(stem.split("__", 1)[0])
    return out


def collect_clip_ids(cache_root: str | Path, split: str) -> set[str]:
    """Return the set of distinct '<SIGNER>__<clip_id>' stems under split."""
    out: set[str] = set()
    d = Path(cache_root) / split
    if not d.exists():
        return out
    for sub in _VALID_SUBSETS:
        if (d / sub).exists():
            for p in (d / sub).glob("*.npz"):
                out.add(p.stem)
    return out

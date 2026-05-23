"""
datasets/dataset.py — WiTADataset and HuggingFace ZIP streaming pipeline.

Self-contained: no imports from the original WiTA repo.

NOTE ON HYPHENS IN THE DISPLAY
-------------------------------
The qualitative sample table shows ground-truth words like "sug-gestion",
"bre-eding", "com-mander".  These hyphens do NOT come from gt.txt — they
are produced by StrLabelConverter.decode() mapping the consecutive-repeat
separator token (index N+1 = '-') back to the '-' character.  For example,
'suggestion' encodes as [..., g, SEP(N+1), g, ...] and decodes as
"sug-gestion".  This is the baseline's intended representation and is NOT
a data error.  The root issue was in VocabConfig (see configs/default.py).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import zipfile
import random
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from ..configs.default import Config
from .vocab import StrLabelConverter, make_converter
from .augmentations import WiTAClipAugmentation
from .collate import make_pad_collate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Korean jamo decomposer
# ---------------------------------------------------------------------------

def _decompose_korean(label: str) -> str:
    try:
        import hgtk
        return hgtk.text.decompose(label)
    except ImportError:
        raise ImportError("pip install hgtk  (required for Korean data)")


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _sorted_frame_names(names: list[str]) -> list[str]:
    def _key(f: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(f))
        return int(m.group()) if m else 0
    return sorted(names, key=_key)


def _read_frames_from_zip(
    zf:         zipfile.ZipFile,
    seq_prefix: str,
    max_frames: int,
) -> list[bytes] | None:
    all_names = zf.namelist()
    frames = [
        n for n in all_names
        if n.startswith(seq_prefix)
        and n.lower().endswith(".png")
        and re.search(r"\d+", os.path.basename(n))
    ]
    if not frames:
        return None

    frames = _sorted_frame_names(frames)

    if len(frames) > max_frames:
        s = (len(frames) - max_frames) // 2
        frames = frames[s : s + max_frames]

    raw: list[bytes] = []
    for name in frames:
        try:
            raw.append(zf.read(name))
        except Exception as e:
            logger.warning("Failed to read frame %s: %s", name, e)

    return raw if raw else None


# ---------------------------------------------------------------------------
# gt.txt parser
# ---------------------------------------------------------------------------

def _parse_gt(zf: zipfile.ZipFile, gt_path: str, lang: str) -> list[str]:
    raw = zf.read(gt_path)
    if lang == "korean":
        try:
            lines = raw.decode("cp949").splitlines()
        except UnicodeDecodeError:
            lines = raw.decode("utf-8").splitlines()
    else:
        lines = raw.decode("utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# ZIP indexing
# ---------------------------------------------------------------------------

def _index_zip(
    zf:        zipfile.ZipFile,
    lang:      str,
    max_frames:int,
) -> list[tuple[list[bytes], str]]:
    all_names = zf.namelist()
    gt_files  = [n for n in all_names if n.endswith("gt.txt")]
    samples: list[tuple[list[bytes], str]] = []

    for gt_path in gt_files:
        parent = gt_path.rsplit("/gt.txt", 1)[0]
        try:
            labels = _parse_gt(zf, gt_path, lang)
        except Exception as e:
            logger.warning("Could not read %s: %s", gt_path, e)
            continue

        for idx, label in enumerate(labels):
            seq_prefix = f"{parent}/{idx}/"
            frames = _read_frames_from_zip(zf, seq_prefix, max_frames)
            if frames is None:
                logger.debug("No frames: %s line %d", gt_path, idx)
                continue
            samples.append((frames, label))

    return samples


# ---------------------------------------------------------------------------
# HuggingFace streaming pipeline
# ---------------------------------------------------------------------------

def stream_and_index(cfg: Config) -> list[tuple[list[bytes], str]]:
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

    logger.info("Streaming %d ZIPs from %s …", len(zip_files), repo_id)

    all_samples: list[tuple[list[bytes], str]] = []

    for i, zip_name in enumerate(zip_files):
        logger.info("[%d/%d] Downloading %s", i + 1, len(zip_files), zip_name)
        local = hf_hub_download(
            repo_id=repo_id, filename=zip_name, repo_type="dataset",
            local_dir=dl_dir, local_dir_use_symlinks=False,
        )
        try:
            with zipfile.ZipFile(local, "r") as zf:
                batch = _index_zip(zf, lang, max_frames)
            all_samples.extend(batch)
            logger.info("  → %d clips  (total so far: %d)", len(batch), len(all_samples))
        except Exception as e:
            logger.error("Failed to process %s: %s", zip_name, e)
        finally:
            if os.path.exists(local):
                os.remove(local)
            if os.path.exists(hf_cache):
                shutil.rmtree(hf_cache, ignore_errors=True)
            os.makedirs(hf_cache, exist_ok=True)

        free_gb = shutil.disk_usage("/").free / (1024 ** 3)
        logger.info("  Disk free: %.2f GB", free_gb)
        if free_gb < 2.0:
            raise RuntimeError(f"Disk critically low ({free_gb:.2f} GB). Stopping.")

    logger.info("Indexing complete: %d total clips", len(all_samples))
    return all_samples


# ---------------------------------------------------------------------------
# WiTADataset
# ---------------------------------------------------------------------------

class WiTADataset(Dataset):
    def __init__(
        self,
        samples:   list[tuple[list[bytes], str]],
        cfg:       Config,
        mode:      str = "train",
        converter: StrLabelConverter | None = None,
    ):
        self.samples   = samples
        self.cfg       = cfg
        self.mode      = mode
        self.lang      = cfg.data.lang
        self.converter = converter or make_converter(cfg.data.lang)
        self.augment   = WiTAClipAugmentation(cfg.aug, cfg.data, mode=mode)

        img_size = cfg.data.img_size
        self.resize = T.Resize(img_size) if img_size != 224 else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_bytes, raw_label = self.samples[idx]

        frames: list[Image.Image] = [
            Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes
        ]
        if self.resize is not None:
            frames = [self.resize(f) for f in frames]

        clip_ctw = self.augment(frames)
        clip = clip_ctw.permute(1, 0, 2, 3)   # [T, C, H, W]

        label = raw_label
        if self.lang == "korean":
            label = _decompose_korean(label)
        label_enc, _ = self.converter.encode(label)

        return clip, label_enc


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    cfg:       Config,
    samples:   list[tuple[list[bytes], str]] | None = None,
    converter: StrLabelConverter | None = None,
) -> tuple[DataLoader, DataLoader]:
    if samples is None:
        samples = stream_and_index(cfg)

    converter = converter or make_converter(cfg.data.lang)

    rng  = random.Random(cfg.data.seed)
    data = samples.copy()
    rng.shuffle(data)
    split = int(len(data) * cfg.data.train_split)
    train_data, val_data = data[:split], data[split:]

    train_ds = WiTADataset(train_data, cfg, mode="train", converter=converter)
    val_ds   = WiTADataset(val_data,   cfg, mode="val",   converter=converter)

    # Pass pad_idx so labels are padded with the correct token (Bug 2 fix).
    collate = make_pad_collate(
        arch=cfg.encoder.arch,
        lang=cfg.data.lang,
        num_res_layers=cfg.encoder.num_res_layers,
        pad_idx=cfg.vocab.pad_idx,   # N+4, NOT 0
    )

    pin     = cfg.train.pin_memory and (cfg.train.num_workers == 0)
    persist = cfg.train.num_workers > 0 and cfg.train.persistent_workers

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=pin, persistent_workers=persist, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=pin, persistent_workers=persist, drop_last=False,
    )

    logger.info(
        "DataLoaders ready: train=%d  val=%d  workers=%d  pin=%s",
        len(train_ds), len(val_ds), cfg.train.num_workers, pin,
    )
    return train_loader, val_loader

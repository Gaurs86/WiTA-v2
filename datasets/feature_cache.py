"""
datasets/feature_cache.py — SigLIP feature extraction + cached dataset.

Workflow
--------
1) `extract_siglip_features(samples, encoder, cfg, out_path)`
       Walks the indexed samples list, decodes PNG frame bytes, normalizes
       with SigLIP stats, runs the (frozen) SigLIP vision encoder, and saves
       a single .pt file containing:
           {
             "feats":      list of [T_i, D] CPU float16 tensors
             "labels":     list of str (raw labels)
             "lengths":    LongTensor [N]  (T_i per clip)
             "model_name": str (provenance)
             "out_dim":    int (D)
           }
       Variable T per clip is preserved — features are NOT padded at save time.

2) `CachedFeatureDataset(cache_path, lang, converter)`
       Loads the cache and serves (feats [T, D], label_enc [L]) tuples.
       NO augmentation — train and val use identical features.

3) `make_cached_dataloaders(cfg, cache_path, converter)`
       Splits cached features into train/val by cfg.data.seed/train_split and
       returns ready-to-use DataLoaders with the SigLIP-aware collate.

Why no augmentation
-------------------
Augmentation would change pixel-level inputs which would invalidate cached
features.  Re-encoding on-the-fly defeats the purpose of caching.  For the
1-2 month dissertation timeline, the trade-off is intentional: prioritize
fast iteration over augmentation gains.  If results saturate we can add an
"augmented cache" variant later (extract K augmented copies per clip).

Disk footprint
--------------
With ~3500 clips × ~32 frames × 1152 dim × 2 bytes (fp16) ≈ 240 MB.
Cache fits comfortably in Kaggle's /kaggle/working/ (20 GB budget).
"""

from __future__ import annotations

import io
import os
import logging
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms.functional as TF

from ..configs.default import Config
from .vocab import StrLabelConverter, make_converter
from ..models.encoders.siglip_encoder import SigLIPVisionEncoder, default_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame preprocessing for SigLIP
# ---------------------------------------------------------------------------

def _pil_to_siglip_tensor(
    img:        Image.Image,
    image_size: int,
) -> torch.Tensor:
    """
    PIL → [3, H, W] float32 in [-1, 1] (SigLIP normalization).
    Resize to (image_size, image_size).  Bilinear, antialias on.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    t = TF.to_tensor(img)                # [3, H, W] in [0, 1]
    return default_normalize(t)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_siglip_features(
    samples:    list[tuple[list[bytes], str]],
    encoder:    SigLIPVisionEncoder,
    out_path:   str,
    batch_size: int = 16,
    device:     str | torch.device = "cuda",
    dtype:      torch.dtype = torch.float16,
) -> dict:
    """
    Extract SigLIP features for every clip and save to disk.

    Parameters
    ----------
    samples    : output of stream_and_index() — list of (frame_bytes, label).
    encoder    : a frozen SigLIPVisionEncoder.
    out_path   : where to save the .pt cache.
    batch_size : how many FRAMES to push through SigLIP at once (memory knob).
    device     : 'cuda' or 'cpu'.
    dtype      : fp16 by default to halve cache size with negligible loss.

    Returns
    -------
    The same dict that was saved to disk.
    """
    import sys, time
    encoder = encoder.to(device).eval()
    image_size = encoder.image_size
    D = encoder.out_dim

    feats_list:  list[torch.Tensor] = []
    labels_list: list[str]          = []
    lengths:     list[int]          = []

    total_clips = len(samples)
    # Visible progress every ~1% (or every clip for tiny smoke tests).
    # Uses print(..., flush=True) instead of logger so Jupyter cells show it
    # immediately even if logging is misconfigured by the host.
    log_every   = max(1, total_clips // 100)
    skipped     = 0
    t0          = time.time()
    print(f"[feature_cache] Encoding {total_clips} clips with {encoder.model_name} "
          f"on {device} (image_size={image_size}, dtype={dtype})", flush=True)

    for ci, (frame_bytes, label) in enumerate(samples):
        try:
            frames = [
                _pil_to_siglip_tensor(
                    Image.open(io.BytesIO(b)), image_size
                )
                for b in frame_bytes
            ]
        except Exception as e:
            logger.warning("Skipping clip %d (decode error): %s", ci, e)
            skipped += 1
            continue

        if not frames:
            skipped += 1
            continue

        x = torch.stack(frames, dim=0).to(device, non_blocking=True)   # [T, 3, H, W]

        chunks: list[torch.Tensor] = []
        for i in range(0, x.shape[0], batch_size):
            chunks.append(encoder(x[i : i + batch_size]))              # [b, D]
        clip_feats = torch.cat(chunks, dim=0).to(dtype).cpu().contiguous()  # [T, D]

        feats_list.append(clip_feats)
        labels_list.append(label)
        lengths.append(clip_feats.shape[0])

        if (ci + 1) % log_every == 0 or (ci + 1) == total_clips:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total_clips - (ci + 1)) / max(rate, 1e-3)
            print(
                f"[feature_cache] {ci + 1}/{total_clips} clips  "
                f"({100*(ci+1)/total_clips:5.1f}%)  "
                f"{rate:.1f} clips/s  ETA {eta/60:5.1f} min  "
                f"skipped={skipped}",
                flush=True,
            )

    cache = {
        "feats":      feats_list,
        "labels":     labels_list,
        "lengths":    torch.tensor(lengths, dtype=torch.long),
        "model_name": encoder.model_name,
        "out_dim":    D,
        "image_size": image_size,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    total_mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    logger.info(
        "[feature_cache] Saved %d clips → %s  (%.1f MB feats, %s)",
        len(feats_list), out_path, total_mb, dtype,
    )
    return cache


# ---------------------------------------------------------------------------
# Cached dataset
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    """
    Loads pre-extracted SigLIP features.

    __getitem__ returns
    --------------------
    (feats [T_i, D] float32, label_enc [L_i] int32)

    feats are auto-converted from cache dtype (fp16) to fp32.
    """

    def __init__(
        self,
        cache:     dict,
        indices:   list[int],
        lang:      str,
        converter: StrLabelConverter | None = None,
    ):
        self.cache     = cache
        self.indices   = indices
        self.lang      = lang
        self.converter = converter or make_converter(lang)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        feats: torch.Tensor = self.cache["feats"][idx].float()    # [T, D]
        label_str: str      = self.cache["labels"][idx]

        if self.lang == "korean":
            from .dataset import _decompose_korean
            label_str = _decompose_korean(label_str)

        label_enc, _ = self.converter.encode(label_str)
        return feats, label_enc


# ---------------------------------------------------------------------------
# Collate for cached features
# ---------------------------------------------------------------------------

def _cached_collate(
    batch:    list[tuple[torch.Tensor, torch.Tensor]],
    pad_idx:  int,
):
    """
    Pad variable-length [T, D] feature tensors along T with zeros.
    Labels padded with pad_idx (matches StrLabelConverter convention).

    Returns
    -------
    feats_pad  : [B, T_max, D] float32
    labels_pad : [B, L_max] int32
    input_lens : [B] long
    label_lens : [B] long
    """
    feats, labels = zip(*batch)
    feats_pad  = pad_sequence(feats,  batch_first=True, padding_value=0.0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.LongTensor([f.shape[0] for f in feats])
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return feats_pad, labels_pad, input_lens, label_lens


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_cached_dataloaders(
    cfg:         Config,
    cache_path:  str,
    converter:   StrLabelConverter | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Load the SigLIP feature cache, train/val split, return DataLoaders.

    Also returns the loaded cache dict so callers can inspect metadata
    (out_dim, model_name) before building the model.
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    n_clips = len(cache["feats"])
    if n_clips == 0:
        raise RuntimeError(f"Empty feature cache: {cache_path}")

    converter = converter or make_converter(cfg.data.lang)

    rng    = random.Random(cfg.data.seed)
    order  = list(range(n_clips))
    rng.shuffle(order)
    split  = int(n_clips * cfg.data.train_split)
    train_idx, val_idx = order[:split], order[split:]

    train_ds = CachedFeatureDataset(cache, train_idx, cfg.data.lang, converter)
    val_ds   = CachedFeatureDataset(cache, val_idx,   cfg.data.lang, converter)

    pad_idx = cfg.vocab.pad_idx
    collate = lambda b: _cached_collate(b, pad_idx=pad_idx)

    persist = cfg.train.num_workers > 0 and cfg.train.persistent_workers
    prefetch_kw = {"prefetch_factor": 2} if cfg.train.num_workers > 0 else {}

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=False, persistent_workers=persist, drop_last=True,
        **prefetch_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=False, persistent_workers=persist, drop_last=False,
        **prefetch_kw,
    )

    logger.info(
        "[feature_cache] DataLoaders ready: train=%d  val=%d  D=%d  source=%s",
        len(train_ds), len(val_ds), cache["out_dim"], cache.get("model_name", "?"),
    )
    return train_loader, val_loader, cache

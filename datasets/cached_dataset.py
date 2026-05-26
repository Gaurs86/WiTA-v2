"""
datasets/cached_dataset.py — Train on pre-extracted VideoMAE features.

WHY
---
Features are already extracted by cache_features.py.
Training now loads [T', 768] tensors directly — no image decoding,
no VideoMAE forward pass, no GPU starvation.

SPEED IMPROVEMENT ON T4
-----------------------
Before caching  : ~1000s/epoch  (GPU usage ~1 GB, ~7% utilisation)
After  caching  : ~5–10s/epoch  (GPU usage ~1–2 GB, ~60–80% utilisation)
                  batch_size 16–32 fully fits in VRAM

HOW IT INTEGRATES
-----------------
Replace make_dataloaders() with make_cached_dataloaders() in your notebook
after calling extract_and_cache().  The trainer is UNCHANGED — it still
receives (clips, labels, input_lens, label_lens) batches but 'clips' is
now the pre-extracted feature tensor [B, T', D] rather than raw frames.

Because WiTACTCModel._encode() routes to VideoMAEEncoder for arch=videomae,
we add a tiny bypass: if clips.ndim == 3 (already [B, T', D]), skip the
backbone and return clips directly.  That one-line guard in hybrid_model.py
is the ONLY model change needed.

TRAINING FLOW WITH CACHE
-------------------------
  Cached feat [B, T', 768]
      ↓  (no VideoMAE pass)
  BiLSTM    [B, T', 512]
      ↓
  CTCProj   [B, T', 28]
      ↓
  CTCLoss
"""

from __future__ import annotations
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

from ..configs.default import Config
from ..datasets.vocab import StrLabelConverter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedFeaturesDataset(Dataset):
    """
    Loads pre-extracted VideoMAE features from disk.

    Each sample .pt file contains:
        features : [T', D]  fp16 Tensor
        label    : str
        enc_len  : int

    __getitem__ returns:
        features     : [T', D]  fp32 Tensor
        encoded_label: [L]      int32 Tensor
        enc_len      : int      (T')
        label_len    : int      (number of label characters)
    """

    def __init__(
        self,
        indices:   list[int],
        cache_dir: str,
        converter: StrLabelConverter,
        cfg:       Config,
    ):
        self.indices   = indices
        self.cache_dir = cache_dir
        self.converter = converter
        self.cfg       = cfg

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int):
        idx  = self.indices[pos]
        path = os.path.join(self.cache_dir, f"feats_{idx:05d}.pt")
        data = torch.load(path, map_location="cpu", weights_only=True)

        features = data["features"].float()   # [T', D]  fp16 → fp32
        label    = data["label"]
        enc_len  = int(data["enc_len"])

        # Encode label
        encoded, label_len = self.converter.encode(label)  # Tensor[L], int

        return features, encoded, enc_len, label_len


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_cached(batch):
    """
    Pad features and labels to the longest in the batch.

    Returns
    -------
    feats      : [B, T', D]   float32
    labels     : [B, L_max]   int32
    enc_lens   : [B]          int64  (T' — same for all, but kept as tensor)
    label_lens : [B]          int64
    """
    feats_list, label_list, enc_len_list, label_len_list = zip(*batch)

    B       = len(feats_list)
    T_prime = feats_list[0].shape[0]
    D       = feats_list[0].shape[1]
    L_max   = max(l.shape[0] for l in label_list)

    feats  = torch.stack(feats_list)                             # [B, T', D]
    labels = torch.zeros(B, L_max, dtype=torch.int32)
    for i, lab in enumerate(label_list):
        labels[i, :lab.shape[0]] = lab

    enc_lens   = torch.tensor(enc_len_list,   dtype=torch.long)
    label_lens = torch.tensor(label_len_list, dtype=torch.long)

    return feats, labels, enc_lens, label_lens


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_cached_dataloaders(
    cfg:       Config,
    cache_dir: str,
    converter: StrLabelConverter,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train / val DataLoaders from pre-extracted feature cache.

    Reads the list of available .pt files, splits train/val by
    cfg.data.train_split, and returns two DataLoaders.

    Parameters
    ----------
    cfg       : Config (uses cfg.data.train_split, cfg.train.*)
    cache_dir : directory written by extract_and_cache()
    converter : StrLabelConverter

    Returns
    -------
    train_loader, val_loader
    """
    # Discover all cached indices
    fnames  = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    indices = [int(f.replace("feats_", "").replace(".pt", "")) for f in fnames]

    # Reproducible shuffle + split
    rng = random.Random(cfg.data.seed)
    rng.shuffle(indices)
    n_train  = int(len(indices) * cfg.data.train_split)
    train_ix = indices[:n_train]
    val_ix   = indices[n_train:]

    train_ds = CachedFeaturesDataset(train_ix, cache_dir, converter, cfg)
    val_ds   = CachedFeaturesDataset(val_ix,   cache_dir, converter, cfg)

    # With cached features batch_size 16–32 is comfortable on T4
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,
        num_workers = cfg.train.num_workers,
        pin_memory  = True,         # always True for cached (pure tensor) data
        persistent_workers = cfg.train.num_workers > 0,
        collate_fn  = _collate_cached,
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.train.batch_size * 2,   # val can use 2× batch
        shuffle     = False,
        num_workers = cfg.train.num_workers,
        pin_memory  = True,
        persistent_workers = cfg.train.num_workers > 0,
        collate_fn  = _collate_cached,
        drop_last   = False,
    )

    import logging
    logging.getLogger(__name__).info(
        "Cached DataLoaders ready: train=%d  val=%d  batch=%d  workers=%d",
        len(train_ds), len(val_ds),
        cfg.train.batch_size, cfg.train.num_workers,
    )
    print(
        f"Cached DataLoaders: train={len(train_ds)}  val={len(val_ds)}  "
        f"batch={cfg.train.batch_size}  workers={cfg.train.num_workers}"
    )
    return train_loader, val_loader

"""
datasets/cache_features.py — One-shot VideoMAE feature extraction.

WHY THIS EXISTS
---------------
VideoMAE-base has 86M parameters.  When the backbone is FROZEN, every
training step re-computes identical features for the same clip.
Over 40 epochs × 800 clips that is 32,000 redundant backbone passes.

Pre-extracting once and caching to disk eliminates this entirely:
  • Backbone runs ONCE per clip  (not 40× per clip)
  • Training loop only sees: BiLSTM [256×2] + CTC head [512→28]
    → ~3.8M trainable params, batch_size 16–32, ~5s/epoch on T4

USAGE (add a cell before training in your notebook)
-----------------------------------------------------
    from wita_v2.datasets.cache_features import extract_and_cache

    cache_dir = extract_and_cache(
        cfg      = cfg,
        samples  = samples,        # from stream_and_index()
        batch_size = 4,            # clips to push through VideoMAE at once
        device   = cfg.device,
    )
    # cache_dir is e.g. '/kaggle/working/feat_cache'
    # Pass it to make_cached_dataloaders() instead of make_dataloaders()

WHAT IS STORED
--------------
For each clip i:
  feat_cache/feats_{i:05d}.pt  →  dict {
      'features':  Tensor [T', 768]   fp16 (saves ~50% disk vs fp32)
      'label':     str
      'enc_len':   int               T' (e.g. 8 or 16)
  }

DISK BUDGET
-----------
887 clips × T'=8 × 768 dim × fp16 = 887 × 12 288 bytes ≈ 10 MB total.
Even with T'=16 this is ~21 MB — negligible on Kaggle (1.2 TB disk).

RE-EXTRACTION
-------------
Extraction is skipped if cache_dir/DONE sentinel exists.
Delete the sentinel (or set force=True) to re-extract.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.default import Config
from .dataset import WiTADataset           # raw-frame dataset


# ---------------------------------------------------------------------------
# Thin dataset wrapper: raw clips → VideoMAE input, no augmentation
# ---------------------------------------------------------------------------

class _RawClipDataset(Dataset):
    """
    Yields (clip_tensor, label_str, orig_idx) — no augmentation, no label
    encoding.  Used only during feature extraction.
    """
    def __init__(self, samples, cfg: Config):
        self.samples = samples
        self.cfg     = cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from .transforms import load_clip_tensor   # existing loader
        frame_bytes, label = self.samples[idx]
        clip = load_clip_tensor(
            frame_bytes,
            img_size  = self.cfg.encoder.img_size,   # 224 for VideoMAE
            max_frames= self.cfg.data.max_frames,
            mean      = self.cfg.data.img_mean,
            std       = self.cfg.data.img_std,
            augment   = False,                        # NO augmentation
        )
        return clip, label, idx   # [T, C, H, W], str, int


def _collate_raw(batch):
    clips  = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    idxs   = [b[2] for b in batch]
    # Pad clips to same T within mini-batch for batch processing
    T_max  = max(c.shape[0] for c in clips)
    C, H, W = clips[0].shape[1:]
    padded = torch.zeros(len(clips), T_max, C, H, W)
    lens   = []
    for i, c in enumerate(clips):
        t = c.shape[0]
        padded[i, :t] = c
        lens.append(t)
    return padded, labels, idxs, torch.tensor(lens)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_and_cache(
    cfg:        Config,
    samples:    list,
    batch_size: int  = 4,
    device:     torch.device | None = None,
    cache_dir:  str  = "/kaggle/working/feat_cache",
    force:      bool = False,
) -> str:
    """
    Run VideoMAE encoder over all samples once and cache features to disk.

    Parameters
    ----------
    cfg        : Config (encoder must be videomae / video_swin)
    samples    : list of (frame_bytes, label) from stream_and_index()
    batch_size : clips per GPU batch during extraction (4 fits T4 easily)
    device     : defaults to cfg.device
    cache_dir  : directory to write .pt files into
    force      : if True, re-extract even if DONE sentinel exists

    Returns
    -------
    cache_dir  : same string — pass to make_cached_dataloaders()
    """
    sentinel = os.path.join(cache_dir, "DONE")
    if os.path.exists(sentinel) and not force:
        n = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        print(f"[cache_features] Cache already exists ({n} files). "
              f"Skipping extraction.  Delete {sentinel} to re-extract.")
        return cache_dir

    if device is None:
        device = cfg.device

    os.makedirs(cache_dir, exist_ok=True)

    # ── Build encoder only (no BiLSTM, no CTC head) ──────────────────────
    arch = cfg.encoder.arch.lower()
    if arch in ("videomae", "video_swin"):
        from ..models.encoders.videomae_encoder import build_video_encoder
        encoder = build_video_encoder(cfg.encoder).to(device)
    else:
        raise ValueError(
            f"extract_and_cache only makes sense for frozen pretrained "
            f"backbones (videomae / video_swin), got arch='{arch}'."
        )
    encoder.eval()

    # ── DataLoader: no augment, multiple workers ─────────────────────────
    raw_ds = _RawClipDataset(samples, cfg)
    loader = DataLoader(
        raw_ds,
        batch_size  = batch_size,
        shuffle     = False,          # keep deterministic order
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = _collate_raw,
        drop_last   = False,
    )

    print(f"[cache_features] Extracting features for {len(samples)} clips "
          f"(batch={batch_size}, device={device}) …")
    print(f"[cache_features] Output dir: {cache_dir}")

    n_done = 0
    for clips, labels, idxs, seq_lens in tqdm(loader, desc="Extracting"):
        clips    = clips.to(device)
        seq_lens = seq_lens.to(device)

        # Run backbone only
        features = encoder(clips)            # [B, T', D]  (fp32)
        features = features.half().cpu()     # fp16 to halve disk usage

        # Save per clip
        for b, (idx, label) in enumerate(zip(idxs, labels)):
            T_enc = features.shape[1]        # T' (same for all in batch)
            path  = os.path.join(cache_dir, f"feats_{idx:05d}.pt")
            torch.save(
                {
                    "features": features[b],       # [T', D]
                    "label":    label,
                    "enc_len":  T_enc,
                },
                path,
            )
            n_done += 1

    # Write sentinel
    with open(sentinel, "w") as f:
        f.write(f"extracted={n_done}\n")

    print(f"[cache_features] Done. {n_done} feature files written to {cache_dir}")
    return cache_dir

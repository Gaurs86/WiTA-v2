"""
datasets/cache_features.py — One-shot VideoMAE feature extraction.
"""

from __future__ import annotations
import io
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.default import Config
from ..datasets.augmentations import WiTAClipAugmentation


class _RawClipDataset(Dataset):
    """
    Decodes frame bytes → [T, C, H, W] tensor.
    Uses WiTAClipAugmentation(mode='val') — ToTensor + Normalize only,
    no random augmentation.  Identical to what WiTADataset uses for val.
    """
    def __init__(self, samples, cfg: Config):
        self.samples = samples
        self.augment = WiTAClipAugmentation(cfg.aug, cfg.data, mode="val")
        img_size     = cfg.data.img_size          # 224 for VideoMAE
        self.resize  = T.Resize(img_size) if img_size != 224 else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_bytes, label = self.samples[idx]
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
        if self.resize is not None:
            frames = [self.resize(f) for f in frames]
        clip_ctw = self.augment(frames)            # [C, T, H, W]
        clip     = clip_ctw.permute(1, 0, 2, 3)   # [T, C, H, W]
        return clip, label, idx


def _collate_raw(batch):
    clips  = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    idxs   = [b[2] for b in batch]
    T_max  = max(c.shape[0] for c in clips)
    C, H, W = clips[0].shape[1:]
    padded = torch.zeros(len(clips), T_max, C, H, W)
    lens   = []
    for i, c in enumerate(clips):
        t = c.shape[0]
        padded[i, :t] = c
        lens.append(t)
    return padded, labels, idxs, torch.tensor(lens)


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
    Run VideoMAE encoder over all samples once and cache [T', D] features.

    Skips extraction if cache_dir/DONE sentinel exists (delete to re-run).
    Stores fp16 tensors — ~21 MB total for 887 clips at T'=16, D=768.

    Returns cache_dir — pass to make_cached_dataloaders().
    """
    sentinel = os.path.join(cache_dir, "DONE")
    if os.path.exists(sentinel) and not force:
        n = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        print(f"[cache_features] Cache already exists ({n} files). "
              f"Delete {sentinel} to re-extract.")
        return cache_dir

    if device is None:
        device = cfg.device
    os.makedirs(cache_dir, exist_ok=True)

    arch = cfg.encoder.arch.lower()
    if arch not in ("videomae", "video_swin"):
        raise ValueError(
            f"extract_and_cache requires a pretrained backbone "
            f"(videomae / video_swin), got '{arch}'."
        )
    from ..models.encoders.videomae_encoder import build_video_encoder
    encoder = build_video_encoder(cfg.encoder).to(device)
    encoder.eval()

    raw_ds = _RawClipDataset(samples, cfg)
    loader = DataLoader(
        raw_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = _collate_raw,
        drop_last   = False,
    )

    print(f"[cache_features] Extracting {len(samples)} clips "
          f"(batch={batch_size}, device={device}) → {cache_dir}")

    n_done = 0
    for clips, labels, idxs, seq_lens in tqdm(loader, desc="Extracting"):
        clips    = clips.to(device)
        features = encoder(clips)               # [B, T', D]
        features = features.half().cpu()        # fp16 saves 50% disk

        for b, (idx, label) in enumerate(zip(idxs, labels)):
            torch.save(
                {
                    "features": features[b],    # [T', D]
                    "label":    label,
                    "enc_len":  int(features.shape[1]),
                },
                os.path.join(cache_dir, f"feats_{idx:05d}.pt"),
            )
            n_done += 1

    with open(sentinel, "w") as f:
        f.write(f"extracted={n_done}\n")

    print(f"[cache_features] Done — {n_done} files written.")
    return cache_dir

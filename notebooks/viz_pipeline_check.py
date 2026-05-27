"""
Pipeline visualization — paste as a new Kaggle cell after Cell 5.

Shows for one random sample:
  Row 1: Raw frames straight from the ZIP (no aug, no resize, no normalize)
  Row 2: After augmentation (what the model actually sees, un-normalized for viewing)
  Row 3: After temporal resample to 32 frames (what VideoMAE sees)
  Row 4: A scrubbing strip — 8 evenly-spaced frames across the whole clip

If Row 1 doesn't show clear writing motion → data is the problem.
If Row 1 looks fine but Row 2 is unrecognizable → augmentation is too strong.
If Row 3 drops critical frames → temporal resampling is the problem.
"""
import io
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Pick a random sample from the indexed list (samples comes from Cell 4)
idx = random.randint(0, len(samples) - 1)
frame_bytes, label = samples[idx]
print(f"Sample {idx}: label='{label}'  raw_frames={len(frame_bytes)}  "
      f"frame_size_bytes={len(frame_bytes[0])}")

# ── Row 1: raw frames ─────────────────────────────────────────────────────
raw_frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
print(f"Raw frame dimensions: {raw_frames[0].size}  (W, H)")
print(f"Raw pixel range: min={np.array(raw_frames[0]).min()}  "
      f"max={np.array(raw_frames[0]).max()}  "
      f"mean={np.array(raw_frames[0]).mean():.1f}")

# ── Row 2: after augmentation (re-run dataset.__getitem__ logic) ──────────
from wita_v2.datasets.dataset import WiTADataset

# Build a one-sample dataset in TRAIN mode to apply augmentations
mini_ds = WiTADataset([(frame_bytes, label)], cfg, mode="train", converter=converter)
clip_tensor, _ = mini_ds[0]    # [T, C, H, W]  normalized

# Un-normalize for viewing
mean = torch.tensor(cfg.data.img_mean).view(1, 3, 1, 1)
std  = torch.tensor(cfg.data.img_std).view(1, 3, 1, 1)
clip_viz = (clip_tensor * std + mean).clamp(0, 1)
print(f"Augmented tensor: shape={tuple(clip_tensor.shape)}  "
      f"min={clip_tensor.min():.2f}  max={clip_tensor.max():.2f}")

# ── Row 3: simulate temporal resample inside encoder ──────────────────────
# encoder does: resample T_raw → 32 frames by per-sample linspace gather
T_raw = clip_tensor.shape[0]
target_T = cfg.encoder.videomae_num_frames   # 32
idx_resample = torch.linspace(0, T_raw - 1, target_T).round().long()
clip_resampled = clip_viz[idx_resample]   # [32, 3, H, W]

# ── Pick frames to display ────────────────────────────────────────────────
def pick_n(seq_len, n):
    return np.linspace(0, seq_len - 1, n).round().astype(int)

n_show = 8

# Row 1: raw
raw_idx = pick_n(len(raw_frames), n_show)
# Row 2: augmented un-normalized
aug_idx = pick_n(T_raw, n_show)
# Row 3: resampled (already 32)
res_idx = pick_n(target_T, n_show)
# Row 4: scrubbing strip - just the raw
strip_idx = pick_n(len(raw_frames), 16)

fig, axes = plt.subplots(4, max(n_show, 16), figsize=(24, 10))
fig.suptitle(f"label='{label}'   raw_frames={len(raw_frames)}   "
             f"after_aug={T_raw}   resampled_to={target_T}",
             fontsize=14)

# Row 1: raw
for j in range(max(n_show, 16)):
    ax = axes[0, j]
    ax.axis("off")
    if j < n_show:
        ax.imshow(raw_frames[raw_idx[j]])
        ax.set_title(f"raw[{raw_idx[j]}]", fontsize=8)
axes[0, 0].set_ylabel("RAW", fontsize=12, rotation=0, labelpad=40)

# Row 2: augmented
for j in range(max(n_show, 16)):
    ax = axes[1, j]
    ax.axis("off")
    if j < n_show:
        img = clip_viz[aug_idx[j]].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"aug[{aug_idx[j]}]", fontsize=8)
axes[1, 0].set_ylabel("AUG", fontsize=12, rotation=0, labelpad=40)

# Row 3: resampled (what VideoMAE sees)
for j in range(max(n_show, 16)):
    ax = axes[2, j]
    ax.axis("off")
    if j < n_show:
        img = clip_resampled[res_idx[j]].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"vmae[{res_idx[j]}]", fontsize=8)
axes[2, 0].set_ylabel("VMAE\nIN", fontsize=12, rotation=0, labelpad=40)

# Row 4: strip of raw (full timeline)
for j in range(max(n_show, 16)):
    ax = axes[3, j]
    ax.axis("off")
    if j < 16 and j < len(strip_idx):
        ax.imshow(raw_frames[strip_idx[j]])
        ax.set_title(f"t={strip_idx[j]}", fontsize=8)
axes[3, 0].set_ylabel("TIMELINE", fontsize=10, rotation=0, labelpad=40)

plt.tight_layout()
plt.savefig("/kaggle/working/pipeline_check.png", dpi=80, bbox_inches="tight")
plt.show()
print("Saved → /kaggle/working/pipeline_check.png")

# ── Sanity counters ──────────────────────────────────────────────────────
print("\n─── Sanity counters ───")
n_zero_frames = (clip_tensor.reshape(T_raw, -1).abs().sum(dim=1) == 0).sum().item()
print(f"Zero-d frames (from DropFrames): {n_zero_frames}/{T_raw}")
print(f"Per-channel mean of un-normalized clip: "
      f"R={clip_viz[:, 0].mean():.3f}  "
      f"G={clip_viz[:, 1].mean():.3f}  "
      f"B={clip_viz[:, 2].mean():.3f}")

# Diff between consecutive raw frames — proxy for motion
raw_np = np.stack([np.array(f) for f in raw_frames]).astype(np.float32)
diffs = np.abs(raw_np[1:] - raw_np[:-1]).mean(axis=(1, 2, 3))
print(f"Inter-frame motion (raw): mean={diffs.mean():.2f}  "
      f"min={diffs.min():.2f}  max={diffs.max():.2f}")
print(f"Frames with ~zero motion (diff<1.0): "
      f"{(diffs < 1.0).sum()}/{len(diffs)}")

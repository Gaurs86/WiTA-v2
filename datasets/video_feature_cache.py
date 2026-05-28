"""
datasets/video_feature_cache.py — X-CLIP segment-feature extraction + cache.

Differs from datasets/feature_cache.py (SigLIP per-frame):
  - Per-CLIP encoding, not per-frame
  - Sliding 8-frame window with stride → sequence of segment features
  - Output: [T_seg_i, D] per clip  (D=projection_dim, default 512)

Workflow
--------
1) `extract_xclip_features(samples, encoder, out_path)`
       For each clip: decode PNG → normalize (X-CLIP stats) → slide 8-frame
       window through encoder → save [T_seg, D] feature tensor.

2) The resulting cache plugs into the same `CachedFeatureDataset` and
   `make_cached_dataloaders` exported from feature_cache.py — no separate
   training-path code needed.

Disk footprint
--------------
3477 clips × ~15 segments × 512 dim × 2 bytes (fp16) ≈ 53 MB.
~5× smaller than the SigLIP per-frame cache.
"""

from __future__ import annotations

import io
import os
import time
import logging
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from ..models.encoders.xclip_encoder import XCLIPVideoEncoder, default_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame preprocessing for X-CLIP
# ---------------------------------------------------------------------------

def _resize_pil(img: Image.Image, image_size: int) -> Image.Image:
    """Ensure PIL image is RGB and (image_size, image_size)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return img


def _pil_to_xclip_tensor(
    img:        Image.Image,
    image_size: int,
) -> torch.Tensor:
    """
    PIL → [3, H, W] float32 normalized with X-CLIP stats.
    Resizes to (image_size, image_size) if needed.
    """
    img = _resize_pil(img, image_size)
    t = TF.to_tensor(img)                # [3, H, W] in [0, 1]
    return default_normalize(t)


def _pil_to_xclip_tensor_no_resize(img: Image.Image) -> torch.Tensor:
    """
    PIL → tensor + X-CLIP normalize, WITHOUT resize.  For use after a
    HandCropper which already resized the cropped frames to target_size.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    t = TF.to_tensor(img)
    return default_normalize(t)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_xclip_features(
    samples:      list[tuple[list[bytes], str]],
    encoder:      XCLIPVideoEncoder,
    out_path:     str,
    seg_chunk:    int = 4,
    device:       str | torch.device = "cuda",
    dtype:        torch.dtype = torch.float16,
    hand_cropper = None,
) -> dict:
    """
    Extract X-CLIP video features for every clip and save to disk.

    Parameters
    ----------
    samples   : output of stream_and_index() — list of (frame_bytes, label).
    encoder   : a frozen XCLIPVideoEncoder.
    out_path  : where to save the .pt cache.
    seg_chunk : how many segments to push through X-CLIP at once (VRAM knob).
                Each segment is num_frames frames at image_size×image_size.
                On T4 with xclip-base, seg_chunk=4 fits comfortably.
    device    : 'cuda' or 'cpu'.
    dtype     : fp16 cache to halve disk footprint.

    Returns
    -------
    Cache dict (also written to out_path) with the SAME layout as the SigLIP
    cache so CachedFeatureDataset / make_cached_dataloaders just work:
      feats:      list of [T_seg_i, D] CPU tensors
      labels:     list of str
      lengths:    LongTensor [N]  (T_seg_i per clip)
      model_name: str
      out_dim:    int (D = projection_dim)
      image_size: int
      num_frames: int (X-CLIP segment size)
      stride:     int (X-CLIP segment stride)
    """
    encoder = encoder.to(device).eval()
    image_size = encoder.image_size
    D          = encoder.out_dim

    # Ensure cropper output size matches what the encoder expects.
    if hand_cropper is not None:
        if hand_cropper.target_size != image_size:
            logger.warning(
                "hand_cropper.target_size=%d != encoder.image_size=%d; "
                "overriding cropper target_size to match.",
                hand_cropper.target_size, image_size,
            )
            hand_cropper.target_size = image_size

    feats_list:  list[torch.Tensor] = []
    labels_list: list[str]          = []
    lengths:     list[int]          = []

    total_clips = len(samples)
    log_every   = max(1, total_clips // 100)
    skipped     = 0
    t0          = time.time()
    print(
        f"[video_feature_cache] Encoding {total_clips} clips with "
        f"{encoder.model_name} on {device} (image_size={image_size}, "
        f"num_frames={encoder.num_frames}, stride={encoder.stride}, "
        f"hand_crop={hand_cropper is not None}, dtype={dtype})",
        flush=True,
    )

    for ci, (frame_bytes, label) in enumerate(samples):
        try:
            pil_frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in frame_bytes]
        except Exception as e:
            logger.warning("Skipping clip %d (decode error): %s", ci, e)
            skipped += 1
            continue

        if not pil_frames:
            skipped += 1
            continue

        # Optional hand cropping — same code path used at deployment.
        # crop_clip returns frames already resized to target_size.
        if hand_cropper is not None:
            pil_frames = hand_cropper.crop_clip(pil_frames)
            frames = [_pil_to_xclip_tensor_no_resize(f) for f in pil_frames]
        else:
            frames = [_pil_to_xclip_tensor(f, image_size) for f in pil_frames]

        clip = torch.stack(frames, dim=0).to(device, non_blocking=True)  # [T_raw, 3, H, W]
        seg_feats = encoder.encode_clip(
            clip, valid_len=clip.shape[0], chunk=seg_chunk,
        )                                                                # [T_seg, D]

        seg_feats_cpu = seg_feats.to(dtype).cpu().contiguous()
        feats_list.append(seg_feats_cpu)
        labels_list.append(label)
        lengths.append(seg_feats_cpu.shape[0])

        if (ci + 1) % log_every == 0 or (ci + 1) == total_clips:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total_clips - (ci + 1)) / max(rate, 1e-3)
            extra   = ""
            if hand_cropper is not None:
                s = hand_cropper.stats()
                extra = f"  hand_det={s['frame_detect_rate']*100:.1f}%"
            print(
                f"[video_feature_cache] {ci + 1}/{total_clips} clips  "
                f"({100*(ci+1)/total_clips:5.1f}%)  "
                f"{rate:.2f} clips/s  ETA {eta/60:5.1f} min  "
                f"skipped={skipped}{extra}",
                flush=True,
            )

    cache = {
        "feats":      feats_list,
        "labels":     labels_list,
        "lengths":    torch.tensor(lengths, dtype=torch.long),
        "model_name": encoder.model_name,
        "out_dim":    D,
        "image_size": image_size,
        "num_frames": encoder.num_frames,
        "stride":     encoder.stride,
        "hand_crop":  hand_cropper is not None,
    }
    if hand_cropper is not None:
        cache["hand_crop_stats"] = hand_cropper.stats()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    total_mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    logger.info(
        "[video_feature_cache] Saved %d clips → %s  (%.1f MB feats, %s)",
        len(feats_list), out_path, total_mb, dtype,
    )
    print(
        f"[video_feature_cache] Saved {len(feats_list)} clips → {out_path}  "
        f"({total_mb:.1f} MB, {dtype})",
        flush=True,
    )
    return cache

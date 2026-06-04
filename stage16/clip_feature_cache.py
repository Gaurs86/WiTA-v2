"""
stage16/clip_feature_cache.py — frozen CLIP ViT-B/16 per-frame feature cache.

Runs every clip's frames through a FROZEN CLIP image encoder ONCE and caches
the per-frame feature sequence [T, D] (D=512, CLIP's projected image embedding)
to disk as float16 .npy.  The training loop then reads cached features and
never runs the ViT -> fits a T4 easily.

Input frames: full-frame resized to 224x224 (no crop) -- matches Stage 13B/14
input.  Optional spatial augmentation (ColorJitter / RandomResizedCrop) can be
applied BEFORE extraction to build an augmented feature-cache variant (spatial
aug CANNOT be applied to already-cached features).

Cache layout: <cache_root>/<split>/<subset>/<SIGNER>__<clip_id>.npy  [T, 512] fp16
Resumable (skips existing .npy).

Usage:
  python -m stage16.clip_feature_cache \
      --data_root /kaggle/input/.../wita-full-english-122signers \
      --cache_root /kaggle/working/clip_feats --split train --subset lex \
      --t 32 --batch 64
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow running as a script: ensure repo root on path so `stage16` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import walk_clips, clip_cache_name, find_data_root   # noqa: E402

CLIP_MODEL = "openai/clip-vit-base-patch16"
CLIP_DIM = 512                      # get_image_features() projected dim
IMAGENET_OK = True                  # CLIP processor handles its own normalization


def _load_clip(device):
    from transformers import CLIPModel, CLIPImageProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    return model, proc


def _sample_frame_paths(clip_dir, t):
    files = sorted(f for f in os.listdir(clip_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        return []
    n = len(files)
    idx = np.linspace(0, n - 1, t).round().astype(int)
    idx = np.clip(idx, 0, n - 1)
    return [os.path.join(clip_dir, files[int(i)]) for i in idx]


@torch.no_grad()
def extract_split(data_root, cache_root, split, subset, t=32, batch=64,
                  spatial_aug=None, overwrite=False, device=None):
    """Cache CLIP features for one split+subset.  spatial_aug: optional callable
    PIL->PIL applied per frame BEFORE CLIP (for an augmented cache variant)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, proc = _load_clip(device)
    out_dir = Path(cache_root) / split / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = walk_clips(data_root, split, subsets=(subset,))
    print(f"[clip_cache] {split}/{subset}: {len(clips)} clips -> {out_dir}  "
          f"T={t} D={CLIP_DIM}", flush=True)
    use_amp = device == "cuda"
    t0 = time.time()
    n_written = n_existed = n_skipped = 0
    for ci, e in enumerate(clips):
        out_npy = out_dir / f"{clip_cache_name(e['signer'], e['clip_id'])}.npy"
        if out_npy.exists() and not overwrite:
            n_existed += 1
            continue
        try:
            paths = _sample_frame_paths(e["dir"], t)
            if not paths:
                n_skipped += 1; continue
            imgs = []
            for p in paths:
                im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
                if spatial_aug is not None:
                    im = spatial_aug(im)
                imgs.append(im)
            feats = []
            for s in range(0, len(imgs), batch):
                chunk = imgs[s:s + batch]
                px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Build the CLIP image embedding explicitly (vision tower ->
                    # visual projection).  This is exactly what get_image_features
                    # does, but returns a plain tensor across transformers versions
                    # (some return a BaseModelOutputWithPooling object).
                    vout = model.vision_model(pixel_values=px)
                    emb = model.visual_projection(vout.pooler_output)  # [n, 512]
                feats.append(emb.float().cpu())
            arr = torch.cat(feats, 0).numpy().astype(np.float16)       # [T, 512]
            tmp = str(out_npy) + ".tmp.npy"
            np.save(tmp, arr); os.replace(tmp, out_npy)
            n_written += 1
        except Exception as ex:
            n_skipped += 1
            print(f"  skip {e['dir']}: {type(ex).__name__}: {ex}", flush=True)
            # Fail fast on a SYSTEMATIC bug: if the first few clips all error
            # with nothing written, it's a code bug, not bad data -- don't
            # silently skip all 10K clips.
            if n_written == 0 and n_existed == 0 and n_skipped >= 3:
                raise RuntimeError(
                    f"First {n_skipped} clips all failed with nothing written -- "
                    f"likely a systematic bug, not bad data. Last error: "
                    f"{type(ex).__name__}: {ex}") from ex
        if (ci + 1) % 200 == 0 or (ci + 1) == len(clips):
            el = time.time() - t0
            rate = (ci + 1) / max(el, 1e-3)
            eta = (len(clips) - ci - 1) / max(rate, 1e-3) / 60
            print(f"  [{ci+1}/{len(clips)}] {rate:.1f} clips/s  ETA {eta:5.1f} min  "
                  f"written={n_written} reused={n_existed} skipped={n_skipped}", flush=True)
    print(f"[clip_cache] done {split}/{subset}: written={n_written} "
          f"reused={n_existed} skipped={n_skipped}  {time.time()-t0:.0f}s", flush=True)
    return {"split": split, "subset": subset, "n": len(clips),
            "written": n_written, "existed": n_existed, "skipped": n_skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--subset", type=str, default="both", choices=["lex", "nonlex", "both"])
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    data_root = args.data_root or find_data_root()
    subsets = ["lex", "nonlex"] if args.subset == "both" else [args.subset]
    for sub in subsets:
        extract_split(data_root, args.cache_root, args.split, sub,
                      t=args.t, batch=args.batch, overwrite=args.overwrite)

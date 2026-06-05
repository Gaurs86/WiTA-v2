"""
stage17/native_gate.py — high-resolution RE-GATE (fair test of the T=32 NO-GO).

The Stage 11 landmark cache is resampled to 32 frames (~2-3 points/letter),
which may be too coarse to render readable words.  This re-runs MediaPipe on a
SMALL sample of clips at NATIVE frame resolution (no 32-resample) and renders
them, to decide whether temporal coarseness (fixable by re-extraction) — or a
fundamental limit of the static-image representation — is why the T=32 gate was
unreadable.  ~5 min for ~36 clips.  Needs the raw frames dataset + mediapipe.
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import walk_clips, find_data_root          # noqa: E402
from stage17.render import render_trajectory                    # noqa: E402

INDEX_FINGER_TIP = 8


def native_fingertip_path(clip_dir, extractor, max_frames=400):
    """Run MediaPipe over ALL native frames of a clip -> (path[T,2], detect_rate).
    Fill-forward for undetected frames (same policy as the cache builder)."""
    files = sorted(f for f in os.listdir(clip_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if max_frames and len(files) > max_frames:
        idx = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        files = [files[int(i)] for i in idx]
    path, last, n_det = [], None, 0
    for fn in files:
        try:
            im = Image.open(os.path.join(clip_dir, fn)).convert("RGB")
        except Exception:
            if last is not None:
                path.append(last)
            continue
        lm = extractor.detect(im)
        if lm is None:
            if last is not None:
                path.append(last)
        else:
            xy = lm[INDEX_FINGER_TIP, :2].astype(np.float32)
            path.append(xy); last = xy; n_det += 1
    if not path:
        return np.zeros((0, 2), np.float32), 0.0
    return np.stack(path, 0), n_det / max(len(files), 1)


def render_native_sample(out_path, raw_root=None, split="train", subset="lex",
                         n=36, cols=6, tile=224, flip_x=True, seed=0,
                         max_frames=400, min_len=3, max_len=8, **render_kw):
    """Render n native-resolution clips (word length in [min_len,max_len]) into a
    labelled contact sheet.  render_kw -> render_trajectory (e.g. smooth_k=5)."""
    raw_root = raw_root or find_data_root()
    from datasets.skeleton_cache import LandmarkExtractor

    clips = walk_clips(raw_root, split, subsets=(subset,))
    clips = [c for c in clips if min_len <= len(c["label"]) <= max_len]
    if not clips:
        print(f"[native_gate] no clips with len in [{min_len},{max_len}]"); return None
    random.seed(seed); random.shuffle(clips)
    clips = clips[:n]

    ext = LandmarkExtractor()
    rows = (len(clips) + cols - 1) // cols
    label_h = 24
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    print(f"[native_gate] rendering {len(clips)} native-res {split}/{subset} clips:", flush=True)
    for i, c in enumerate(clips):
        xy, det = native_fingertip_path(c["dir"], ext, max_frames=max_frames)
        img = (render_trajectory(xy, size=tile, flip_x=flip_x, **render_kw)
               if len(xy) >= 2 else Image.new("RGB", (tile, tile), (255, 255, 255)))
        r, cc = divmod(i, cols)
        x0, y0 = cc * tile, r * (tile + label_h)
        sheet.paste(img, (x0, y0))
        dr.text((x0 + 4, y0 + tile + 5),
                f"{c['label']}  T={len(xy)} det={det:.2f}", fill=(0, 0, 0))
        print(f"  {c['label']:>16}  T={len(xy):3d}  det={det:.2f}", flush=True)
    ext.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    print(f"[native_gate] wrote {len(clips)} native-res renders -> {out_path}", flush=True)
    return out_path

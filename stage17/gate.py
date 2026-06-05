"""
stage17/gate.py — the GO / NO-GO gate (run BEFORE any GPU).

1. detection_report : per-clip MediaPipe hand-detection rate distribution
   (Step-1 data-quality signal; the paper's main worry is tracking failure).
2. sample_sheet     : render ~30 train clips next to their GT words (like the
   paper's Fig 2 / Fig 7).  If a human can't read a fair fraction, the VLM
   won't either -> STOP and fix tracking/rendering before training.

Both read only the Stage 11 landmark cache (no GPU, no MediaPipe).
"""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image, ImageDraw

from stage17.common import iter_clips, load_clip_npz, fingertip_xy
from stage17.render import render_trajectory


def detection_report(cache_root, split="train", subsets=("lex", "nonlex"),
                     sample=None, seed=0):
    """Print + return the per-clip detection-rate distribution."""
    clips = iter_clips(cache_root, split, subsets)
    if not clips:
        print(f"[gate] no clips found under {cache_root}/{split}"); return None
    if sample:
        random.seed(seed)
        clips = random.sample(clips, min(sample, len(clips)))
    det = np.array([load_clip_npz(c["npz"])["detected"] for c in clips], dtype=float)
    det = det[np.isfinite(det)]
    if len(det) == 0:
        print("[gate] no detection values in cache"); return None
    print(f"[gate] detection rate over {len(det)} {split} clips: "
          f"mean={det.mean():.3f}  median={np.median(det):.3f}  "
          f"p10={np.percentile(det, 10):.3f}  p25={np.percentile(det, 25):.3f}  "
          f"| frac<0.5 = {(det < 0.5).mean() * 100:.1f}%  "
          f"frac<0.25 = {(det < 0.25).mean() * 100:.1f}%", flush=True)
    return det


def sample_sheet(cache_root, out_path, split="train", subsets=("lex", "nonlex"),
                 n=30, cols=6, tile=224, flip_x=True, seed=0,
                 worst_det=False, **render_kw):
    """Render n clips into a labelled contact sheet PNG.

    worst_det=True picks the LOWEST-detection clips (stress test); otherwise a
    random sample.  render_kw is forwarded to render_trajectory (e.g.
    encode_color=False, encode_width=False for ablation sheets)."""
    clips = iter_clips(cache_root, split, subsets)
    if not clips:
        print(f"[gate] no clips found under {cache_root}/{split}"); return None
    if worst_det:
        scored = [(load_clip_npz(c["npz"])["detected"], c) for c in clips]
        scored.sort(key=lambda x: x[0])
        clips = [c for _, c in scored[:n]]
    else:
        random.seed(seed)
        clips = random.sample(clips, min(n, len(clips)))

    rows = (len(clips) + cols - 1) // cols
    label_h = 24
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    for i, c in enumerate(clips):
        d = load_clip_npz(c["npz"])
        img = render_trajectory(fingertip_xy(d["feature"]), size=tile,
                                flip_x=flip_x, **render_kw)
        r, cc = divmod(i, cols)
        x0, y0 = cc * tile, r * (tile + label_h)
        sheet.paste(img, (x0, y0))
        dr.text((x0 + 4, y0 + tile + 5),
                f"{d['label']}  [{d['subset'][:2]}] det={d['detected']:.2f}",
                fill=(0, 0, 0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    print(f"[gate] wrote sample sheet ({len(clips)} clips, "
          f"{'worst-detection' if worst_det else 'random'}) -> {out_path}", flush=True)
    return out_path

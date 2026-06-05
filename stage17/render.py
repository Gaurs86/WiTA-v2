"""
stage17/render.py — turn a fingertip (x,y) trajectory into ONE RGB image that
makes letters separable (the crux of the approach).

Encodings (each is a deliberate cue for the VLM / a human reader):
  * horizontal FLIP            third-person camera mirrors the writing; flip so
                               text reads left-to-right (verify on a known word)
  * center + uniform SCALE     fit a fixed canvas with margin, aspect preserved
                               (never distort letter shapes)
  * uniform ARC-LENGTH resample   removes writing-speed bias from the geometry
  * HUE gradient along path    = temporal ORDER (start->end); the single most
                               important cue for disambiguating overlapping
                               strokes in one pen-lift-free scribble
  * line WIDTH = inverse speed slow strokes (letters) thick, fast low-curvature
                               transitions (between letters) thin -> visual
                               letter separation
  * anti-alias                 render at `supersample`x then LANCZOS downscale

One PNG per clip is cached; the VLM reads only the PNGs.
"""

from __future__ import annotations

import colorsys
import numpy as np
from PIL import Image, ImageDraw


def _smooth(xy: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(xy) < k:
        return xy
    ker = np.ones(k) / k
    out = xy.copy()
    for c in range(xy.shape[1]):
        out[:, c] = np.convolve(xy[:, c], ker, mode="same")
    out[0] = xy[0]; out[-1] = xy[-1]              # keep endpoints anchored
    return out


def _dedup(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    keep = [0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - xy[keep[-1]])) > eps:
            keep.append(i)
    return xy[keep] if len(keep) >= 2 else xy


def _arc_resample(xy: np.ndarray, n: int):
    """Resample [T,2] to n points uniformly by arc length.

    Returns (pts[n,2], tfrac[n], tau[n]) where tau is the ORIGINAL continuous
    frame index at each resampled point (monotonic) and tfrac = tau/(T-1).
    tau drives both the hue (temporal order) and the width (local time density
    = inverse pen speed)."""
    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])       # cumulative arc length
    total = float(s[-1])
    idx = np.arange(len(xy), dtype=np.float64)
    if total <= 0:                                    # degenerate (no motion)
        pts = np.repeat(xy[:1], n, axis=0).astype(np.float32)
        return pts, np.linspace(0, 1, n), np.zeros(n)
    u = np.linspace(0.0, total, n)
    tau = np.interp(u, s, idx)
    x = np.interp(u, s, xy[:, 0]); y = np.interp(u, s, xy[:, 1])
    pts = np.stack([x, y], 1).astype(np.float32)
    tfrac = tau / max(len(xy) - 1, 1)
    return pts, tfrac, tau


def render_trajectory(xy, size: int = 256, n_points: int = 512, supersample: int = 4,
                      flip_x: bool = True, margin: float = 0.12, smooth_k: int = 3,
                      w_min: float = 1.0, w_max: float = 7.0,
                      bg=(255, 255, 255), encode_width: bool = True,
                      encode_color: bool = True) -> Image.Image:
    """fingertip path [T,2] in [0,1] image coords -> PIL RGB (size x size)."""
    xy = np.asarray(xy, np.float32).copy()
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        return Image.new("RGB", (size, size), bg)
    if flip_x:
        xy[:, 0] = 1.0 - xy[:, 0]
    xy = _smooth(xy, smooth_k)
    xy = _dedup(xy)
    if len(xy) < 2:
        return Image.new("RGB", (size, size), bg)

    pts, tfrac, tau = _arc_resample(xy, n_points)

    # WIDTH from local original-time density dtau/dk (slow writing -> thick).
    dtau = np.clip(np.gradient(tau), 0, None)
    lo, hi = np.percentile(dtau, 10), np.percentile(dtau, 90)
    wn = np.clip((dtau - lo) / max(hi - lo, 1e-6), 0, 1)
    widths = (w_min + wn * (w_max - w_min)) if encode_width else np.full(n_points, w_max * 0.5)

    # NORMALIZE geometry: bbox center + uniform scale to canvas with margin.
    S = size * supersample
    mn, mx = pts.min(0), pts.max(0)
    span = float((mx - mn).max()) or 1.0
    scale = (1 - 2 * margin) * S / span
    center = (mn + mx) / 2.0
    cp = (pts - center) * scale + S / 2.0             # [n,2] canvas coords

    img = Image.new("RGB", (S, S), bg)
    dr = ImageDraw.Draw(img)
    for i in range(1, len(cp)):
        p0 = (float(cp[i - 1][0]), float(cp[i - 1][1]))
        p1 = (float(cp[i][0]),     float(cp[i][1]))
        if encode_color:
            r, g, b = colorsys.hsv_to_rgb(float(tfrac[i]) * 0.85, 0.9, 0.85)
            col = (int(r * 255), int(g * 255), int(b * 255))
        else:
            col = (20, 20, 20)
        w = max(int(round(widths[i] * supersample)), 1)
        dr.line([p0, p1], fill=col, width=w)
        rad = max(w // 2, 1)                           # round joint, no gaps
        dr.ellipse([p1[0] - rad, p1[1] - rad, p1[0] + rad, p1[1] + rad], fill=col)

    return img.resize((size, size), Image.LANCZOS)


def _selftest(path: str = "/tmp/stage17_render_selftest.png") -> str:
    """Render a synthetic varying-speed figure-eight; assert non-blank + saved.
    (Readability of REAL words is judged at the gate, not here.)"""
    t = np.linspace(0, 2 * np.pi, 40)
    # non-uniform time sampling -> exercises the speed->width encoding
    t = t + 0.25 * np.sin(t)
    xy = np.stack([0.5 + 0.3 * np.sin(t), 0.5 + 0.28 * np.sin(2 * t)], 1).astype(np.float32)
    img = render_trajectory(xy, size=256)
    arr = np.asarray(img)
    assert arr.shape == (256, 256, 3), arr.shape
    assert arr.std() > 5, "render looks blank"
    # start of path should be reddish, end bluish (temporal hue gradient present)
    img.save(path)
    print(f"[stage17.render] self-test OK -> {path}  mean={arr.mean():.1f} std={arr.std():.1f}")
    return path


if __name__ == "__main__":
    _selftest()

"""
scripts/audit_tracker_quality.py — Task A from the post-Stage-1-v3 prompt.

Quantifies whether the hard-regime CER tail (PHW, PJH, SYB, KJM, KNY, LKS,
KIM, YMG) is driven by MediaPipe tracker dropouts / drift, or by genuinely
harder writing.  Output gates the visibility-gate design choice in Task B
(Stage 3 fingertip pool).

Two run modes
-------------
--mode cache       (default, fast):
    Reads the existing skeleton_features_t32.pt cache.  Computes the
    per-clip quality metrics from the cache's 190-dim per-frame descriptor:
        position joint-8  ->  feats[:, 24:27]   (INDEX_FINGER_TIP)
        visibility column ->  feats[:, 189]
    Caveat: the cache is resampled to T_native=32 with fill-forward on
    dropouts.  Dropout-pattern metrics are derived from the visibility
    column thresholded at 0.5.  Adequate for the headline Pearson test;
    use --mode reextract for native-resolution audit + overlay GIFs.

--mode reextract:
    Re-streams the WiTA zips for the 16 audited signers (8 hard, 8 easy),
    re-runs MediaPipe HandLandmarker with the exact Stage-1 parameters,
    computes metrics on the NATIVE (unresampled, non-filled) per-frame
    sequence, and renders 5 overlay GIFs per signer.  ~30 min on CPU.

Deliverables (both modes)
-------------------------
reports/tracker_audit/per_signer_quality.csv
reports/tracker_audit/quality_vs_cer.png
manifests/audit_clips.json                  (sampled clip indices, seed=42)

Reextract mode adds
-------------------
reports/tracker_audit/clip_overlays/<signer>_<i>.gif

Pass/fail rule
--------------
If Pearson(visibility_rate, CER) <= -0.5 OR Pearson(mean_dropout_len, CER)
>= +0.5, tracker quality is the dominant residual.  Keep the visibility-
gate dim in Task B.  If |r| < 0.3 for all three, drop it (and inspect the
GIFs to confirm visually).

Usage
-----
python scripts/audit_tracker_quality.py \
    --stage1v3-results /path/to/stage1v3_results.json \
    --cache-path       /path/to/skeleton_features_t32.pt \
    --output-dir       reports/tracker_audit \
    --mode             cache

Or for the full audit with overlays (any environment with mediapipe + HF
access):
python scripts/audit_tracker_quality.py \
    --stage1v3-results /path/to/stage1v3_results.json \
    --output-dir       reports/tracker_audit \
    --mode             reextract
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np

logger = logging.getLogger("tracker_audit")


# ---------------------------------------------------------------------------
# Signer regimes (from Stage 1 v3 §1.3)
# ---------------------------------------------------------------------------

HARD_SIGNERS = ['PHW', 'PJH', 'SYB', 'KJM', 'KNY', 'LKS', 'KIM', 'YMG']
EASY_SIGNERS = ['KIS', 'YJH', 'KHY', 'HYW', 'KSH', 'JSA', 'YSY', 'KSJ']


# ---------------------------------------------------------------------------
# Per-signer CER from Stage 1 v3 results
# ---------------------------------------------------------------------------

def load_per_signer_cer(stage1v3_results_path: str) -> dict[str, float]:
    """
    Build {signer_id: best_val_cer} from the Stage 1 v3 no_dann variant
    across all 5 folds.  Each signer appears in exactly one val fold, so
    the union of per-fold best_per_signer_val_cer dicts is a 39-entry map.
    """
    with open(stage1v3_results_path) as f:
        results = json.load(f)
    per_signer: dict[str, float] = {}
    for r in results:
        if r.get('variant') != 'no_dann':
            continue
        per_signer.update(r.get('best_per_signer_val_cer', {}) or {})
    if not per_signer:
        raise RuntimeError(
            f"No per-signer CER entries found for no_dann in {stage1v3_results_path}. "
            f"Make sure best_per_signer_val_cer is populated in dann_train.py results."
        )
    return per_signer


# ---------------------------------------------------------------------------
# Quality-metric core (works on per-frame arrays in BOTH modes)
# ---------------------------------------------------------------------------

def per_frame_quality_metrics(
    tip_x:    np.ndarray,        # [T]
    tip_y:    np.ndarray,        # [T]
    detected: np.ndarray,        # [T] bool / {0,1}
    *,
    acc_thr:  float = 0.05,
) -> dict:
    """
    Compute the 7 per-clip quality metrics from per-frame fingertip
    coordinates + detection flags.  See the algorithm in the post-Stage-1-v3
    prompt §2 Task A.

    All inputs are in NORMALISED image coordinates ([0,1]).  Outputs are
    floats or ints (no torch tensors).
    """
    detected = detected.astype(bool)
    n = len(detected)
    out = dict(
        visibility_rate     = float(detected.mean()) if n else float('nan'),
        tip_jitter          = float('nan'),
        acc_spike_count     = 0,
        n_dropouts          = 0,
        mean_dropout_len    = 0.0,
        trajectory_length   = 0.0,
    )
    if n < 2:
        return out

    # First differences (only count consecutive-valid frame pairs).
    dx = np.diff(tip_x)
    dy = np.diff(tip_y)
    valid_pair = detected[1:] & detected[:-1]
    if valid_pair.any():
        dists = np.sqrt(dx[valid_pair] ** 2 + dy[valid_pair] ** 2)
        out['tip_jitter']        = float(dists.mean())
        out['trajectory_length'] = float(dists.sum())

    # Second differences (consecutive-valid triples).
    if n >= 3:
        ddx = np.diff(tip_x, n=2)
        ddy = np.diff(tip_y, n=2)
        valid_trip = detected[2:] & detected[1:-1] & detected[:-2]
        if valid_trip.any():
            spike = (np.abs(ddx[valid_trip]) + np.abs(ddy[valid_trip])) > acc_thr
            out['acc_spike_count'] = int(spike.sum())

    # Rising-to-dropout transitions and dropout-run lengths.
    drop_edges = (detected[:-1] & ~detected[1:])
    out['n_dropouts'] = int(drop_edges.sum())
    runs, cur = [], 0
    for d in detected:
        if not d:
            cur += 1
        elif cur > 0:
            runs.append(cur); cur = 0
    if cur > 0:
        runs.append(cur)
    out['mean_dropout_len'] = float(np.mean(runs)) if runs else 0.0
    return out


# ---------------------------------------------------------------------------
# Mode 1: extract metrics from the existing cache
# ---------------------------------------------------------------------------

def audit_from_cache(
    cache_path:       str,
    n_per_signer:     int,
    seed:             int,
    audit_manifest:   str,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Returns
    -------
    signer_metrics : {signer_id: {metric: mean_over_clips}}
    sampled_clips  : list of {signer, clip_idx, label} dicts (also written
                     to the audit manifest)
    """
    import torch
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)

    feats:    list = cache['feats']             # list[Tensor [T, 190]]
    subjects: list = cache['subjects']

    # Group clip indices by signer.
    by_signer: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(subjects):
        by_signer[s].append(i)

    sampled_clips: list[dict] = []
    rng = random.Random(seed)
    audited = sorted(set(HARD_SIGNERS + EASY_SIGNERS))
    for s in audited:
        if s not in by_signer:
            logger.warning("Signer %s not present in cache — skipping.", s)
            continue
        idxs = sorted(by_signer[s])
        k = min(n_per_signer, len(idxs))
        chosen = rng.sample(idxs, k)
        for ci in sorted(chosen):
            sampled_clips.append({
                'signer':   s,
                'clip_idx': int(ci),
                'label':    cache['labels'][ci],
                'regime':   'hard' if s in HARD_SIGNERS else 'easy',
            })

    # Write the audit manifest so reextract mode (or a re-run) can reproduce.
    os.makedirs(os.path.dirname(audit_manifest) or ".", exist_ok=True)
    with open(audit_manifest, 'w') as f:
        json.dump({
            'seed':         seed,
            'n_per_signer': n_per_signer,
            'hard':         HARD_SIGNERS,
            'easy':         EASY_SIGNERS,
            'clips':        sampled_clips,
        }, f, indent=2)
    logger.info("Wrote audit manifest -> %s  (%d clips)",
                audit_manifest, len(sampled_clips))

    # Joint 8 = INDEX_FINGER_TIP; (x,y,z) live at feature dims [24:27].
    # Visibility column is the last one (index 189).
    per_clip: dict[int, dict] = {}
    for entry in sampled_clips:
        ci = entry['clip_idx']
        f  = feats[ci].float().numpy()        # [T, 190]
        tx = f[:, 24]
        ty = f[:, 25]
        vis = f[:, 189]
        # In the cache, visibility is post-resample so values are in [0, 1].
        # Threshold at 0.5 to recover a binary detection signal.
        detected = (vis >= 0.5).astype(np.float32)
        per_clip[ci] = per_frame_quality_metrics(tx, ty, detected)

    # Aggregate per-signer means.
    signer_metrics: dict[str, dict] = {}
    for s in audited:
        clips = [e for e in sampled_clips if e['signer'] == s]
        if not clips:
            continue
        rows = [per_clip[c['clip_idx']] for c in clips]
        signer_metrics[s] = {
            m: float(np.mean([r[m] for r in rows])) for m in rows[0].keys()
        }
        signer_metrics[s]['n_clips'] = len(clips)
    return signer_metrics, sampled_clips


# ---------------------------------------------------------------------------
# Mode 2: re-extract from raw frames with MediaPipe (native resolution +
# overlay GIFs)
# ---------------------------------------------------------------------------

def audit_from_reextract(
    output_dir:       str,
    n_per_signer:     int,
    seed:             int,
    audit_manifest:   str,
    max_overlays:     int,
    hf_repo_id:       str,
    hf_lang:          str,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Streams the WiTA zips for the audited signers only, runs MediaPipe per
    frame natively, computes metrics, and renders overlay GIFs.

    Requires mediapipe, PIL, imageio, huggingface_hub on the system.
    """
    # Lazy imports so cache-mode users don't need these.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from PIL import Image
    import imageio
    from wita_v2.datasets.skeleton_cache import LandmarkExtractor
    from wita_v2.configs.default import Config, DataConfig, EncoderConfig, TrainConfig
    from wita_v2.datasets.subject_splits import stream_and_index_with_subjects

    audited = sorted(set(HARD_SIGNERS + EASY_SIGNERS))

    # Build a minimal cfg purely to drive the streamer.
    cfg = Config(
        data=DataConfig(
            hf_repo_id  = hf_repo_id,
            lang        = hf_lang,
            max_zips    = None,
            max_frames  = 64,
            train_split = 0.90,
            seed        = seed,
        ),
        encoder=EncoderConfig(arch='siglip'),
        train=TrainConfig(num_epochs=1, batch_size=1, seed=seed,
                          checkpoint_dir='/tmp/_unused'),
    ).build()

    logger.info("Streaming WiTA zips (filter to %d audited signers)...",
                len(audited))
    samples = stream_and_index_with_subjects(cfg)        # list[(frames, label, subj)]
    samples = [s for s in samples if s[2] in audited]
    logger.info("Got %d clips across %d audited signers.",
                len(samples),
                len({s[2] for s in samples}))

    by_signer: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_signer[s[2]].append(i)

    sampled_clips: list[dict] = []
    rng = random.Random(seed)
    for s in audited:
        if s not in by_signer:
            logger.warning("Signer %s not present in stream — skipping.", s)
            continue
        idxs = sorted(by_signer[s])
        k = min(n_per_signer, len(idxs))
        chosen = sorted(rng.sample(idxs, k))
        for ci in chosen:
            sampled_clips.append({
                'signer':       s,
                'sample_idx':   int(ci),
                'label':        samples[ci][1],
                'regime':       'hard' if s in HARD_SIGNERS else 'easy',
                'n_raw_frames': len(samples[ci][0]),
            })

    with open(audit_manifest, 'w') as f:
        json.dump({
            'seed':         seed,
            'n_per_signer': n_per_signer,
            'hard':         HARD_SIGNERS,
            'easy':         EASY_SIGNERS,
            'clips':        sampled_clips,
        }, f, indent=2)
    logger.info("Wrote audit manifest -> %s", audit_manifest)

    overlays_dir = os.path.join(output_dir, 'clip_overlays')
    os.makedirs(overlays_dir, exist_ok=True)

    extractor = LandmarkExtractor()
    per_clip: dict[int, dict] = {}
    n_overlay_per_signer: dict[str, int] = defaultdict(int)

    for c_i, entry in enumerate(sampled_clips):
        ci         = entry['sample_idx']
        signer     = entry['signer']
        frame_list = samples[ci][0]
        T          = len(frame_list)

        # Per-frame raw extraction (no fill-forward).
        tip_x    = np.zeros(T, dtype=np.float32)
        tip_y    = np.zeros(T, dtype=np.float32)
        detected = np.zeros(T, dtype=np.float32)
        per_frame_lm: list = []                # None or [21,3] for overlay
        for t, b in enumerate(frame_list):
            try:
                img = Image.open(io.BytesIO(b))
            except Exception:
                per_frame_lm.append(None); continue
            lm = extractor.detect(img)
            if lm is None:
                per_frame_lm.append(None); continue
            detected[t] = 1.0
            tip_x[t]    = lm[8, 0]
            tip_y[t]    = lm[8, 1]
            per_frame_lm.append(lm)
        per_clip[ci] = per_frame_quality_metrics(tip_x, tip_y, detected)

        # Render overlay GIF for the first `max_overlays` clips per signer.
        if n_overlay_per_signer[signer] < max_overlays:
            gif_path = os.path.join(
                overlays_dir,
                f"{signer}_{n_overlay_per_signer[signer]:02d}.gif",
            )
            _render_overlay_gif(frame_list, per_frame_lm, gif_path, fps=5)
            n_overlay_per_signer[signer] += 1
            logger.info("Wrote overlay -> %s", gif_path)

        if (c_i + 1) % 10 == 0 or (c_i + 1) == len(sampled_clips):
            logger.info("Audited %d/%d clips...", c_i + 1, len(sampled_clips))
    extractor.close()

    # Aggregate per-signer.
    signer_metrics: dict[str, dict] = {}
    for s in audited:
        clips = [e for e in sampled_clips if e['signer'] == s]
        if not clips:
            continue
        rows = [per_clip[c['sample_idx']] for c in clips]
        signer_metrics[s] = {
            m: float(np.mean([r[m] for r in rows])) for m in rows[0].keys()
        }
        signer_metrics[s]['n_clips'] = len(clips)
    return signer_metrics, sampled_clips


def _render_overlay_gif(
    frame_bytes_list: list[bytes],
    per_frame_lm:     list,
    out_path:         str,
    fps:              int = 5,
) -> None:
    """Red dot at fingertip, red border on dropout frames, full hand skeleton."""
    from PIL import Image
    import imageio

    frames = []
    last_tip = None
    trail: list[tuple[int, int]] = []
    for b, lm in zip(frame_bytes_list, per_frame_lm):
        try:
            img = np.array(Image.open(io.BytesIO(b)).convert('RGB'))
        except Exception:
            continue
        H, W = img.shape[:2]
        out = img.copy()
        if lm is None:
            # 8-px red border = dropout
            out[:8, :, :] = [255, 0, 0]
            out[-8:, :, :] = [255, 0, 0]
            out[:, :8, :] = [255, 0, 0]
            out[:, -8:, :] = [255, 0, 0]
            last_tip = None
        else:
            tx = int(np.clip(lm[8, 0] * W, 0, W - 1))
            ty = int(np.clip(lm[8, 1] * H, 0, H - 1))
            trail.append((tx, ty))
            # Trail: faded red line
            for (px, py) in trail[-40:]:
                out[max(py-1, 0):py+2, max(px-1, 0):px+2] = [255, 80, 80]
            # Current tip: bright red square
            out[max(ty-4, 0):ty+5, max(tx-4, 0):tx+5] = [255, 0, 0]
            last_tip = (tx, ty)
        frames.append(out)
    if frames:
        imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)


# ---------------------------------------------------------------------------
# Aggregation, joining, plotting
# ---------------------------------------------------------------------------

def build_csv_and_scatter(
    signer_metrics: dict[str, dict],
    per_signer_cer: dict[str, float],
    output_dir:     str,
) -> dict:
    """
    Joins metrics + CER per signer, writes per_signer_quality.csv and
    quality_vs_cer.png.  Returns a verdict dict with Pearson correlations.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    os.makedirs(output_dir, exist_ok=True)
    signers = sorted(signer_metrics.keys())
    cer = {s: per_signer_cer.get(s, float('nan')) for s in signers}

    # Per-signer CSV.
    csv_path = os.path.join(output_dir, 'per_signer_quality.csv')
    metric_keys = ['n_clips', 'visibility_rate', 'tip_jitter',
                   'acc_spike_count', 'n_dropouts', 'mean_dropout_len',
                   'trajectory_length']
    with open(csv_path, 'w') as f:
        f.write('signer,regime,val_cer,' + ','.join(metric_keys) + '\n')
        for s in signers:
            row = signer_metrics[s]
            regime = 'hard' if s in HARD_SIGNERS else 'easy'
            f.write(
                f'{s},{regime},{cer[s]:.4f},'
                + ','.join(f'{row[k]:.6f}' for k in metric_keys)
                + '\n'
            )
    logger.info("Wrote %s", csv_path)

    # 3-panel scatter: visibility_rate, mean_dropout_len, acc_spike_count vs CER.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_plots = [
        ('visibility_rate',  -1, 'visibility_rate (frames where MP detected)'),
        ('mean_dropout_len', +1, 'mean_dropout_len (frames per dropout run)'),
        ('acc_spike_count',  +1, 'acc_spike_count (|d2x|+|d2y| > 0.05)'),
    ]
    verdict = {}
    for ax, (m, expected_sign, xlabel) in zip(axes, metric_plots):
        xs = np.array([signer_metrics[s][m] for s in signers], dtype=float)
        ys = np.array([cer[s] for s in signers], dtype=float)
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        xs_v, ys_v = xs[valid], ys[valid]
        colors = ['#d62728' if s in HARD_SIGNERS else '#1f77b4' for s in signers]
        ax.scatter(xs, ys, c=colors, s=60, edgecolors='black', linewidths=0.5)
        for s, x, y in zip(signers, xs, ys):
            ax.annotate(s, (x, y), fontsize=7, alpha=0.6,
                        xytext=(3, 3), textcoords='offset points')
        if valid.sum() >= 3:
            r, p = pearsonr(xs_v, ys_v)
        else:
            r, p = float('nan'), float('nan')
        verdict[m] = {'r': float(r), 'p': float(p), 'n': int(valid.sum())}
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Stage 1 v3 no_dann val CER')
        ax.set_title(f'{m}\nPearson r={r:+.3f}, p={p:.3f}')
        ax.grid(True, linestyle=':', alpha=0.3)
    # Legend dummy points
    axes[0].scatter([], [], c='#d62728', label='hard regime')
    axes[0].scatter([], [], c='#1f77b4', label='easy regime')
    axes[0].legend(loc='best', frameon=False)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'quality_vs_cer.png')
    plt.savefig(fig_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)

    # Pass/fail rule.
    vis_r = verdict['visibility_rate']['r']
    drop_r = verdict['mean_dropout_len']['r']
    tracker_dominant = (
        (not np.isnan(vis_r)  and vis_r  <= -0.5) or
        (not np.isnan(drop_r) and drop_r >= +0.5)
    )
    tracker_null = (
        (np.isnan(vis_r)  or abs(vis_r)  < 0.3) and
        (np.isnan(drop_r) or abs(drop_r) < 0.3)
    )
    verdict['decision'] = (
        'tracker_dominant' if tracker_dominant else
        'tracker_null'     if tracker_null else
        'tracker_partial'
    )
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage1v3-results', required=True,
        help='Path to stage1v3_results.json (per-signer CER source).')
    p.add_argument('--mode', choices=['cache', 'reextract'], default='cache')
    p.add_argument('--cache-path',
        help='Path to skeleton_features_t32.pt (required for --mode cache).')
    p.add_argument('--output-dir', default='reports/tracker_audit')
    p.add_argument('--audit-manifest', default='manifests/audit_clips.json')
    p.add_argument('--n-per-signer', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-overlays', type=int, default=5,
        help='GIFs per signer (reextract mode only).')
    p.add_argument('--hf-repo-id',   default='yewon816/WiTA')
    p.add_argument('--hf-lang',      default='english')
    p.add_argument('--log-level',    default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )

    os.makedirs(args.output_dir, exist_ok=True)
    per_signer_cer = load_per_signer_cer(args.stage1v3_results)
    logger.info("Loaded per-signer CER for %d signers.", len(per_signer_cer))

    if args.mode == 'cache':
        if not args.cache_path:
            p.error("--cache-path is required when --mode cache")
        signer_metrics, sampled = audit_from_cache(
            cache_path     = args.cache_path,
            n_per_signer   = args.n_per_signer,
            seed           = args.seed,
            audit_manifest = args.audit_manifest,
        )
    else:
        signer_metrics, sampled = audit_from_reextract(
            output_dir     = args.output_dir,
            n_per_signer   = args.n_per_signer,
            seed           = args.seed,
            audit_manifest = args.audit_manifest,
            max_overlays   = args.max_overlays,
            hf_repo_id     = args.hf_repo_id,
            hf_lang        = args.hf_lang,
        )

    verdict = build_csv_and_scatter(
        signer_metrics  = signer_metrics,
        per_signer_cer  = per_signer_cer,
        output_dir      = args.output_dir,
    )
    print('\n=== Tracker quality audit verdict ===')
    print(f'  mode                 : {args.mode}')
    print(f'  signers audited      : {len(signer_metrics)}')
    for m, v in verdict.items():
        if m == 'decision': continue
        print(f'  {m:<20s}: r={v["r"]:+.3f}  p={v["p"]:.4f}  (n={v["n"]})')
    print(f'  decision             : {verdict["decision"]}')
    if verdict['decision'] == 'tracker_dominant':
        print('\n  -> Keep visibility-gate dim in Task B.')
    elif verdict['decision'] == 'tracker_null':
        print('\n  -> Drop visibility-gate dim in Task B; inspect overlay GIFs to confirm.')
    else:
        print('\n  -> Partial signal. Inspect GIFs and decide manually.')

    summary_path = os.path.join(args.output_dir, 'audit_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'mode':           args.mode,
            'verdict':        verdict,
            'signer_metrics': signer_metrics,
            'per_signer_cer': per_signer_cer,
        }, f, indent=2, sort_keys=True)
    print(f'\n  full summary -> {summary_path}')


if __name__ == '__main__':
    main()

"""
reports/template/per_signer_scatter.py — reusable per-signer scatter plot.

Standardised across all stage reports per Task D.  Hard-regime and easy-regime
thresholds are drawn as horizontal lines so visual comparison across stages
is one-to-one.

CLI:
    python -m wita_v2.reports.template.per_signer_scatter \
        --results /path/to/stage_results.json \
        --variant no_dann \
        --out     reports/stage1v3/per_signer_no_dann.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HARD_SIGNERS = ['PHW', 'PJH', 'SYB', 'KJM', 'KNY', 'LKS', 'KIM', 'YMG']
EASY_SIGNERS = ['KIS', 'YJH', 'KHY', 'HYW', 'KSH', 'JSA', 'YSY', 'KSJ']


def per_signer_scatter(
    per_signer_cer: dict[str, float],
    out_path:       str,
    *,
    title:          str = "Per-signer val CER",
    easy_thr:       float = 0.55,
    hard_thr:       float = 0.75,
    figsize:        tuple[float, float] = (10.0, 4.0),
    dpi:            int = 150,
) -> str:
    """
    Render the standardised per-signer CER scatter plot.

    Returns
    -------
    out_path (after the figure has been saved).
    """
    items  = sorted(per_signer_cer.items(), key=lambda kv: kv[1])
    xs     = list(range(len(items)))
    ys     = [v for _, v in items]
    colors = [
        '#d62728' if s in HARD_SIGNERS else
        '#1f77b4' if s in EASY_SIGNERS else
        '#7f7f7f'
        for s, _ in items
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, s=28, c=colors, edgecolor='black', linewidths=0.4)
    ax.axhline(easy_thr, color='green', linestyle='--', alpha=0.5,
               label=f'easy ≤ {easy_thr}')
    ax.axhline(hard_thr, color='red',   linestyle='--', alpha=0.5,
               label=f'hard ≥ {hard_thr}')
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _ in items], rotation=90, fontsize=7)
    ax.set_ylabel('val CER (best epoch)')
    ax.set_xlabel('signer (sorted)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _per_signer_from_results(results_path: str, variant: str) -> dict[str, float]:
    """Aggregate per-fold best_per_signer_val_cer for one variant."""
    with open(results_path) as f:
        results = json.load(f)
    out: dict[str, float] = {}
    for r in results:
        if r.get('variant') != variant:
            continue
        out.update(r.get('best_per_signer_val_cer', {}) or {})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', required=True,
        help='Path to stage*_results.json (list of fold dicts).')
    p.add_argument('--variant', required=True,
        help='Variant name to filter on (e.g. no_dann, stage3, stage4_early).')
    p.add_argument('--out', required=True, help='Output PNG path.')
    p.add_argument('--title', default=None,
        help='Plot title; defaults to "Per-signer val CER — <variant>".')
    args = p.parse_args(argv)

    per_signer = _per_signer_from_results(args.results, args.variant)
    if not per_signer:
        raise SystemExit(
            f"No per_signer entries found in {args.results} for variant={args.variant}"
        )
    title = args.title or f'Per-signer val CER — {args.variant}'
    out_path = per_signer_scatter(per_signer, args.out, title=title)
    print(f'wrote {out_path}  (n_signers={len(per_signer)})')


if __name__ == '__main__':
    main()

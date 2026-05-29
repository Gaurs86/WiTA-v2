"""
reports/template/cv_summary.py — reusable CV aggregation + Wilcoxon helper.

Standardised numerical reporting across stages per Task D.

Library API:
    summary    = cv_summary({"no_dann": [...]})
    paired     = paired_wilcoxon(a, b)                  # dict
    rendered   = render_table(summary)                  # str

CLI:
    python -m wita_v2.reports.template.cv_summary \
        --results /path/to/stage_results.json \
        --variants no_dann dann_a1 dann_a03 \
        --baseline no_dann
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon


def cv_summary(per_fold: dict[str, list[float]]) -> dict[str, dict]:
    """
    Aggregate per-variant CV results to mean ± std (and min/max).

    per_fold : {variant_name: [fold0_cer, fold1_cer, ...]}.  None / NaN
               entries are skipped.

    Returns {variant_name: {mean, std, min, max, n}}.
    """
    out: dict[str, dict] = {}
    for name, vals in per_fold.items():
        arr = np.array([v for v in vals if v is not None and not np.isnan(v)],
                       dtype=float)
        if not len(arr):
            out[name] = {'mean': float('nan'), 'std': float('nan'),
                         'min':  float('nan'), 'max': float('nan'), 'n': 0}
            continue
        out[name] = {
            'mean': float(arr.mean()),
            'std':  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'min':  float(arr.min()),
            'max':  float(arr.max()),
            'n':    int(len(arr)),
        }
    return out


def paired_wilcoxon(a: list[float], b: list[float]) -> dict:
    """
    Paired two-sided Wilcoxon signed-rank test.  Drops pairs where either
    side is None/NaN.

    Returns {W, p, n, mean_diff, n_a_lt_b, n_a_gt_b}.
    """
    pairs = [
        (x, y) for x, y in zip(a, b)
        if x is not None and y is not None
        and not np.isnan(x) and not np.isnan(y)
    ]
    if len(pairs) < 2:
        return dict(W=float('nan'), p=float('nan'), n=len(pairs),
                    mean_diff=float('nan'), n_a_lt_b=0, n_a_gt_b=0)
    aa = np.array([p[0] for p in pairs]); bb = np.array([p[1] for p in pairs])
    try:
        res = wilcoxon(aa, bb, zero_method='wilcox', alternative='two-sided')
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        # All-zero differences: degenerate but report it cleanly.
        W, p = 0.0, 1.0
    return dict(
        W=W, p=p, n=len(pairs),
        mean_diff=float((aa - bb).mean()),
        n_a_lt_b=int((aa < bb).sum()),
        n_a_gt_b=int((aa > bb).sum()),
    )


def render_table(summary: dict[str, dict]) -> str:
    lines = ['variant              mean      std       min       max       n']
    for name, d in summary.items():
        lines.append(
            f'{name:<20s} {d["mean"]:>7.4f}  {d["std"]:>7.4f}  '
            f'{d["min"]:>7.4f}  {d["max"]:>7.4f}  {d["n"]:>3d}'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_per_fold(results_path: str, variants: list[str]) -> dict[str, list]:
    with open(results_path) as f:
        results = json.load(f)
    folds = sorted({r['fold'] for r in results})
    out: dict[str, list] = {v: [None] * len(folds) for v in variants}
    for r in results:
        if r['variant'] in out:
            out[r['variant']][r['fold']] = r['best_val_cer']
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',  required=True)
    p.add_argument('--variants', nargs='+', required=True,
        help='Variant names to summarise.')
    p.add_argument('--baseline', default=None,
        help='Variant to use as the paired-Wilcoxon baseline (optional).')
    args = p.parse_args(argv)

    per_fold = _load_per_fold(args.results, args.variants)
    summary  = cv_summary(per_fold)
    print(render_table(summary))

    if args.baseline and args.baseline in per_fold:
        base = per_fold[args.baseline]
        for v in args.variants:
            if v == args.baseline:
                continue
            w = paired_wilcoxon(base, per_fold[v])
            print(
                f'\nPaired Wilcoxon  {args.baseline} vs {v}:\n'
                f'  W={w["W"]:.2f}  p={w["p"]:.4f}  n={w["n"]}\n'
                f'  mean_diff ({args.baseline} - {v}) = {w["mean_diff"]:+.4f}\n'
                f'  {args.baseline} < {v} in {w["n_a_lt_b"]}/{w["n"]} folds  '
                f'({args.baseline} > {v} in {w["n_a_gt_b"]}/{w["n"]} folds)'
            )


if __name__ == '__main__':
    main()

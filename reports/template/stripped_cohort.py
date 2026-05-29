"""
reports/template/stripped_cohort.py — full-cohort vs PHW/KIM-stripped CER.

Per the HRNet-swap experiment (verdict: null), PHW and KIM are accepted
as a dataset-side limit.  From Stage 3 onwards, every stage report emits
two numbers:

  * full_cohort_mean   — 5-fold mean over all val signers per fold
  * stripped_mean      — 5-fold mean computed AFTER excluding PHW (fold 2
                         val signers) and KIM (fold 1 val signers) from
                         each fold's per-signer denominator

The stripped number isolates model-side progress from a hard dataset-side
floor; the full number remains the headline that connects to comparable
literature.

Usage as a library:
    from wita_v2.reports.template.stripped_cohort import dual_cohort_summary
    summary = dual_cohort_summary('reports/stage3/stage3_results.json',
                                  variant='stage3')

CLI:
    python -m wita_v2.reports.template.stripped_cohort \
        --results /path/to/results.json \
        --variants stage3
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# The two signers identified by Task A as dataset-side limits and confirmed
# by the HRNet swap experiment as not tracker-recoverable.
DROPPED_SIGNERS = {"PHW", "KIM"}


def per_fold_stripped_cer(per_signer_cer: dict[str, float],
                          per_signer_len: dict[str, int] | None = None) -> float:
    """
    Compute one fold's CER after removing PHW/KIM from the per-signer dict.

    Without per_signer_len we fall back to a UNIFORMLY-WEIGHTED mean over
    the surviving signers — exactly correct only when all signers have
    similar val-clip counts.  Pass per_signer_len for the clip-weighted
    correct value.
    """
    surviving = {s: c for s, c in per_signer_cer.items()
                 if s not in DROPPED_SIGNERS}
    if not surviving:
        return float('nan')
    if per_signer_len is None:
        return float(np.mean(list(surviving.values())))
    num, den = 0.0, 0.0
    for s, cer in surviving.items():
        L = per_signer_len.get(s, 0)
        num += cer * L
        den += L
    return num / den if den > 0 else float('nan')


def dual_cohort_summary(
    results_path: str,
    variant:      str,
) -> dict:
    """
    Returns a dict with:
        full_per_fold     : list of best_val_cer per fold (with PHW/KIM)
        stripped_per_fold : list of best_val_cer per fold (without)
        full_mean / full_std
        stripped_mean / stripped_std
        n_dropped_folds   : which folds had PHW or KIM in their val set
    """
    with open(results_path) as f:
        results = json.load(f)
    fold_to_entry: dict[int, dict] = {}
    for r in results:
        if r.get('variant') != variant:
            continue
        fold_to_entry[r['fold']] = r
    folds = sorted(fold_to_entry.keys())

    full, stripped, dropped_in_fold = [], [], []
    for f in folds:
        e = fold_to_entry[f]
        full.append(float(e['best_val_cer']))
        per_signer = e.get('best_per_signer_val_cer', {}) or {}
        had_dropped = bool(set(per_signer) & DROPPED_SIGNERS)
        dropped_in_fold.append(had_dropped)
        # If no PHW/KIM signers in this fold's val, stripped == full.
        if not had_dropped:
            stripped.append(float(e['best_val_cer']))
        else:
            stripped.append(per_fold_stripped_cer(per_signer))

    f_arr = np.array(full); s_arr = np.array(stripped)
    return {
        'folds':              folds,
        'full_per_fold':      full,
        'stripped_per_fold':  stripped,
        'dropped_in_fold':    dropped_in_fold,
        'full_mean':          float(f_arr.mean()),
        'full_std':           float(f_arr.std(ddof=1)) if len(f_arr) > 1 else 0.0,
        'stripped_mean':      float(s_arr.mean()),
        'stripped_std':       float(s_arr.std(ddof=1)) if len(s_arr) > 1 else 0.0,
        'delta_mean':         float(f_arr.mean() - s_arr.mean()),
    }


def render_table(summaries: dict[str, dict]) -> str:
    lines = [
        '                       full-cohort         PHW/KIM-stripped     Δ (full-stripped)',
        '                       mean   ± std        mean   ± std         mean',
        '-' * 84,
    ]
    for name, s in summaries.items():
        lines.append(
            f'{name:<22s} {s["full_mean"]:.4f} ± {s["full_std"]:.4f}    '
            f'{s["stripped_mean"]:.4f} ± {s["stripped_std"]:.4f}    '
            f'{s["delta_mean"]:+.4f}'
        )
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',  required=True)
    p.add_argument('--variants', nargs='+', required=True)
    args = p.parse_args(argv)

    out: dict[str, dict] = {}
    for v in args.variants:
        out[v] = dual_cohort_summary(args.results, v)
    print(render_table(out))
    print('\nFolds containing PHW or KIM in their val set '
          '(stripped recomputed for these):')
    for v, s in out.items():
        affected = [f for f, b in zip(s['folds'], s['dropped_in_fold']) if b]
        print(f'  {v}: folds {affected}')


if __name__ == '__main__':
    main()

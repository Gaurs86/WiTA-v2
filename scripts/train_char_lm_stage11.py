"""
scripts/train_char_lm_stage11.py — train a char n-gram LM on the WiTA
paper-split train labels (Stage 11 prerequisite for the 9b decoder).

The Stage 9a-version of this script used the 38-signer fold-N labels; for
Stage 11 the LM trains on the full 122-signer TRAIN split (no signers from
val or test appear -- subject-disjoint).

Usage:
    python scripts/train_char_lm_stage11.py \
        --cache-root /kaggle/input/wita-full-english-landmark-cache/landmark_cache_122 \
        --subsets    lex nonlex \
        --order      4 \
        --out        /kaggle/working/lm/wita_train_4gram.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--subsets',    nargs='+', default=['lex', 'nonlex'])
    p.add_argument('--order',      type=int,  default=4)
    p.add_argument('--out',        required=True)
    p.add_argument('--log-level',  default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )
    log = logging.getLogger('train_char_lm_stage11')

    from wita_v2.models.char_ngram_lm import CharNgramLM

    cache_root = Path(args.cache_root)
    corpus: list[str] = []
    for subset in args.subsets:
        d = cache_root / "train" / subset
        if not d.exists():
            raise SystemExit(f"Missing {d}")
        for npz in sorted(d.glob("*.npz")):
            with np.load(npz, allow_pickle=False) as data:
                corpus.append(str(data['label'].item()).lower())
    log.info("Collected %d train labels from %s.",
             len(corpus), args.subsets)

    lm = CharNgramLM(order=args.order)
    lm.train(corpus)
    log.info("Trained: %s", lm)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    lm.save(args.out)
    log.info("Wrote %s", args.out)
    print(lm)


if __name__ == '__main__':
    main()

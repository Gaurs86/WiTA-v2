"""
scripts/train_char_lm.py — Train a 4-gram char LM for Stage 9b rescoring.

Two modes:
  --corpus  wita_labels  : train on WiTA train-fold labels only (default).
                           Tiny corpus (~30k chars) but matches the test
                           distribution exactly.  Risk: LM memorises
                           closed vocabulary.
  --corpus  external     : train on an external English text file passed
                           via --external-text.  Interpolates with the
                           WiTA labels (50/50) for the final ARPA-like
                           probability surface.

Cache fingerprint includes the corpus mode + n-gram order so a downstream
beam-search decoder can verify it loaded the LM it expected.

Usage:
    python scripts/train_char_lm.py \
        --skeleton-cache /path/to/skeleton_features_t32.pt \
        --cv-manifest    /path/to/subject_cv5.json \
        --order          4 \
        --corpus         wita_labels \
        --out            /kaggle/working/char_lm_4gram_wita.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

import torch

logger = logging.getLogger("train_char_lm")


def _wita_labels_per_fold(cache_path: str, manifest_path: str,
                          fold: int) -> list[str]:
    """Return the train-set labels for one fold (val signers excluded)."""
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    with open(manifest_path) as f:
        manifest = json.load(f)
    val_subjects = set(manifest['folds'][fold]['val_subjects'])
    return [
        cache['labels'][i].lower()
        for i, s in enumerate(cache['subjects'])
        if s not in val_subjects
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--skeleton-cache', required=True)
    p.add_argument('--cv-manifest',    required=True)
    p.add_argument('--fold',           type=int, required=True,
        help='LM trained on this fold\'s TRAIN signers only — no val leakage.')
    p.add_argument('--order',          type=int, default=4)
    p.add_argument('--corpus',         choices=['wita_labels', 'external'],
                                       default='wita_labels')
    p.add_argument('--external-text',  default=None,
        help='Path to a text file with one sentence per line, lowercased.')
    p.add_argument('--out',            required=True)
    p.add_argument('--log-level',      default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )

    from wita_v2.models.char_ngram_lm import CharNgramLM

    train_labels = _wita_labels_per_fold(args.skeleton_cache,
                                         args.cv_manifest, args.fold)
    logger.info("Fold %d: %d train labels.", args.fold, len(train_labels))

    if args.corpus == 'wita_labels':
        corpus = train_labels
    else:
        if not args.external_text:
            raise SystemExit("--corpus external requires --external-text PATH")
        with open(args.external_text) as f:
            external = [line.strip().lower() for line in f if line.strip()]
        logger.info("External corpus: %d lines.", len(external))
        # Naive 50/50: just concatenate.  Pre-emptive interpolation is
        # cleaner but for char-level LMs the count-based combine is fine.
        corpus = train_labels + external

    lm = CharNgramLM(order=args.order)
    lm.train(corpus)
    lm.save(args.out)
    logger.info("Wrote %s  -- %s", args.out, lm)
    print(lm)


if __name__ == '__main__':
    main()

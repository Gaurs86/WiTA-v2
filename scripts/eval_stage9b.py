"""
scripts/eval_stage9b.py — Stage 9b rescoring on Stage 9a checkpoints.

No retraining.  For each fold:
  1. Load the Stage 9a checkpoint (encoder + decoder).
  2. Re-encode all val clips (greedy attention decode for the attn row).
  3. Run CTC prefix beam search with the per-fold trained char LM.
  4. Per-clip headline = argmin over {CTC greedy, attention greedy, CTC+LM beam}.

Output: stage9b_results.json  with per-fold val CER for each decoder mode
        + the headline best-of-three.  Schema-compatible with the
        Stage 9a results JSON (same per_fold structure + per_signer_cer)
        so the existing dual-cohort summary and PDF builders just work.

Usage (Kaggle):
    python scripts/eval_stage9b.py \
        --skeleton-cache /path/to/skeleton_features_t32.pt \
        --cv-manifest    /path/to/subject_cv5.json \
        --checkpoint-glob "/kaggle/input/*/checkpoints/stage9a_fold*_best.pt" \
        --lm-glob        "/kaggle/input/*/char_lm_4gram_wita_fold*.pkl" \
        --out            /kaggle/working/stage9b_results.json \
        --alpha          0.5 \
        --beta           0.0 \
        --beam           32
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

import numpy as np
import torch
import editdistance

logger = logging.getLogger("eval_stage9b")


# ---------------------------------------------------------------------------

def _load_per_fold_artifacts(ckpt_glob: str, lm_glob: str):
    """Build {fold: (ckpt_path, lm_path)}.  Per-fold check ensures LM not trained on val."""
    ck = sorted(glob.glob(ckpt_glob))
    lm = sorted(glob.glob(lm_glob))
    def _fold_of(path):
        m = re.search(r'fold(\d+)', os.path.basename(path))
        if not m:
            raise SystemExit(f'Could not extract fold N from {path}')
        return int(m.group(1))
    out: dict[int, tuple[str, str]] = {}
    by_fold_ck = {_fold_of(p): p for p in ck}
    by_fold_lm = {_fold_of(p): p for p in lm}
    for f, cpath in by_fold_ck.items():
        if f not in by_fold_lm:
            raise SystemExit(f'fold {f}: checkpoint found at {cpath} but '
                             f'no matching LM in {lm_glob}')
        out[f] = (cpath, by_fold_lm[f])
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--skeleton-cache', required=True)
    p.add_argument('--cv-manifest',    required=True)
    p.add_argument('--checkpoint-glob', required=True,
        help='Glob for stage9a_fold*_best.pt files.')
    p.add_argument('--lm-glob',        required=True,
        help='Glob for char_lm_4gram_*_fold*.pkl files.')
    p.add_argument('--out',            required=True)
    p.add_argument('--alpha',          type=float, default=0.5)
    p.add_argument('--beta',           type=float, default=0.0)
    p.add_argument('--beam',           type=int,   default=32)
    p.add_argument('--variant',        default='stage9b')
    p.add_argument('--log-level',      default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )

    from wita_v2.configs.default            import Config, DataConfig, EncoderConfig, TrainConfig
    from wita_v2.datasets.cv_splits         import fold_indices, load_cv5_manifest
    from wita_v2.datasets.vocab             import make_converter
    from wita_v2.models.conformer_ctc       import ConformerCTC
    from wita_v2.models.attention_decoder   import AttentionDecoder
    from wita_v2.models.char_ngram_lm       import CharNgramLM
    from wita_v2.inference.beam_search      import ctc_prefix_beam_search

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config(
        data=DataConfig(hf_repo_id='yewon816/WiTA', lang='english',
                        max_zips=None, max_frames=64, seed=42),
        encoder=EncoderConfig(arch='siglip'),
        train=TrainConfig(num_epochs=1, batch_size=32, seed=42,
                          checkpoint_dir='/tmp/_unused'),
    ).build()
    converter = make_converter(cfg.data.lang)

    cache    = torch.load(args.skeleton_cache, map_location='cpu', weights_only=False)
    manifest = load_cv5_manifest(args.cv_manifest)

    # Map CTC ids -> char strings  (1..N for chars, N+1 for sep)
    chars = cfg.vocab.chars
    id_to_char: dict[int, str] = {i + 1: c for i, c in enumerate(chars)}
    # CTC repeat separator at sep_idx is internal; drop in final string.
    sep_idx = cfg.vocab.sep_idx

    artifacts = _load_per_fold_artifacts(args.checkpoint_glob, args.lm_glob)
    logger.info("Found %d folds with both checkpoint and LM.", len(artifacts))

    all_results: list[dict] = []
    for fold, (ckpt_path, lm_path) in sorted(artifacts.items()):
        logger.info("[fold %d] ckpt=%s  lm=%s", fold, ckpt_path, lm_path)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        lm   = CharNgramLM.load(lm_path)

        # Build encoder + decoder, load weights.
        encoder = ConformerCTC(
            input_dim=cache['out_dim'], vocab_size=cfg.vocab.ctc_vocab_size,
            d_model=256, n_layers=4, n_heads=4, conv_kernel=15,
            dropout=0.2, upsample=2, input_layernorm=False,
        ).to(device).eval()
        encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)

        decoder = AttentionDecoder(
            att_vocab_size=cfg.vocab.attn_vocab_size,
            bos_idx=cfg.vocab.sos_idx, eos_idx=cfg.vocab.eos_idx,
            d_model=256, n_layers=3, n_heads=4, ff_mult=4, dropout=0.2,
        ).to(device).eval()
        decoder.load_state_dict(ckpt['decoder_state_dict'], strict=False)

        _, val_idx = fold_indices(manifest, fold, cache['subjects'])
        logger.info("[fold %d] %d val clips", fold, len(val_idx))

        cer_ctc_g_n = cer_ctc_g_d = 0
        cer_attn_n  = cer_attn_d  = 0
        cer_beam_n  = cer_beam_d  = 0
        cer_best_n  = cer_best_d  = 0
        per_subj_err: dict[str, int] = {}
        per_subj_len: dict[str, int] = {}

        with torch.no_grad():
            for ci in val_idx:
                feats = cache['feats'][ci].float().unsqueeze(0).to(device)   # [1,T,190]
                in_lens = torch.LongTensor([feats.shape[1]]).to(device)
                h, pad_mask = encoder.encode(feats, in_lens)
                log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
                lp = log_probs[0].float().cpu().numpy()              # [T_out, V]

                # Reference.
                gt = cache['labels'][ci].lower()
                subj = cache['subjects'][ci]

                # 1) CTC greedy.
                argmax = lp.argmax(axis=-1)
                merged: list[int] = []; prev = None
                for t in argmax:
                    if t != prev and t != cfg.vocab.blank_idx:
                        merged.append(int(t))
                    prev = t
                pred_ctc = ''.join(
                    id_to_char[i] for i in merged
                    if i in id_to_char and i != sep_idx
                )

                # 2) Attention greedy.
                attn_ids = decoder.greedy_decode(h, pad_mask)[0].tolist()
                pred_attn = ''.join(
                    id_to_char[i] for i in attn_ids
                    if i in id_to_char and i != sep_idx
                )

                # 3) CTC + LM prefix beam search.
                pred_beam, _ = ctc_prefix_beam_search(
                    lp, blank=cfg.vocab.blank_idx, sep=sep_idx,
                    beam=args.beam,
                    id_to_char=id_to_char, lm=lm,
                    alpha=args.alpha, beta=args.beta,
                )

                # 4) Best of three per clip.
                e_ctc  = editdistance.eval(gt, pred_ctc)
                e_attn = editdistance.eval(gt, pred_attn)
                e_beam = editdistance.eval(gt, pred_beam)
                e_best = min(e_ctc, e_attn, e_beam)
                Lref = len(gt)

                cer_ctc_g_n += e_ctc;  cer_ctc_g_d += Lref
                cer_attn_n  += e_attn; cer_attn_d  += Lref
                cer_beam_n  += e_beam; cer_beam_d  += Lref
                cer_best_n  += e_best; cer_best_d  += Lref

                per_subj_err[subj] = per_subj_err.get(subj, 0) + e_best
                per_subj_len[subj] = per_subj_len.get(subj, 0) + Lref

        def safe_cer(n, d): return n / max(d, 1)
        cer_ctc  = safe_cer(cer_ctc_g_n, cer_ctc_g_d)
        cer_attn = safe_cer(cer_attn_n,  cer_attn_d)
        cer_beam = safe_cer(cer_beam_n,  cer_beam_d)
        cer_best = safe_cer(cer_best_n,  cer_best_d)

        per_signer_cer = {s: per_subj_err[s] / max(per_subj_len[s], 1)
                          for s in per_subj_err}
        logger.info(
            "[fold %d] CER  ctc=%.4f  attn=%.4f  beam=%.4f  best=%.4f",
            fold, cer_ctc, cer_attn, cer_beam, cer_best,
        )

        all_results.append({
            'fold':            fold,
            'variant':         args.variant,
            'best_val_cer':    cer_best,
            'cer_ctc':         cer_ctc,
            'cer_attn':        cer_attn,
            'cer_beam':        cer_beam,
            'alpha':           args.alpha,
            'beta':            args.beta,
            'beam':            args.beam,
            'best_per_signer_val_cer': per_signer_cer,
            'checkpoint':      ckpt_path,
            'lm':              lm_path,
        })
        # Persist after each fold.
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  saved -> {args.out}')

    print('\n=== Stage 9b summary ===')
    for r in all_results:
        print(f"  fold {r['fold']}: CER ctc={r['cer_ctc']:.4f}  "
              f"attn={r['cer_attn']:.4f}  beam={r['cer_beam']:.4f}  "
              f"best={r['best_val_cer']:.4f}")
    arr = np.array([r['best_val_cer'] for r in all_results])
    if len(arr) >= 2:
        print(f'\n  mean ± std (best): {arr.mean():.4f} ± {arr.std(ddof=1):.4f}')


if __name__ == '__main__':
    main()

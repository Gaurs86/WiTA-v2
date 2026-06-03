"""
scripts/eval_stage11_decoders.py — Stage 11 decoder-side fixes (no retraining).

Operates on a saved Stage 11 checkpoint and a paper-split landmark cache.
Supports five decoder modes:

  * ctc_greedy   : CTC argmax + blank/duplicate collapse  (Stage 11 baseline component)
  * attn_greedy  : attention argmax per step              (Stage 11 baseline component)
  * ctc_lm_beam  : Stage 9b — CTC prefix beam search with char-LM shallow fusion
  * attn_beam    : B1   — attention beam search with length normalisation
  * joint        : B3   — attn beam search + joint CTC + attn + LM rescoring

Two run modes:

  --on val       : free hyperparameter sweep, prints per-config CER on val
  --on test      : SINGLE evaluation with the val-best hyperparameters
                   from --hparams JSON (or hard-coded in the notebook)

The test path writes a marker file at /kaggle/working/logs/
.stage11_decfix_test_evaluated_<config_hash>; re-running with the same
config raises a RuntimeError per §7.

Usage examples
--------------
Train LM once:
  python scripts/train_char_lm_stage11.py --cache-root <cache> --out lm.pkl

Sweep KenLM (alpha, beta) on val:
  python scripts/eval_stage11_decoders.py \
      --cache-root <cache> --checkpoint stage11_best.pt \
      --lm lm.pkl --mode ctc_lm_beam --on val \
      --sweep alpha=0.3,0.5,0.7,1.0 beta=0.0,0.5,1.0

Single test eval with the val-best config:
  python scripts/eval_stage11_decoders.py \
      --cache-root <cache> --checkpoint stage11_best.pt \
      --lm lm.pkl --mode ctc_lm_beam --on test \
      --hparams '{"alpha":0.5,"beta":0.5}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import editdistance

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


# ---------------------------------------------------------------------------

LENGTH_BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 999)]
def _bucket(L: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= L <= hi: return f"{lo}-{hi if hi < 999 else 'inf'}"
    return "unknown"


def _build_models(ckpt_path: str, cfg, device: str):
    """Load stage11_best.pt into encoder + decoder."""
    from wita_v2.models.conformer_ctc     import ConformerCTC
    from wita_v2.models.attention_decoder import AttentionDecoder

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    dec_n_layers = int(state.get('dec_n_layers', 2))
    encoder = ConformerCTC(
        input_dim=190, vocab_size=cfg.vocab.ctc_vocab_size,
        d_model=256, n_layers=4, n_heads=4, conv_kernel=15,
        dropout=0.2, upsample=2, input_layernorm=False,
    ).to(device).eval()
    decoder = AttentionDecoder(
        att_vocab_size=cfg.vocab.attn_vocab_size,
        bos_idx=cfg.vocab.sos_idx, eos_idx=cfg.vocab.eos_idx,
        d_model=256, n_layers=dec_n_layers, n_heads=4,
        ff_mult=4, dropout=0.2,
    ).to(device).eval()
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoder.load_state_dict(state['decoder_state_dict'], strict=False)
    return encoder, decoder, state


# ---------------------------------------------------------------------------

def _evaluate(encoder, decoder, loader, cfg, device, *,
              mode, hparams, lm=None):
    """
    Walk a paper-split loader and decode each clip via the chosen strategy.
    Returns aggregate dict {overall_cer, lex_cer, nonlex_cer,
                            length_cer, n_clips, per_signer}.
    """
    from wita_v2.inference.joint_decode import decode_one_clip

    pairs_by_subset: dict[str, list[dict]] = defaultdict(list)
    per_signer_err: dict[str, int] = defaultdict(int)
    per_signer_len: dict[str, int] = defaultdict(int)
    per_bucket_err: dict[str, int] = defaultdict(int)
    per_bucket_len: dict[str, int] = defaultdict(int)
    n_total_err = n_total_len = 0
    t0 = time.time()
    n_done = 0
    for feats, labels, in_lens, lab_lens, signers, subsets, label_strs in loader:
        feats   = feats.to(device);   labels   = labels.to(device)
        in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)
        B = feats.size(0)
        for b in range(B):
            gt   = label_strs[b]
            pred = decode_one_clip(
                encoder, decoder,
                feats[b:b+1], in_lens[b:b+1],
                cfg=cfg, mode=mode, lm=lm, **hparams,
            )
            err  = int(editdistance.eval(gt, pred))
            L    = len(gt)
            pairs_by_subset[subsets[b]].append(
                {'gt': gt, 'pred': pred, 'err': err, 'L': L,
                 'signer': signers[b], 'subset': subsets[b]}
            )
            n_total_err += err; n_total_len += L
            per_signer_err[signers[b]] += err; per_signer_len[signers[b]] += L
            bb = _bucket(L)
            per_bucket_err[bb] += err; per_bucket_len[bb] += L
            n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {n_done} clips  ({n_done/max(elapsed,1e-3):.1f} clips/s)  "
                  f"running CER {n_total_err/max(n_total_len,1):.4f}",
                  flush=True)
    overall = n_total_err / max(n_total_len, 1)
    per_subset = {}
    for ss, rows in pairs_by_subset.items():
        ne = sum(r['err'] for r in rows)
        nl = sum(r['L']   for r in rows)
        per_subset[ss] = ne / max(nl, 1)
    per_signer = {s: per_signer_err[s] / max(per_signer_len[s], 1)
                  for s in per_signer_err}
    per_length = {b: per_bucket_err[b] / max(per_bucket_len[b], 1)
                  for b in per_bucket_err}
    return {
        'overall_cer': overall,
        'lex_cer':     per_subset.get('lex',    float('nan')),
        'nonlex_cer':  per_subset.get('nonlex', float('nan')),
        'length_cer':  per_length,
        'per_signer':  per_signer,
        'n_clips':     n_total_len and (n_done),
    }


# ---------------------------------------------------------------------------

def _config_hash(mode: str, hparams: dict) -> str:
    h = {'mode': mode, **{k: hparams[k] for k in sorted(hparams)}}
    return hashlib.md5(json.dumps(h, sort_keys=True).encode()).hexdigest()[:10]


def _parse_sweep(items: list[str]) -> dict[str, list]:
    """
    Parse --sweep 'alpha=0.3,0.5,0.7' 'beta=0.0,1.0' 'beam=8' into a dict.

    Values are parsed as int when possible (no decimal point) and float
    otherwise.  Integer-typed hyperparameters like beam width and
    symbol_top_k must remain int because numpy / torch APIs are strict.
    """
    def _smart_cast(s: str):
        s = s.strip()
        # Honour explicit int form (no '.'/'e'/'E') -> int.
        if s and all(c in '0123456789-+' for c in s):
            try:
                return int(s)
            except ValueError:
                pass
        return float(s)

    out: dict[str, list] = {}
    for it in items:
        k, v = it.split('=', 1)
        out[k] = [_smart_cast(x) for x in v.split(',') if x.strip()]
    return out


def _iter_sweep(sweep: dict[str, list[float]]):
    """Cartesian product of named axes."""
    import itertools
    if not sweep: yield {}; return
    keys = list(sweep.keys())
    for combo in itertools.product(*(sweep[k] for k in keys)):
        yield {k: v for k, v in zip(keys, combo)}


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--lm',         default=None,
                   help='Path to a CharNgramLM .pkl (optional; required for LM modes).')
    p.add_argument('--mode', required=True,
                   choices=['ctc_greedy', 'attn_greedy', 'ctc_beam',
                            'ctc_lm_beam', 'attn_beam', 'joint'])
    p.add_argument('--on',   required=True, choices=['val', 'test'])
    p.add_argument('--sweep', nargs='*', default=[],
                   help="Sweep axes, e.g. 'alpha=0.3,0.5' 'beta=0.0,1.0'. "
                        'Only valid with --on val.')
    p.add_argument('--hparams', default='{}',
                   help='JSON dict of hyperparameters for --on test. '
                        'Pass the val-best config here.')
    p.add_argument('--out', default='/kaggle/working/logs/stage11_decoders_results.json')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s')

    from torch.utils.data import DataLoader
    from wita_v2.configs.default              import Config, DataConfig, EncoderConfig, TrainConfig
    from wita_v2.datasets.landmark_paper_split import WiTAPaperSplitDataset
    from wita_v2.training.stage11_train       import _collate
    from wita_v2.datasets.vocab               import make_converter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config(
        data=DataConfig(hf_repo_id='yewon816/WiTA', lang='english',
                        max_zips=None, max_frames=64, seed=42),
        encoder=EncoderConfig(arch='siglip'),
        train=TrainConfig(num_epochs=1, batch_size=args.batch_size, seed=42,
                          checkpoint_dir='/tmp/_unused'),
    ).build()
    converter = make_converter(cfg.data.lang)
    encoder, decoder, state = _build_models(args.checkpoint, cfg, device)

    # Optional LM.
    lm = None
    if args.lm:
        from wita_v2.models.char_ngram_lm import CharNgramLM
        lm = CharNgramLM.load(args.lm)
        print(f'Loaded LM: {lm}', flush=True)

    ds = WiTAPaperSplitDataset(
        args.cache_root, args.on, subsets=('lex', 'nonlex'),
        converter=converter, transform=None,
    )
    coll = lambda b: _collate(b, pad_idx=cfg.vocab.pad_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=coll)
    print(f'Eval split={args.on}  n_clips={len(ds)}  mode={args.mode}',
          flush=True)

    # ----- val sweep -----
    if args.on == 'val':
        sweep = _parse_sweep(args.sweep)
        # If no --sweep axes were given, honour --hparams as the single
        # config.  Previously this branch ignored --hparams entirely,
        # producing rows with `hparams={}` and the script's hard-coded
        # decode defaults.
        if not sweep:
            single_hp = json.loads(args.hparams) if args.hparams else {}
            configs = [single_hp]
        else:
            configs = list(_iter_sweep(sweep))
        results: list[dict] = []
        for hp in configs:
            print(f'\n--- val sweep config: {hp}')
            out = _evaluate(encoder, decoder, loader, cfg, device,
                            mode=args.mode, hparams=hp, lm=lm)
            row = {'mode': args.mode, 'hparams': hp, **out}
            print(f'  val_overall_cer = {out["overall_cer"]:.4f}  '
                  f'lex={out["lex_cer"]:.4f}  nonlex={out["nonlex_cer"]:.4f}  '
                  f'len_bucket={out["length_cer"]}')
            results.append(row)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        # Best by overall_cer.
        best = min(results, key=lambda r: r['overall_cer'])
        print(f'\nBest val config: {best["hparams"]}  '
              f'overall_cer = {best["overall_cer"]:.4f}')
        return 0

    # ----- single test eval (gated by marker) -----
    hp = json.loads(args.hparams)
    h = _config_hash(args.mode, hp)
    marker = f'/kaggle/working/logs/.stage11_decfix_test_evaluated_{h}'
    if os.path.exists(marker):
        raise SystemExit(
            f'Test already evaluated for this config (marker {marker}).  '
            'Per §7 of the prompt the test set must be touched once per row.'
        )
    out = _evaluate(encoder, decoder, loader, cfg, device,
                    mode=args.mode, hparams=hp, lm=lm)
    summary = {
        'mode':              args.mode,
        'hparams':           hp,
        'test_overall_cer':  out['overall_cer'],
        'test_lex_cer':      out['lex_cer'],
        'test_nonlex_cer':   out['nonlex_cer'],
        'test_length_cer':   out['length_cer'],
        'test_per_signer':   out['per_signer'],
        'paper_baseline':    {'overall': 0.2924, 'lex': 0.281, 'nonlex': 0.365},
        'stage11_baseline':  {'overall': 0.4498, 'lex': 0.4348, 'nonlex': 0.5449},
        'checkpoint':        args.checkpoint,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(marker) or '.', exist_ok=True)
    with open(marker, 'w') as f:
        f.write(args.out)
    print('\n' + '=' * 64)
    print(f'  STAGE 11 DECODER FIX  test headline  ({args.mode})')
    print('=' * 64)
    print(f'  hparams           : {hp}')
    print(f'  test_overall_cer  : {out["overall_cer"]:.4f}   '
          f'(stage11={0.4498}, paper={0.2924})')
    print(f'  test_lex_cer      : {out["lex_cer"]:.4f}   '
          f'(stage11={0.4348}, paper={0.281})')
    print(f'  test_nonlex_cer   : {out["nonlex_cer"]:.4f}   '
          f'(stage11={0.5449}, paper={0.365})')
    print(f'  test_length_cer   : {out["length_cer"]}')
    print('=' * 64 + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

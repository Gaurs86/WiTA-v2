"""
scripts/eval_phw_kim_hrnet.py — HRNet-swap experiment §3.6.

Evaluates the EXISTING Stage 1 v3 no_dann checkpoint (architecturally
identical to Stage 1 v2) on each per-backend cache produced by
`extract_phw_kim_keypoints.py`.

Per the §8 contract, the ONLY thing that changes between rows is the
keypoint backend.  Model weights, head architecture, decoder, blank index,
T_native, upsample, vocab — everything else is identical to the locked
Stage 1 v2 recipe.

Fold mapping (derived from logs.zip's per-fold history JSONs):
    PHW -> fold 2 (no_dann best val CER 0.6993)
    KIM -> fold 1 (no_dann best val CER 0.6955)

Outputs:
    reports/hrnet_swap/per_signer_results.csv
    reports/hrnet_swap/per_clip_cer.csv          (one row per (clip, backend))
    reports/hrnet_swap/per_clip_scatter.png
    reports/hrnet_swap/verdict.json

Usage (Kaggle):
    python scripts/eval_phw_kim_hrnet.py \
        --cache-dir       /kaggle/working/caches/hrnet_swap \
        --checkpoint-fold1 /kaggle/working/checkpoints/stage1v3_fold1_no_dann_best.pt \
        --checkpoint-fold2 /kaggle/working/checkpoints/stage1v3_fold2_no_dann_best.pt \
        --out-dir         /kaggle/working/reports/hrnet_swap
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import editdistance
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger("eval_phw_kim")


# Hard-coded fold assignment (single-experiment script; see module docstring).
SIGNER_TO_FOLD = {"PHW": 2, "KIM": 1}
BACKENDS = ["mediapipe_default", "mediapipe_sensitive", "rtmpose_hand"]


# ---------------------------------------------------------------------------
# Dataset wrapper that selects a single signer's clips from a backend cache
# ---------------------------------------------------------------------------

class _SignerSubset(Dataset):
    def __init__(self, cache: dict, signer: str, converter):
        self.cache = cache
        self.indices = [i for i, s in enumerate(cache["subjects"]) if s == signer]
        if not self.indices:
            raise RuntimeError(
                f"Signer {signer} not present in cache (subjects: "
                f"{sorted(set(cache['subjects']))})."
            )
        self.converter = converter

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        feats = self.cache["feats"][idx].float()
        enc, _ = self.converter.encode(self.cache["labels"][idx])
        return feats, enc, self.cache["labels"][idx]


def _collate(batch, pad_idx: int):
    feats, labels, label_strs = zip(*batch)
    feats_pad  = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.LongTensor([f.shape[0] for f in feats])
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return feats_pad, labels_pad, input_lens, label_lens, list(label_strs)


def _decode_argmax(log_probs: torch.Tensor, enc_lens: torch.Tensor, blank: int):
    out = []
    argmax = log_probs.argmax(dim=-1)
    for b in range(argmax.shape[0]):
        seq = argmax[b, : int(enc_lens[b].item())].tolist()
        merged = []; prev = None
        for t in seq:
            if t != prev and t != blank:
                merged.append(t)
            prev = t
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _build_stage1v2_model(cfg, device: str):
    from wita_v2.models.conformer_ctc import ConformerCTC
    model = ConformerCTC(
        input_dim   = 190,
        vocab_size  = cfg.vocab.ctc_vocab_size,
        d_model     = 256,
        n_layers    = 4,
        n_heads     = 4,
        conv_kernel = 15,
        dropout     = 0.2,
        upsample    = 2,
        input_layernorm = False,   # Stage 1 v2 default
    ).to(device)
    return model


def _load_checkpoint(model: nn.Module, ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing state_dict keys (first 5): %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected state_dict keys (first 5): %s", unexpected[:5])
    return state


# ---------------------------------------------------------------------------
# Per-(backend, signer) evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cer(
    model:    nn.Module,
    cache:    dict,
    signer:   str,
    *,
    cfg,
    batch_size: int = 32,
) -> tuple[float, list[dict]]:
    converter = _make_converter(cfg.data.lang)
    pad_idx = cfg.vocab.pad_idx
    blank   = cfg.vocab.blank_idx
    ds = _SignerSubset(cache, signer, converter)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate(b, pad_idx=pad_idx),
    )
    model.eval()
    total_err = 0; total_len = 0
    per_clip_rows: list[dict] = []
    for feats, labels, in_lens, lab_lens, label_strs in loader:
        feats   = feats.to(next(model.parameters()).device)
        in_lens = in_lens.to(feats.device)
        log_probs, enc_lens = model(feats, in_lens)
        preds = _decode_argmax(log_probs, enc_lens, blank)
        for b in range(len(preds)):
            gt = label_strs[b]
            pred_str = "".join(
                cfg.vocab.chars[t-1] if 1 <= t <= len(cfg.vocab.chars) else "?"
                for t in preds[b]
            )
            err = int(editdistance.eval(gt, pred_str))
            total_err += err; total_len += len(gt)
            per_clip_rows.append({
                "signer":   signer,
                "label":    gt,
                "pred":     pred_str,
                "edit_dist": err,
                "len":      len(gt),
                "cer":      err / max(len(gt), 1),
            })
    return total_err / max(total_len, 1), per_clip_rows


def _make_converter(lang: str):
    from wita_v2.datasets.vocab import make_converter
    return make_converter(lang)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-dir', required=True,
        help='Dir produced by extract_phw_kim_keypoints.py.')
    p.add_argument('--checkpoint-fold1', required=True,
        help='Path to stage1v3_fold1_no_dann_best.pt (KIM eval).')
    p.add_argument('--checkpoint-fold2', required=True,
        help='Path to stage1v3_fold2_no_dann_best.pt (PHW eval).')
    p.add_argument('--out-dir',  default='reports/hrnet_swap')
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # Build Stage 1 v2 cfg.
    from wita_v2.configs.default import (
        Config, DataConfig, EncoderConfig, TrainConfig,
    )
    cfg = Config(
        data=DataConfig(hf_repo_id='yewon816/WiTA', lang='english',
                        max_zips=None, max_frames=64, seed=42),
        encoder=EncoderConfig(arch='siglip'),
        train=TrainConfig(num_epochs=1, batch_size=32, seed=42,
                          checkpoint_dir='/tmp/_unused'),
    ).build()

    # Load both checkpoints into separate model instances.
    model_fold1 = _build_stage1v2_model(cfg, args.device)
    _load_checkpoint(model_fold1, args.checkpoint_fold1)
    model_fold2 = _build_stage1v2_model(cfg, args.device)
    _load_checkpoint(model_fold2, args.checkpoint_fold2)

    # Per-backend eval.
    results: dict[tuple[str, str], float] = {}
    per_clip_all: list[dict] = []
    detect_rates: dict[str, float] = {}
    for backend in BACKENDS:
        cache_path = os.path.join(args.cache_dir, f"landmarks_{backend}_phw_kim.pt")
        if not os.path.exists(cache_path):
            logger.warning("[%s] cache missing at %s — skipping.", backend, cache_path)
            continue
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        detect_rates[backend] = float(cache.get('frame_detect_rate', float('nan')))
        for signer in ['PHW', 'KIM']:
            try:
                fold = SIGNER_TO_FOLD[signer]
                model = model_fold1 if fold == 1 else model_fold2
                cer, rows = evaluate_cer(model, cache, signer, cfg=cfg)
                for r in rows:
                    r['backend'] = backend
                per_clip_all.extend(rows)
                results[(backend, signer)] = cer
                print(f'  {backend:<22s}  {signer}  CER={cer:.4f}  '
                      f'(over {sum(1 for r in rows)} clips)')
            except Exception as e:
                logger.error('Eval failed for (%s, %s): %s', backend, signer, e)

    # Per-signer summary.
    summary_rows = []
    for backend in BACKENDS:
        for signer in ['PHW', 'KIM']:
            summary_rows.append({
                'backend': backend,
                'signer':  signer,
                'cer':     results.get((backend, signer)),
                'detect_rate_global': detect_rates.get(backend),
            })
    csv_path = os.path.join(args.out_dir, 'per_signer_results.csv')
    with open(csv_path, 'w') as f:
        f.write('backend,signer,cer,detect_rate\n')
        for r in summary_rows:
            f.write(
                f'{r["backend"]},{r["signer"]},'
                + (f'{r["cer"]:.4f}' if r["cer"] is not None else '')
                + ','
                + (f'{r["detect_rate_global"]:.4f}' if r["detect_rate_global"] is not None else '')
                + '\n'
            )
    print(f'wrote {csv_path}')

    # Per-clip CSV.
    clip_csv = os.path.join(args.out_dir, 'per_clip_cer.csv')
    with open(clip_csv, 'w') as f:
        f.write('backend,signer,label,pred,edit_dist,len,cer\n')
        for r in per_clip_all:
            f.write(
                f'{r["backend"]},{r["signer"]},"{r["label"]}","{r["pred"]}",'
                f'{r["edit_dist"]},{r["len"]},{r["cer"]:.4f}\n'
            )
    print(f'wrote {clip_csv}')

    # Per-clip scatter.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = {'mediapipe_default': '#1f77b4',
                  'mediapipe_sensitive': '#ff7f0e',
                  'rtmpose_hand': '#2ca02c'}
        markers = {'PHW': 'o', 'KIM': 's'}
        for backend in BACKENDS:
            for signer in ['PHW', 'KIM']:
                ys = [r['cer'] for r in per_clip_all
                      if r['backend'] == backend and r['signer'] == signer]
                xs = list(range(len(ys)))
                if ys:
                    ax.scatter(
                        xs, ys, s=30, alpha=0.75,
                        c=colors.get(backend, 'gray'),
                        marker=markers[signer],
                        label=f'{backend} — {signer}',
                    )
        ax.set_xlabel('clip index (within signer)')
        ax.set_ylabel('per-clip CER')
        ax.set_title('HRNet swap — per-clip CER per (backend, signer)')
        ax.legend(loc='best', fontsize=8, frameon=False)
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'per_clip_scatter.png'), dpi=140)
        plt.close()
        print(f'wrote {os.path.join(args.out_dir, "per_clip_scatter.png")}')
    except Exception as e:
        logger.warning('Scatter plot failed: %s', e)

    # Verdict (per §4).
    baseline = (results.get(('mediapipe_default', 'PHW')) or 0.0,
                results.get(('mediapipe_default', 'KIM')) or 0.0)
    verdict = {
        'baseline_phw': baseline[0],
        'baseline_kim': baseline[1],
        'deltas':       {},
        'classification': None,
    }
    for backend in ['mediapipe_sensitive', 'rtmpose_hand']:
        d_phw = baseline[0] - (results.get((backend, 'PHW')) or baseline[0])
        d_kim = baseline[1] - (results.get((backend, 'KIM')) or baseline[1])
        verdict['deltas'][backend] = {'phw': d_phw, 'kim': d_kim}
    # Classify against §4 thresholds.
    best = max(
        ((b, verdict['deltas'][b]['phw'] + verdict['deltas'][b]['kim'])
         for b in verdict['deltas']),
        key=lambda x: x[1], default=(None, 0.0),
    )[0]
    if best:
        dp = verdict['deltas'][best]['phw']
        dk = verdict['deltas'][best]['kim']
        if dp >= 0.10 and dk >= 0.10:
            verdict['classification'] = 'strong_pass'
        elif (dp >= 0.10 and dk < 0.05) or (dk >= 0.10 and dp < 0.05):
            verdict['classification'] = 'partial_pass'
        elif dp < 0.05 and dk < 0.05:
            verdict['classification'] = 'null'
        elif dp < -0.05 or dk < -0.05:
            verdict['classification'] = 'worse'
        else:
            verdict['classification'] = 'partial_pass'
    with open(os.path.join(args.out_dir, 'verdict.json'), 'w') as f:
        json.dump(verdict, f, indent=2)
    print('\n=== Verdict ===')
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()

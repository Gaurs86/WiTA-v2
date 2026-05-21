#!/usr/bin/env python
"""
scripts/evaluate.py — Standalone evaluation script for WiTA v2.

Loads a saved checkpoint, runs CTC and attention decoding over the
validation set, prints per-sample predictions, and reports summary
CER / WER metrics.

Usage
-----
  # Evaluate best checkpoint (defaults to /kaggle/working/checkpoints/best.pt):
  python scripts/evaluate.py

  # Specify a checkpoint explicitly:
  python scripts/evaluate.py --ckpt /kaggle/working/checkpoints/epoch_040.pt

  # Evaluate on a different language:
  python scripts/evaluate.py --lang korean --ckpt /path/to/korean_best.pt

  # Limit to N validation batches for a quick sanity check:
  python scripts/evaluate.py --max_batches 20

  # Write a JSON report:
  python scripts/evaluate.py --out_json eval_report.json

CLI flags
---------
  All flags that affect data loading or the model architecture must match
  the values used during training (lang, arch, enc_dim, recurrent, etc.).
  The script reads these from the checkpoint's metadata dict where possible,
  falling back to CLI defaults.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

# Make sure the repo root is on the path when called as scripts/evaluate.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WiTA v2 — Standalone Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint
    p.add_argument(
        "--ckpt",
        default=None,
        help="Path to .pt checkpoint. Defaults to <ckpt_dir>/best.pt.",
    )
    p.add_argument("--ckpt_dir", default="/kaggle/working/checkpoints")

    # Data (must match training)
    p.add_argument("--repo",       default="yewon816/WiTA")
    p.add_argument("--lang",       default="english", choices=["english", "korean", "both"])
    p.add_argument("--max_zips",   default=None, type=int,
                   help="Limit ZIPs for a fast smoke-test; None = full dataset.")
    p.add_argument("--img_size",   default=112, type=int)
    p.add_argument("--max_frames", default=64,  type=int)
    p.add_argument("--batch_size", default=8,   type=int)
    p.add_argument("--workers",    default=2,   type=int)
    p.add_argument("--seed",       default=42,  type=int)

    # Model (must match training — ignored if checkpoint contains metadata)
    p.add_argument("--arch",      default="r3d", choices=["r3d","mc3","rmc3","r2plus1d","r2d"])
    p.add_argument("--enc_dim",   default=256,   type=int)
    p.add_argument("--recurrent", default="lstm", choices=["lstm","gru","transformer","none"])
    p.add_argument("--hidden",    default=256,   type=int, dest="hidden_size")
    p.add_argument("--rnn_layers",default=2,     type=int)

    # Evaluation
    p.add_argument("--max_batches", default=None, type=int,
                   help="Limit validation batches; None = full val set.")
    p.add_argument("--decode",      default="both", choices=["ctc", "attn", "both"],
                   help="Which decode head(s) to evaluate.")
    p.add_argument("--show_n",      default=30,  type=int,
                   help="Print this many GT/pred sample rows.")
    p.add_argument("--out_json",    default=None,
                   help="Optional path to write a JSON report.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config builder (mirrors main.py but pared down for eval)
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace, ckpt_meta: dict | None = None):
    """Build a Config, preferring values embedded in the checkpoint."""
    from configs.default import (
        Config, DataConfig, AugConfig, EncoderConfig,
        RecurrentConfig, AttnDecoderConfig, TrainConfig,
    )

    # Pull overrides from checkpoint metadata when available
    lang     = (ckpt_meta or {}).get("lang",         args.lang)
    arch     = (ckpt_meta or {}).get("encoder_arch",  args.arch)
    enc_dim  = (ckpt_meta or {}).get("encoder_dim",   args.enc_dim)

    cfg = Config(
        data=DataConfig(
            hf_repo_id  = args.repo,
            lang        = lang,
            max_zips    = args.max_zips,
            img_size    = args.img_size,
            max_frames  = args.max_frames,
            seed        = args.seed,
        ),
        aug=AugConfig(),
        encoder=EncoderConfig(
            arch    = arch,
            out_dim = enc_dim,
        ),
        recurrent=RecurrentConfig(
            arch        = args.recurrent,
            hidden_size = args.hidden_size,
            num_layers  = args.rnn_layers,
        ),
        attn=AttnDecoderConfig(),
        train=TrainConfig(
            batch_size  = args.batch_size,
            num_workers = args.workers,
            checkpoint_dir = args.ckpt_dir,
            val_limit   = args.max_batches,
            qual_n      = args.show_n,
        ),
    )
    return cfg.build()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mean_cer_wer(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Compute mean CER and mean WER over a list of (gt, pred) string pairs."""
    from datasets.vocab import cer as _cer, wer as _wer
    total_ce = total_cl = total_we = total_wl = 0
    for gt, pred in pairs:
        ce, cl = _cer(gt, pred)
        we, wl = _wer(gt, pred)
        total_ce += min(ce, cl)
        total_cl += cl
        total_we += min(we, wl)
        total_wl += wl
    mean_cer = total_ce / max(total_cl, 1)
    mean_wer = total_we / max(total_wl, 1)
    return mean_cer, mean_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("wita.evaluate")

    # ── Resolve checkpoint path ──────────────────────────────────────────
    ckpt_path = args.ckpt or os.path.join(args.ckpt_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    logger.info("Loading checkpoint: %s", ckpt_path)

    raw_ckpt  = torch.load(ckpt_path, map_location="cpu")
    ckpt_meta = {k: raw_ckpt[k] for k in
                 ("lang", "encoder_arch", "encoder_dim",
                  "ctc_vocab_size", "attn_vocab_size",
                  "val_cer_ctc", "val_cer_attn")
                 if k in raw_ckpt}
    if ckpt_meta:
        logger.info("Checkpoint metadata: %s", ckpt_meta)

    # ── Config & model ───────────────────────────────────────────────────
    cfg   = _build_config(args, ckpt_meta)
    device = cfg.device
    logger.info("Device: %s | lang: %s | arch: %s | recurrent: %s",
                device, cfg.data.lang, cfg.encoder.arch, cfg.recurrent.arch)

    from models.hybrid_model import build_model
    model = build_model(cfg).to(device)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded | params: %s | checkpoint epoch: %s",
                f"{n_params:,}", raw_ckpt.get("epoch", "?"))

    # ── Dataset ──────────────────────────────────────────────────────────
    from datasets.vocab import make_converter
    from datasets.dataset import make_dataloaders, stream_and_index

    converter = make_converter(cfg.data.lang)
    samples   = stream_and_index(cfg)
    _, val_loader = make_dataloaders(cfg, samples, converter)
    logger.info("Val set: %d batches", len(val_loader))

    # ── Evaluate ─────────────────────────────────────────────────────────
    from evaluation.evaluator import evaluate_cer, print_sample_table

    results: dict = {
        "checkpoint":  ckpt_path,
        "epoch":       raw_ckpt.get("epoch"),
        "lang":        cfg.data.lang,
        "encoder":     cfg.encoder.arch,
        "recurrent":   cfg.recurrent.arch,
    }

    run_ctc  = args.decode in ("ctc",  "both")
    run_attn = args.decode in ("attn", "both")

    if run_ctc:
        logger.info("Decoding: CTC greedy …")
        cer_ctc, pairs_ctc = evaluate_cer(
            model, val_loader, converter, cfg,
            decode_mode="ctc", max_batches=args.max_batches,
        )
        wer_ctc, _ = _mean_cer_wer(pairs_ctc)   # WER recomputed from pairs
        _, wer_ctc_val = _mean_cer_wer(pairs_ctc)
        logger.info("CTC  — CER: %.4f  |  WER: %.4f", cer_ctc, wer_ctc_val)
        results["ctc_cer"] = round(cer_ctc,  4)
        results["ctc_wer"] = round(wer_ctc_val, 4)
        results["ctc_pairs"] = pairs_ctc[:args.show_n]

    if run_attn:
        logger.info("Decoding: Attention greedy …")
        cer_attn, pairs_attn = evaluate_cer(
            model, val_loader, converter, cfg,
            decode_mode="attn", max_batches=args.max_batches,
        )
        _, wer_attn_val = _mean_cer_wer(pairs_attn)
        logger.info("Attn — CER: %.4f  |  WER: %.4f", cer_attn, wer_attn_val)
        results["attn_cer"] = round(cer_attn,   4)
        results["attn_wer"] = round(wer_attn_val, 4)
        results["attn_pairs"] = pairs_attn[:args.show_n]

    # ── Sample table ─────────────────────────────────────────────────────
    if run_ctc or run_attn:
        print_sample_table(
            model, val_loader, converter, cfg,
            epoch=raw_ckpt.get("epoch"), max_batches=args.max_batches,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    W = 56
    print(f"\n{'═'*W}")
    print("  Evaluation Summary")
    print(f"{'═'*W}")
    if run_ctc:
        print(f"  CTC  CER : {results['ctc_cer']:.4f}   WER : {results['ctc_wer']:.4f}")
    if run_attn:
        print(f"  Attn CER : {results['attn_cer']:.4f}   WER : {results['attn_wer']:.4f}")
    print(f"{'═'*W}\n")

    # ── Optional JSON report ─────────────────────────────────────────────
    if args.out_json:
        # pairs are (str, str) — JSON-serialisable already
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Report written → %s", args.out_json)


if __name__ == "__main__":
    main()

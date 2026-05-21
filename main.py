"""
main.py — WiTA v2 CLI entry point.

Usage
-----
  # Train with defaults:
  python main.py

  # Override config flags:
  python main.py --lang english --arch r3d --epochs 40 --batch 4 --accum 4

  # Resume:
  python main.py --resume /kaggle/working/checkpoints/latest.pt

  # Evaluate only (no training):
  python main.py --eval_only --resume /kaggle/working/checkpoints/best.pt

  # Smoke test (2 ZIPs, 2 epochs):
  python main.py --max_zips 2 --epochs 2
"""
from __future__ import annotations
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch


def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "run.log")),
        ],
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WiTA v2 — Air-Writing Recognition")

    # Data
    p.add_argument("--repo",      default="yewon816/WiTA",  help="HuggingFace dataset repo")
    p.add_argument("--lang",      default="english",        choices=["english", "korean", "both"])
    p.add_argument("--max_zips",  default=None, type=int,   help="Limit number of ZIPs (debug)")
    p.add_argument("--img_size",  default=112,  type=int)
    p.add_argument("--max_frames",default=64,   type=int)

    # Encoder
    p.add_argument("--arch",      default="r3d",  choices=["r3d","mc3","rmc3","r2plus1d","r2d"])
    p.add_argument("--layers",    default=1, type=int, dest="num_res_layers")
    p.add_argument("--enc_dim",   default=256, type=int)
    p.add_argument("--pooling",   default="average", choices=["average","max"])
    p.add_argument("--pretrained",action="store_true")

    # Recurrent head
    p.add_argument("--recurrent", default="lstm", choices=["lstm","gru","transformer","none"])
    p.add_argument("--hidden",    default=256, type=int, dest="hidden_size")
    p.add_argument("--rnn_layers",default=2,   type=int)

    # Training
    p.add_argument("--epochs",    default=40,   type=int)
    p.add_argument("--batch",     default=4,    type=int, dest="batch_size")
    p.add_argument("--accum",     default=4,    type=int, dest="accum_steps")
    p.add_argument("--lr",        default=3e-4, type=float)
    p.add_argument("--workers",   default=2,    type=int)
    p.add_argument("--optimizer", default="adamw",choices=["adamw","adam","sgd","rmsprop"])
    p.add_argument("--scheduler", default="onecycle",choices=["onecycle","warmup_multistep","steplr","none"])
    p.add_argument("--val_limit", default=50,   type=int, help="Max val batches per pass (0=full)")
    p.add_argument("--seed",      default=42,   type=int)

    # Loss
    p.add_argument("--lambda_start", default=0.5, type=float)
    p.add_argument("--lambda_min",   default=0.2, type=float)

    # Checkpointing
    p.add_argument("--ckpt_dir",  default="/kaggle/working/checkpoints")
    p.add_argument("--resume",    default=None, type=str)
    p.add_argument("--save_freq", default=5,    type=int)

    # Mode
    p.add_argument("--eval_only", action="store_true", help="Skip training; eval best.pt")
    p.add_argument("--export",    action="store_true", help="Export best.pt after training")

    return p.parse_args()


def build_config(args: argparse.Namespace):
    """Translate CLI args → Config dataclass."""
    from configs.default import (Config, DataConfig, AugConfig, EncoderConfig,
                                  RecurrentConfig, AttnDecoderConfig, TrainConfig)

    cfg = Config(
        data=DataConfig(
            hf_repo_id  = args.repo,
            lang        = args.lang,
            max_zips    = args.max_zips,
            img_size    = args.img_size,
            max_frames  = args.max_frames,
            seed        = args.seed,
        ),
        aug=AugConfig(),     # augmentation stays at sensible defaults
        encoder=EncoderConfig(
            arch            = args.arch,
            num_res_layers  = args.num_res_layers,
            out_dim         = args.enc_dim,
            pooling         = args.pooling,
            pretrained      = args.pretrained,
        ),
        recurrent=RecurrentConfig(
            arch        = args.recurrent,
            hidden_size = args.hidden_size,
            num_layers  = args.rnn_layers,
        ),
        attn=AttnDecoderConfig(),
        train=TrainConfig(
            num_epochs      = args.epochs,
            batch_size      = args.batch_size,
            accum_steps     = args.accum_steps,
            lr              = args.lr,
            num_workers     = args.workers,
            optimizer       = args.optimizer,
            scheduler       = args.scheduler,
            val_limit       = args.val_limit if args.val_limit > 0 else None,
            seed            = args.seed,
            lambda_ctc_start= args.lambda_start,
            lambda_ctc_min  = args.lambda_min,
            checkpoint_dir  = args.ckpt_dir,
            resume_path     = args.resume,
            save_frequency  = args.save_freq,
        ),
    )
    return cfg.build()


def main() -> None:
    args = parse_args()
    from configs.default import Config
    cfg  = build_config(args)

    log_dir = os.path.join(args.ckpt_dir, "..", "logs")
    _setup_logging(log_dir)
    _seed_everything(cfg.train.seed)

    logger = logging.getLogger("wita.main")
    logger.info("WiTA v2  |  device=%s  lang=%s  arch=%s  recurrent=%s",
                cfg.device, cfg.data.lang, cfg.encoder.arch, cfg.recurrent.arch)
    logger.info("CTC vocab=%d  Attn vocab=%d", cfg.vocab.ctc_vocab_size, cfg.vocab.attn_vocab_size)

    # ── Dataset ──────────────────────────────────────────────────────────
    from datasets.vocab import make_converter
    from datasets.dataset import make_dataloaders, stream_and_index

    converter = make_converter(cfg.data.lang)

    if not args.eval_only:
        samples = stream_and_index(cfg)
    else:
        # Eval-only: just build val set from a small sample
        logger.info("Eval-only mode: loading a minimal val set for sanity check.")
        from datasets.dataset import WiTADataset
        cfg.data.max_zips = 1
        samples = stream_and_index(cfg)

    train_loader, val_loader = make_dataloaders(cfg, samples, converter)

    # ── Model ────────────────────────────────────────────────────────────
    from models.hybrid_model import build_model
    model = build_model(cfg)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model built  |  trainable params: %s", f"{n_p:,}")

    # ── Train ────────────────────────────────────────────────────────────
    if not args.eval_only:
        from training.trainer import train
        best_model = train(model, train_loader, val_loader, converter, cfg)
    else:
        best_model = model
        resume = args.resume or os.path.join(args.ckpt_dir, "best.pt")
        if os.path.isfile(resume):
            ckpt = torch.load(resume, map_location=cfg.device)
            best_model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded weights from %s", resume)
        else:
            logger.warning("No checkpoint found at %s", resume)

    # ── Final evaluation ─────────────────────────────────────────────────
    from evaluation.evaluator import evaluate_cer, print_sample_table
    logger.info("=== Final Evaluation (full val set) ===")

    cer_ctc, _ = evaluate_cer(
        best_model, val_loader, converter, cfg, decode_mode="ctc", max_batches=None)
    cer_attn, _ = evaluate_cer(
        best_model, val_loader, converter, cfg, decode_mode="attn", max_batches=None)

    logger.info("Final CER  CTC  : %.4f", cer_ctc)
    logger.info("Final CER  Attn : %.4f", cer_attn)
    print_sample_table(best_model, val_loader, converter, cfg, epoch=args.epochs, max_batches=None)

    # ── Export ───────────────────────────────────────────────────────────
    if args.export or not args.eval_only:
        export_path = os.path.join(args.ckpt_dir, "phase1_export.pt")
        torch.save({
            "model_state_dict":  best_model.state_dict(),
            "ctc_vocab_size":    cfg.vocab.ctc_vocab_size,
            "attn_vocab_size":   cfg.vocab.attn_vocab_size,
            "lang":              cfg.data.lang,
            "encoder_arch":      cfg.encoder.arch,
            "encoder_dim":       cfg.encoder.out_dim,
            "val_cer_ctc":       cer_ctc,
            "val_cer_attn":      cer_attn,
        }, export_path)
        sz = os.path.getsize(export_path) / 1e6
        logger.info("Exported → %s  (%.1f MB)", export_path, sz)


if __name__ == "__main__":
    main()

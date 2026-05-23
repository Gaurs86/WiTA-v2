"""
training/trainer.py — WiTA v2 training loop.

FIX (Bug 3 — AMP scope correction)
-------------------------------------
The original code called `model(...)` OUTSIDE any autocast context (relying
on the model's own internal autocast), then wrapped ONLY the loss in a
second `with autocast(...)` block.  This broke the intended single-scope AMP
flow.

Fix: a single `with autocast(device_type=..., enabled=_use_amp)` now wraps
BOTH the model forward pass AND the loss computation.  The model's internal
autocast has been removed (see hybrid_model.py).  This matches the canonical
PyTorch AMP recipe and is safe for DataParallel.

Additional note on scheduler stepping
--------------------------------------
OneCycleLR.step() must be called exactly once per optimiser step.  The
original code skipped scheduler.step() whenever the AMP scaler reduced its
scale (i.e. when Inf/NaN was detected).  This is correct behaviour — when
the optimiser step is skipped due to Inf gradients, the scheduler should
not advance either.  This logic is preserved unchanged.
"""
from __future__ import annotations
import os, time, logging
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

_use_amp   = torch.cuda.is_available()
_amp_device = "cuda" if _use_amp else "cpu"


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m

from torch.utils.data import DataLoader

from ..configs.default import Config
from ..datasets.vocab import StrLabelConverter
from .losses import hybrid_loss, prepare_attn_targets, get_lambda_ctc
from .schedulers import build_optimizer, build_scheduler
from ..evaluation.evaluator import evaluate_cer, print_sample_table
from ..utils.checkpoint import save as ckpt_save, load as ckpt_load

logger = logging.getLogger(__name__)


def _train_epoch(
    model, loader: DataLoader, optimizer, scheduler, scaler,
    lambda_ctc: float, cfg: Config, epoch: int,
) -> tuple[float, float, float]:
    model.train()
    total_sum = ctc_sum = attn_sum = 0.0
    n = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        clips, labels, input_lens, label_lens = batch
        clips      = clips.to(cfg.device, non_blocking=True)
        labels     = labels.to(cfg.device, non_blocking=True)
        input_lens = input_lens.to(cfg.device)
        label_lens = label_lens.to(cfg.device)

        tgt_in, tgt_out, tgt_pad = prepare_attn_targets(labels, label_lens, cfg.vocab)

        # ── Single autocast scope: wraps BOTH model forward and loss ──────
        # This is the canonical PyTorch AMP pattern.  The model's own
        # autocast wrapper has been removed (see hybrid_model.py) so that
        # all fp16 operations live inside this single scope.
        with autocast(device_type=_amp_device, enabled=_use_amp):
            ctc_lp, attn_logits = model(clips, input_lens, tgt_in, tgt_pad)
            loss, ctc_v, attn_v = hybrid_loss(
                ctc_lp, attn_logits, labels, tgt_out,
                input_lens, label_lens, lambda_ctc, cfg.vocab, cfg.train,
            )

        loss = loss / cfg.train.accum_steps
        scaler.scale(loss).backward()

        total_sum += loss.item() * cfg.train.accum_steps
        ctc_sum   += ctc_v
        attn_sum  += attn_v
        n         += 1

        is_boundary = (step + 1) % cfg.train.accum_steps == 0
        is_last     = (step + 1) == len(loader)
        if is_boundary or is_last:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            _scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step the scheduler when the optimiser actually stepped
            # (i.e. no Inf/NaN caused the scaler to skip the update).
            if scheduler and scaler.get_scale() >= _scale_before:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.train.log_interval == 0:
            logger.debug("Ep%d step%d/%d loss=%.4f ctc=%.4f attn=%.4f (%.0fs)",
                         epoch, step, len(loader),
                         loss.item() * cfg.train.accum_steps,
                         ctc_v, attn_v, time.time() - t0)

    dv = max(n, 1)
    return total_sum / dv, ctc_sum / dv, attn_sum / dv


def train(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    converter:      StrLabelConverter,
    cfg:            Config,
) -> nn.Module:
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    model = model.to(cfg.device)
    if torch.cuda.device_count() > 1:
        logger.info("DataParallel: %d GPUs detected — wrapping model",
                    torch.cuda.device_count())
        model = nn.DataParallel(model)

    total_steps = cfg.train.num_epochs * max(1, len(train_loader) // cfg.train.accum_steps)
    optimizer   = build_optimizer(model.parameters(), cfg.train)
    scheduler   = build_scheduler(optimizer, total_steps, cfg.train)
    scaler      = GradScaler(device=_amp_device, enabled=_use_amp)

    start_epoch = 0
    best_cer    = float("inf")

    resume = cfg.train.resume_path
    if resume and os.path.isfile(resume):
        start_epoch, best_cer = ckpt_load(
            resume, _unwrap(model), optimizer, scheduler, scaler, cfg.device)
    elif resume:
        logger.warning("RESUME_PATH '%s' not found — starting from scratch.", resume)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        lambda_ctc = get_lambda_ctc(epoch, cfg.train.num_epochs, cfg.train)
        t0 = time.time()

        avg_total, avg_ctc, avg_attn = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            lambda_ctc, cfg, epoch)

        val_cer_ctc, _ = evaluate_cer(
            model, val_loader, converter, cfg,
            decode_mode="ctc", max_batches=cfg.train.val_limit,
        )

        val_cer_attn = None
        run_attn = (epoch + 1) % cfg.train.qual_every_n == 0
        if run_attn:
            val_cer_attn, _ = evaluate_cer(
                model, val_loader, converter, cfg,
                decode_mode="attn", max_batches=cfg.train.val_limit,
            )
            print_sample_table(model, val_loader, converter, cfg,
                                epoch=epoch + 1, max_batches=cfg.train.val_limit)

        lr    = scheduler.get_last_lr()[0] if scheduler else cfg.train.lr
        gmb   = torch.cuda.memory_allocated(cfg.device) / 1e6 if torch.cuda.is_available() else 0
        attn_s = f"{val_cer_attn:.4f}" if val_cer_attn is not None else "  ——  "

        line = (f"[Ep {epoch+1:3d}/{cfg.train.num_epochs}] "
                f"loss={avg_total:.4f}(ctc={avg_ctc:.4f} attn={avg_attn:.4f}) | "
                f"val_cer_ctc={val_cer_ctc:.4f} attn={attn_s} | "
                f"λ={lambda_ctc:.3f} lr={lr:.2e} GPU={gmb:.0f}MB {time.time()-t0:.0f}s")
        print(line); logger.info(line)

        latest = os.path.join(cfg.train.checkpoint_dir, "latest.pt")
        ckpt_save(latest, _unwrap(model), optimizer, scheduler, scaler,
                  epoch, best_cer, val_cer_ctc)

        if val_cer_ctc < best_cer:
            best_cer = val_cer_ctc
            best = os.path.join(cfg.train.checkpoint_dir, "best.pt")
            ckpt_save(best, _unwrap(model), optimizer, scheduler, scaler,
                      epoch, best_cer, val_cer_ctc)
            logger.info("★ New best CER=%.4f → %s", best_cer, best)

        if (epoch + 1) % cfg.train.save_frequency == 0:
            p = os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch+1:03d}.pt")
            ckpt_save(p, _unwrap(model), optimizer, scheduler, scaler,
                      epoch, best_cer, val_cer_ctc)

    best = os.path.join(cfg.train.checkpoint_dir, "best.pt")
    if os.path.isfile(best):
        ckpt = torch.load(best, map_location=cfg.device)
        _unwrap(model).load_state_dict(ckpt["model_state_dict"])
        logger.info("Best model loaded (CER=%.4f)", ckpt.get("val_cer", best_cer))

    return model

"""
training/trainer.py — WiTA v2 training loop.

Features
--------
  Mixed precision AMP          (T4 memory bandwidth halved)
  Gradient accumulation        (effective batch = batch_size × accum_steps)
  Lambda annealing             (CTC weight 0.5 → 0.2)
  Fast validation (VAL_LIMIT)  (partial pass every epoch)
  Robust checkpointing         (latest.pt + best.pt, all 5 state-dicts)
  Resume from checkpoint       (exact epoch, optimizer, scheduler, scaler)
  Qualitative sample table     (every N epochs)
"""
from __future__ import annotations
import os, time, logging
import torch
import torch.nn as nn
import torch.amp as _amp
from torch.amp import autocast as _autocast_cls

# Compatibility shim: pick the right AMP device at runtime
def _make_scaler(device):
    """Return a GradScaler appropriate for the device (no-op on CPU)."""
    if hasattr(device, 'type'):
        dtype = device.type
    else:
        dtype = str(device).split(':')[0]   # 'cuda', 'cpu', 'mps'
    return _amp.GradScaler(dtype, enabled=(dtype == 'cuda'))

def _autocast(device):
    """Context manager for AMP; no-op on CPU."""
    if hasattr(device, 'type'):
        dtype = device.type
    else:
        dtype = str(device).split(':')[0]
    return _autocast_cls(device_type=dtype, enabled=(dtype == 'cuda'))
from torch.utils.data import DataLoader

from ..configs.default import Config
from ..datasets.vocab import StrLabelConverter
from .losses import hybrid_loss, prepare_attn_targets, get_lambda_ctc
from .schedulers import build_optimizer, build_scheduler
from ..evaluation.evaluator import evaluate_cer, print_sample_table
from ..utils.checkpoint import save as ckpt_save, load as ckpt_load

logger = logging.getLogger(__name__)


def _train_epoch(
    model, loader: DataLoader, optimizer, scheduler, scaler: GradScaler,
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

        with _autocast(cfg.device):
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
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.train.log_interval == 0:
            logger.debug("Ep%d step%d/%d loss=%.4f ctc=%.4f attn=%.4f (%.0fs)",
                         epoch, step, len(loader), loss.item() * cfg.train.accum_steps,
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
    """
    Full training loop. Returns the best-checkpoint model (weights loaded in-place).
    """
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    model = model.to(cfg.device)

    total_steps = cfg.train.num_epochs * max(1, len(train_loader) // cfg.train.accum_steps)
    optimizer   = build_optimizer(model.parameters(), cfg.train)
    scheduler   = build_scheduler(optimizer, total_steps, cfg.train)
    scaler      = _make_scaler(cfg.device)

    start_epoch = 0
    best_cer    = float("inf")

    resume = cfg.train.resume_path
    if resume and os.path.isfile(resume):
        start_epoch, best_cer = ckpt_load(resume, model, optimizer, scheduler, scaler, cfg.device)
    elif resume:
        logger.warning("RESUME_PATH '%s' not found — starting from scratch.", resume)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        lambda_ctc = get_lambda_ctc(epoch, cfg.train.num_epochs, cfg.train)
        t0 = time.time()

        avg_total, avg_ctc, avg_attn = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, lambda_ctc, cfg, epoch)

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

        # Save latest every epoch
        latest = os.path.join(cfg.train.checkpoint_dir, "latest.pt")
        ckpt_save(latest, model, optimizer, scheduler, scaler, epoch, best_cer, val_cer_ctc)

        # Save best
        if val_cer_ctc < best_cer:
            best_cer = val_cer_ctc
            best = os.path.join(cfg.train.checkpoint_dir, "best.pt")
            ckpt_save(best, model, optimizer, scheduler, scaler, epoch, best_cer, val_cer_ctc)
            logger.info("★ New best CER=%.4f → %s", best_cer, best)

        # Periodic save
        if (epoch + 1) % cfg.train.save_frequency == 0:
            p = os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch+1:03d}.pt")
            ckpt_save(p, model, optimizer, scheduler, scaler, epoch, best_cer, val_cer_ctc)

    # Reload best
    best = os.path.join(cfg.train.checkpoint_dir, "best.pt")
    if os.path.isfile(best):
        ckpt = torch.load(best, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Best model loaded (CER=%.4f)", ckpt.get("val_cer", best_cer))

    return model

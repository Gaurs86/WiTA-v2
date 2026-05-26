"""
training/trainer.py — WiTA v2 CTC-only training loop.

Refactored from hybrid CTC + Attention trainer.

Removed
-------
• prepare_attn_targets()    call  (attention-only)
• hybrid_loss()             call  → replaced with ctc_loss()
• get_lambda_ctc()          call  (lambda annealing, attention-only)
• attn_sum / avg_attn       tracking
• val_cer_attn evaluation
• lambda (λ) in log line
• decode_mode="attn" evaluation path

Kept / added
-------------
• Single autocast scope wrapping model forward + loss (AMP fix preserved)
• Gradient clipping  (cfg.train.grad_clip)
• Mixed precision    (GradScaler)
• Scheduler support  (OneCycleLR / warmup_multistep / steplr)
• Gradient accumulation (cfg.train.accum_steps)
• Backbone unfreeze schedule (cfg.train.unfreeze_after_epoch)
• Checkpoint save/load (latest.pt, best.pt, epoch_NNN.pt)
• DataParallel support

AMP note
--------
A single `with autocast(...)` scope wraps BOTH the model forward pass AND
the loss computation.  This is the canonical PyTorch AMP recipe.  The model
has no internal autocast.

Scheduler stepping
------------------
OneCycleLR.step() is called only when the optimizer actually stepped
(i.e. scale did not decrease due to Inf/NaN gradients).  This prevents
OneCycleLR from advancing its internal step counter without a corresponding
optimizer step, which would raise ValueError after all budget steps are used.
"""
from __future__ import annotations
import os, time, logging
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..configs.default import Config
from ..datasets.vocab import StrLabelConverter
from .losses import ctc_loss
from .schedulers import build_optimizer, build_scheduler
from ..evaluation.evaluator import evaluate_cer, print_sample_table
from ..utils.checkpoint import save as ckpt_save, load as ckpt_load

logger = logging.getLogger(__name__)

_use_amp    = torch.cuda.is_available()
_amp_device = "cuda" if _use_amp else "cpu"


def _unwrap(m: nn.Module) -> nn.Module:
    """Unwrap DataParallel to access the underlying model."""
    return m.module if isinstance(m, nn.DataParallel) else m


def _build_param_groups(model: nn.Module, cfg: Config) -> list[dict]:
    """
    Split parameters into backbone vs head groups for discriminative LR.

    Backbone group: encoder.backbone.* (the pretrained VideoMAE / Video Swin).
    Head group    : everything else (recurrent head, CTC projection, encoder
                    projection if any).

    The backbone group's "lr" is cfg.train.lr * cfg.train.backbone_lr_mult.
    Both groups carry weight_decay = cfg.train.weight_decay.

    Note: parameters are included in the optimizer even when requires_grad is
    False at startup. AdamW skips updates for params whose .grad is None, so
    the frozen-warmup phase is unaffected. Once unfreeze_backbone() flips
    requires_grad, gradients flow and the backbone group steps at its lower
    LR — the schedule never needs to know about the unfreeze event.
    """
    inner = _unwrap(model)
    backbone_params: list = []
    head_params:     list = []

    encoder = getattr(inner, "encoder", None)
    backbone = getattr(encoder, "backbone", None) if encoder is not None else None
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()

    for p in inner.parameters():
        (backbone_params if id(p) in backbone_ids else head_params).append(p)

    head_lr     = cfg.train.lr
    backbone_lr = cfg.train.lr * cfg.train.backbone_lr_mult

    groups = [{"params": head_params, "lr": head_lr, "name": "head"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    return groups


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    cfg:       Config,
    epoch:     int,
) -> float:
    """
    Run one training epoch.

    Returns
    -------
    avg_ctc_loss : float  — mean CTC loss over all steps
    """
    model.train()
    total_sum = 0.0
    n         = 0
    t0        = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        clips, labels, input_lens, label_lens = batch
        clips      = clips.to(cfg.device, non_blocking=True)
        labels     = labels.to(cfg.device, non_blocking=True)
        input_lens = input_lens.to(cfg.device)
        label_lens = label_lens.to(cfg.device)

        # ── Single autocast scope: model forward + loss ───────────────────
        # Canonical PyTorch AMP pattern — do NOT split across two scopes.
        with autocast(device_type=_amp_device, enabled=_use_amp):
            # model returns (ctc_log_probs [B,T,V], enc_lens [B])
            ctc_lp, enc_lens = model(clips, input_lens)

            loss = ctc_loss(
                ctc_log_probs  = ctc_lp,
                targets        = labels,
                input_lengths  = enc_lens,    # ← scaled encoded lengths
                target_lengths = label_lens,
                vocab          = cfg.vocab,
            )

        # Gradient accumulation
        loss = loss / cfg.train.accum_steps
        scaler.scale(loss).backward()

        total_sum += loss.item() * cfg.train.accum_steps
        n         += 1

        is_boundary = (step + 1) % cfg.train.accum_steps == 0
        is_last     = (step + 1) == len(loader)

        if is_boundary or is_last:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            _scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Only step scheduler when the optimizer actually stepped.
            # If scaler reduced scale (Inf/NaN), both optimizer and
            # scheduler steps are skipped to keep them in sync.
            if scheduler and scaler.get_scale() >= _scale_before:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        if step % cfg.train.log_interval == 0:
            logger.debug(
                "Ep%d step%d/%d ctc_loss=%.4f (%.0fs)",
                epoch, step, len(loader),
                loss.item() * cfg.train.accum_steps,
                time.time() - t0,
            )

    return total_sum / max(n, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    converter:    StrLabelConverter,
    cfg:          Config,
) -> nn.Module:
    """
    Full training loop: epochs, evaluation, checkpointing.

    Backbone unfreezing
    -------------------
    If cfg.train.unfreeze_after_epoch > 0, the visual backbone is frozen
    during the first N epochs so the CTC head trains in isolation.  After
    epoch N, unfreeze_backbone() is called to enable end-to-end fine-tuning.
    This stabilises early training significantly with large pretrained models.

    Checkpoint files
    ----------------
    latest.pt          — saved every epoch
    best.pt            — saved when val CER improves
    epoch_NNN.pt       — saved every cfg.train.save_frequency epochs
    """
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    model = model.to(cfg.device)

    if torch.cuda.device_count() > 1 and cfg.train.batch_size >= torch.cuda.device_count():
        logger.info("DataParallel: %d GPUs — wrapping model",
                    torch.cuda.device_count())
        model = nn.DataParallel(model)

    # Steps-per-epoch for OneCycleLR budget calculation
    _n          = len(train_loader)
    _full       = _n // cfg.train.accum_steps
    _remainder  = int(_n % cfg.train.accum_steps != 0)
    steps_per_epoch = max(1, _full + _remainder)
    total_steps     = cfg.train.num_epochs * steps_per_epoch

    # Discriminative LR: backbone gets cfg.train.backbone_lr_mult × head LR.
    # build_scheduler will read each group's "lr" for OneCycleLR.max_lr.
    param_groups = _build_param_groups(model, cfg)
    optimizer    = build_optimizer(param_groups, cfg.train)
    scheduler    = build_scheduler(optimizer, total_steps, cfg.train)
    logger.info(
        "Optimizer param groups: %s",
        ", ".join(f"{g.get('name','?')}(lr={g['lr']:.2e}, n={len(g['params'])})"
                  for g in optimizer.param_groups),
    )
    scaler    = GradScaler(device=_amp_device, enabled=_use_amp)

    start_epoch = 0
    best_cer    = float("inf")

    # ── Resume from checkpoint ────────────────────────────────────────────
    resume = cfg.train.resume_path
    if resume and os.path.isfile(resume):
        start_epoch, best_cer = ckpt_load(
            resume, _unwrap(model), optimizer, scheduler, scaler, cfg.device
        )
        logger.info("Resumed from %s (epoch %d, best_cer=%.4f)",
                    resume, start_epoch, best_cer)
    elif resume:
        logger.warning("resume_path '%s' not found — starting from scratch.", resume)

    # ── Training epochs ───────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.train.num_epochs):

        # ── Backbone unfreezing schedule ─────────────────────────────────
        unfreeze_at = cfg.train.unfreeze_after_epoch
        if unfreeze_at > 0 and epoch == unfreeze_at:
            logger.info("Epoch %d: unfreezing backbone for end-to-end fine-tuning",
                        epoch)
            _unwrap(model).unfreeze_backbone()

        t0 = time.time()

        avg_ctc = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, epoch
        )

        # ── Validation ───────────────────────────────────────────────────
        val_cer, _ = evaluate_cer(
            model, val_loader, converter, cfg,
            decode_mode="ctc",
            max_batches=cfg.train.val_limit,
        )

        # Qualitative sample table every N epochs
        if (epoch + 1) % cfg.train.qual_every_n == 0:
            print_sample_table(
                model, val_loader, converter, cfg,
                epoch=epoch + 1,
                max_batches=cfg.train.val_limit,
            )

        lr  = scheduler.get_last_lr()[0] if scheduler else cfg.train.lr
        gmb = (
            torch.cuda.memory_allocated(cfg.device) / 1e6
            if torch.cuda.is_available() else 0
        )
        frozen = (
            "frozen" if epoch < cfg.train.unfreeze_after_epoch else "unfrozen"
        )

        line = (
            f"[Ep {epoch+1:3d}/{cfg.train.num_epochs}] "
            f"ctc_loss={avg_ctc:.4f} | "
            f"val_cer={val_cer:.4f} | "
            f"lr={lr:.2e} backbone={frozen} "
            f"GPU={gmb:.0f}MB {time.time()-t0:.0f}s"
        )
        print(line)
        logger.info(line)

        # ── Checkpointing ─────────────────────────────────────────────────
        latest = os.path.join(cfg.train.checkpoint_dir, "latest.pt")
        ckpt_save(latest, _unwrap(model), optimizer, scheduler, scaler,
                  epoch, best_cer, val_cer)

        if val_cer < best_cer:
            best_cer = val_cer
            best     = os.path.join(cfg.train.checkpoint_dir, "best.pt")
            ckpt_save(best, _unwrap(model), optimizer, scheduler, scaler,
                      epoch, best_cer, val_cer)
            logger.info("★ New best CER=%.4f → %s", best_cer, best)

        if (epoch + 1) % cfg.train.save_frequency == 0:
            p = os.path.join(cfg.train.checkpoint_dir,
                             f"epoch_{epoch+1:03d}.pt")
            ckpt_save(p, _unwrap(model), optimizer, scheduler, scaler,
                      epoch, best_cer, val_cer)

    # ── Load best weights before returning ───────────────────────────────
    best = os.path.join(cfg.train.checkpoint_dir, "best.pt")
    if os.path.isfile(best):
        ckpt = torch.load(best, map_location=cfg.device, weights_only=True)
        _unwrap(model).load_state_dict(ckpt["model_state_dict"])
        logger.info("Best model loaded (CER=%.4f)",
                    ckpt.get("val_cer", best_cer))

    return model

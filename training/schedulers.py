"""
training/schedulers.py — Learning rate schedulers.
Re-implements WarmupMultiStepLR from WiTA baseline utils.py.
"""
from __future__ import annotations
import torch
from torch.optim.lr_scheduler import _LRScheduler
from ..configs.default import TrainConfig


class WarmupMultiStepLR(_LRScheduler):
    """Warmup then multi-step decay. Mirrors WiTA baseline exactly."""
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_iters=500, last_epoch=-1):
        self.milestones   = sorted(milestones)
        self.gamma        = gamma
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            alpha = step / max(self.warmup_iters, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        factor = self.gamma ** sum(step >= m for m in self.milestones)
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(optimizer, total_steps: int, cfg: TrainConfig):
    """Return the configured scheduler (or None).

    For OneCycleLR with multiple param groups, max_lr is a list — one entry
    per group, taken from each group's "lr" key (set in build_optimizer).
    """
    stype = cfg.scheduler.lower()
    if stype == "onecycle":
        max_lr_per_group = [g["lr"] for g in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr_per_group if len(max_lr_per_group) > 1 else cfg.lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_pct, anneal_strategy="cos",
            final_div_factor=cfg.final_div_factor,
        )
    if stype == "warmup_multistep":
        iter_size = total_steps // cfg.num_epochs
        milestones = [iter_size * cfg.scheduler_step * (i + 1)
                      for i in range(cfg.num_epochs)]
        return WarmupMultiStepLR(optimizer, milestones,
                                 gamma=cfg.scheduler_gamma, warmup_iters=500)
    if stype == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=total_steps // cfg.num_epochs * cfg.scheduler_step,
            gamma=cfg.scheduler_gamma)
    return None  # "none"


def build_optimizer(params, cfg: TrainConfig):
    """Return configured optimizer."""
    otype = cfg.optimizer.lower()
    kw = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if otype == "adamw":
        return torch.optim.AdamW(params, betas=(cfg.beta1, cfg.beta2), **kw)
    if otype == "adam":
        return torch.optim.Adam(params, betas=(cfg.beta1, cfg.beta2), **kw)
    if otype == "sgd":
        return torch.optim.SGD(params, momentum=0.9, **kw)
    if otype == "rmsprop":
        return torch.optim.RMSprop(params, **kw)
    raise ValueError(f"Unknown optimizer '{otype}'")

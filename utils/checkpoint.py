"""utils/checkpoint.py — Checkpoint save / load."""
from __future__ import annotations
import os, logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save(path: str, model: nn.Module, optimizer, scheduler, scaler,
         epoch: int, best_cer: float, val_cer: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":   scaler.state_dict(),
        "best_cer":            best_cer,
        "val_cer":             val_cer,
    }, path)
    logger.info("Checkpoint saved → %s", path)


def load(path: str, model: nn.Module, optimizer, scheduler, scaler,
         device: torch.device) -> tuple[int, float]:
    """Returns (start_epoch, best_cer)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start = ckpt["epoch"] + 1
    best  = ckpt.get("best_cer", float("inf"))
    logger.info("Resumed from %s  (epoch %d → %d, best_cer=%.4f)",
                path, ckpt["epoch"], start, best)
    return start, best

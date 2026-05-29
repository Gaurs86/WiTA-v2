"""
models/grl.py — Gradient Reversal Layer (Ganin et al., 2016).

Single-purpose module: identity forward, gradient negated and scaled by λ on
backward.  Used by the signer-adversarial Stage 1 v3 head to make the
Conformer encoder produce signer-invariant features.

Reference
---------
Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation,"
ICML 2015 / Ganin et al., "Domain-Adversarial Training of Neural Networks,"
JMLR 2016.  Schedule used downstream:
    λ(p) = (2 / (1 + exp(-γ p))) − 1,   γ = 10,   p ∈ [0, 1]
Implemented in `lambda_schedule()` below.
"""

from __future__ import annotations

import math
import torch
from torch.autograd import Function


class _GradientReversal(Function):
    """Inner autograd Function — identity forward, -λ * grad backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Multiply by -λ on the way back.  The second return is for `lambda_`,
        # which is a python float, so no gradient.
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Functional wrapper.  Use as:
        x_rev = gradient_reversal(x, lambda_=current_lambda)
        # x_rev is identical to x on forward, but grad is negated * lambda_.
    """
    return _GradientReversal.apply(x, lambda_)


def lambda_schedule(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    Ganin schedule: λ(p) = 2 / (1 + exp(-γ p)) − 1,   p = step / total_steps.

    Smoothly ramps λ from 0 at step=0 to ≈1 at step=total_steps.  Allows the
    CTC objective to dominate during early warm-up, then phases in the
    adversarial signal as training progresses.
    """
    if total_steps <= 0:
        return 1.0
    p = max(0.0, min(1.0, step / total_steps))
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

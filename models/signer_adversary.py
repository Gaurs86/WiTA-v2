"""
models/signer_adversary.py — DANN-style signer-adversarial head.

Architecture (Stage 1 v3 plan §3.2):
    conformer_out [B, T, d_model]
        → mean over T  → [B, d_model]
        → GRL(λ)       → [B, d_model]
        → Linear(d_model → 128) → GELU → Dropout(0.3)
        → Linear(128 → n_signers) → logits
        → CrossEntropy(signer_target)

Total loss in the trainer: L = L_ctc + α · L_signer_adv

The GRL multiplies the gradient by −λ on the way back into the encoder, so
minimising the standard CE loss on the signer logits is equivalent to the
encoder MAXIMISING signer confusion while the signer head still MINIMISES
its own CE.  λ is ramped via lambda_schedule() in models/grl.py.

Why mean-pool the encoder output instead of per-timestep classification:
  * the discriminator gets a single fixed-shape vector regardless of T
  * fewer parameters and a sharper "writer fingerprint" the encoder must scrub
  * matches the original Ganin et al. recipe for image classifiers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import gradient_reversal


class SignerAdversary(nn.Module):
    """
    Mean-pool → GRL → MLP → signer logits.

    Parameters
    ----------
    d_model   : encoder hidden dim (256 for Stage 1 v2 Conformer)
    n_signers : number of training-fold signers (output dim of the head)
    hidden    : MLP hidden width (128 per plan)
    dropout   : applied between the two MLP layers
    """

    def __init__(
        self,
        d_model:   int,
        n_signers: int,
        hidden:    int = 128,
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.n_signers = n_signers
        self.fc1 = nn.Linear(d_model, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_signers)

    def forward(
        self,
        encoder_out: torch.Tensor,
        lambda_:     float,
        mask:        torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        encoder_out : [B, T, d_model] — pre-upsample Conformer output
        lambda_     : current GRL coefficient (set by Ganin schedule)
        mask        : optional [B, T] bool, True at PAD positions; if given,
                      we mean over non-PAD positions only.
        returns     : [B, n_signers] logits
        """
        if encoder_out.dim() != 3:
            raise ValueError(
                f"expected [B, T, d_model], got {tuple(encoder_out.shape)}"
            )
        if mask is None:
            pooled = encoder_out.mean(dim=1)
        else:
            # mask=True at pad, so weight by ~mask
            keep = (~mask).float().unsqueeze(-1)        # [B, T, 1]
            n = keep.sum(dim=1).clamp(min=1.0)          # [B, 1]
            pooled = (encoder_out * keep).sum(dim=1) / n

        x = gradient_reversal(pooled, lambda_=lambda_)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

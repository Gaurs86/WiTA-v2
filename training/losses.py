"""
training/losses.py — CTC-only loss.

Refactored from hybrid CTC + Attention loss to CTC-only.

Removed
-------
• hybrid_loss()         — replaced by ctc_loss()
• get_lambda_ctc()      — lambda annealing (attention-only concept)
• prepare_attn_targets()— teacher-forcing target builder (attention-only)
• _ce_fn()              — cross-entropy cache (attention-only)

Kept
----
• _ctc_fn()             — singleton CTCLoss cache
• ctc_loss()            — clean single-function CTC loss computation

Tensor convention reminder
--------------------------
  Model output  : [B, T, V]   (batch-first)
  CTCLoss input : [T, B, V]   (time-first)
  → always permute(1, 0, 2) before passing to CTCLoss
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..configs.default import VocabConfig


# ---------------------------------------------------------------------------
# Singleton CTCLoss (constructed once per blank_idx value)
# ---------------------------------------------------------------------------

_ctc_cache: dict[int, nn.CTCLoss] = {}


def _ctc_fn(blank_idx: int) -> nn.CTCLoss:
    """Return a cached nn.CTCLoss for the given blank index."""
    if blank_idx not in _ctc_cache:
        _ctc_cache[blank_idx] = nn.CTCLoss(
            blank=blank_idx,
            reduction="mean",
            zero_infinity=True,  # silences Inf losses from degenerate clips
        )
    return _ctc_cache[blank_idx]


# ---------------------------------------------------------------------------
# CTC loss
# ---------------------------------------------------------------------------

def ctc_loss(
    ctc_log_probs:  torch.Tensor,    # [B, T, V]  batch-first log-softmax output
    targets:        torch.Tensor,    # [B, L]     1-based char indices (CTC targets)
    input_lengths:  torch.Tensor,    # [B]        encoded sequence lengths
    target_lengths: torch.Tensor,    # [B]        true label lengths
    vocab:          VocabConfig,
) -> torch.Tensor:
    """
    Compute CTC loss.

    Parameters
    ----------
    ctc_log_probs  : [B, T, V]  — log-softmax probabilities from the model.
                     These are batch-first; this function handles the permute.
    targets        : [B, L]     — integer target labels (0-indexed CTC targets).
    input_lengths  : [B]        — enc_lens returned by model.forward().
                     MUST be the scaled encoded lengths, NOT the raw frame counts.
    target_lengths : [B]        — number of valid labels per sample.
    vocab          : VocabConfig for blank_idx.

    Returns
    -------
    loss : scalar tensor with autograd graph attached.

    Tensor convention
    -----------------
    nn.CTCLoss expects:
        log_probs  : [T, B, V]  — time-first
        targets    : [B*L] or [B, L]
        input_lengths  : [B]
        target_lengths : [B]

    This function permutes [B, T, V] → [T, B, V] internally.
    Model outputs are NEVER passed time-first to CTCLoss elsewhere.
    """
    loss = _ctc_fn(vocab.blank_idx)(
        ctc_log_probs.permute(1, 0, 2),  # [B, T, V] → [T, B, V]  ← required
        targets,
        input_lengths,
        target_lengths,
    )

    # NaN guard for degenerate clips / extreme early-training instability
    if torch.isnan(loss):
        loss = ctc_log_probs.new_zeros(1, requires_grad=True).squeeze()

    return loss

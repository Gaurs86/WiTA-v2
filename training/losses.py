"""
training/losses.py — Hybrid CTC + Attention loss.

Self-contained. No external repo dependency.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..configs.default import VocabConfig, TrainConfig


# ---------------------------------------------------------------------------
# Singleton loss modules (constructed once per process)
# ---------------------------------------------------------------------------
_ctc_cache:  dict[int, nn.CTCLoss]          = {}
_ce_cache:   dict[tuple, nn.CrossEntropyLoss] = {}


def _ctc_fn(blank_idx: int) -> nn.CTCLoss:
    if blank_idx not in _ctc_cache:
        _ctc_cache[blank_idx] = nn.CTCLoss(
            blank=blank_idx, reduction="mean", zero_infinity=True
        )
    return _ctc_cache[blank_idx]


def _ce_fn(pad_idx: int, label_smoothing: float) -> nn.CrossEntropyLoss:
    key = (pad_idx, label_smoothing)
    if key not in _ce_cache:
        _ce_cache[key] = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=label_smoothing
        )
    return _ce_cache[key]


# ---------------------------------------------------------------------------
# Attention target builder
# ---------------------------------------------------------------------------

def prepare_attn_targets(
    labels:        torch.Tensor,    # [B, L]  CTC int targets (1-based chars, 0-padded)
    label_lengths: torch.Tensor,    # [B]
    vocab:         VocabConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build teacher-forcing inputs and CE targets for the attention decoder.

    tgt_input  [B, L+1] = <sos> + labels          (decoder input)
    tgt_output [B, L+1] = labels + <eos>           (CE target)
                          positions beyond label_len+1 masked to PAD

    Returns
    -------
    tgt_input       : [B, L+1]  long
    tgt_output      : [B, L+1]  long
    tgt_key_padding : [B, L+1]  bool  (True = padded, ignored by decoder)
    """
    B, L   = labels.shape
    device = labels.device

    sos = torch.full((B, 1), vocab.sos_idx, dtype=torch.long, device=device)
    eos = torch.full((B, 1), vocab.eos_idx, dtype=torch.long, device=device)

    tgt_input  = torch.cat([sos, labels], dim=1)    # [B, L+1]
    tgt_output = torch.cat([labels, eos], dim=1)    # [B, L+1]

    for b in range(B):
        l = int(label_lengths[b].item())
        # tgt_input: positions l+1 onward are zero-padding from pad_collate → PAD_IDX
        tgt_input[b,  l + 1:] = vocab.pad_idx
        # tgt_output: keep EOS at position l; mask everything after
        tgt_output[b, l + 1:] = vocab.pad_idx

    tgt_key_padding = tgt_input.eq(vocab.pad_idx)   # [B, L+1] bool
    return tgt_input, tgt_output, tgt_key_padding


# ---------------------------------------------------------------------------
# Lambda annealing
# ---------------------------------------------------------------------------

def get_lambda_ctc(epoch: int, total_epochs: int, train_cfg: TrainConfig) -> float:
    """Linearly anneal CTC weight from lambda_ctc_start → lambda_ctc_min."""
    span = train_cfg.lambda_ctc_start - train_cfg.lambda_ctc_min
    lam  = train_cfg.lambda_ctc_start - (epoch / max(total_epochs - 1, 1)) * span
    return max(train_cfg.lambda_ctc_min, float(lam))


# ---------------------------------------------------------------------------
# Hybrid loss
# ---------------------------------------------------------------------------

def hybrid_loss(
    ctc_log_probs:  torch.Tensor,    # [B, T, ctc_vocab]  (batch-first from model)
    attn_logits:    torch.Tensor,    # [B, L, attn_vocab]
    targets:        torch.Tensor,    # [B, L_ctc]  1-based char indices
    attn_targets:   torch.Tensor,    # [B, L+1]    with EOS
    input_lengths:  torch.Tensor,    # [B]
    target_lengths: torch.Tensor,    # [B]
    lambda_ctc:     float,
    vocab:          VocabConfig,
    train_cfg:      TrainConfig,
) -> tuple[torch.Tensor, float, float]:
    """
    total = lambda_ctc * ctc_loss + (1 - lambda_ctc) * attn_loss

    Returns  (total_loss, ctc_scalar, attn_scalar)
    """
    ctc_loss = _ctc_fn(vocab.blank_idx)(
        ctc_log_probs.permute(1, 0, 2),  # [B, T, V] → [T, B, V] for nn.CTCLoss
        targets, input_lengths, target_lengths
    )

    B, L, V = attn_logits.shape
    attn_loss = _ce_fn(vocab.pad_idx, train_cfg.label_smoothing)(
        attn_logits.reshape(B * L, V),
        attn_targets.reshape(B * L),
    )

    # NaN guard (degenerate clips in early epochs)
    if torch.isnan(ctc_loss):
        ctc_loss = ctc_log_probs.new_zeros(1, requires_grad=True).squeeze()
    if torch.isnan(attn_loss):
        attn_loss = attn_logits.new_zeros(1, requires_grad=True).squeeze()

    total = lambda_ctc * ctc_loss + (1.0 - lambda_ctc) * attn_loss
    return total, ctc_loss.item(), attn_loss.item()

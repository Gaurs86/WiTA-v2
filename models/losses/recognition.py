"""Recognition losses for CTC, attention, and hybrid training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from configs.default import ObjectiveConfig
from datasets.vocab import Vocabulary


@dataclass
class LossOutput:
    total: torch.Tensor
    ctc: float | None = None
    attention: float | None = None
    ctc_weight: float = 0.0


def ctc_weight_for_epoch(epoch: int, total_epochs: int, cfg: ObjectiveConfig) -> float:
    if cfg.mode == "ctc":
        return 1.0
    if cfg.mode == "attention":
        return 0.0
    span = cfg.ctc_weight_start - cfg.ctc_weight_end
    value = cfg.ctc_weight_start - (epoch / max(total_epochs - 1, 1)) * span
    return max(cfg.ctc_weight_end, float(value))


def prepare_attention_targets(
    labels: torch.Tensor,
    lengths: torch.Tensor,
    vocab: Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, max_len = labels.shape
    device = labels.device
    decoder_input = torch.full((batch, max_len + 1), vocab.pad_idx, dtype=torch.long, device=device)
    decoder_output = torch.full((batch, max_len + 1), vocab.pad_idx, dtype=torch.long, device=device)
    decoder_input[:, 0] = vocab.sos_idx

    for row in range(batch):
        length = int(lengths[row])
        if length:
            decoder_input[row, 1 : length + 1] = labels[row, :length]
            decoder_output[row, :length] = labels[row, :length]
        decoder_output[row, length] = vocab.eos_idx

    padding_mask = decoder_input.eq(vocab.pad_idx)
    return decoder_input, decoder_output, padding_mask


class RecognitionLoss(nn.Module):
    def __init__(self, vocab: Vocabulary, cfg: ObjectiveConfig):
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.ctc_loss = nn.CTCLoss(
            blank=vocab.blank_idx,
            reduction="mean",
            zero_infinity=cfg.zero_infinity,
        )
        self.attention_loss = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=cfg.label_smoothing,
        )

    def forward(
        self,
        *,
        ctc_log_probs: torch.Tensor | None,
        attention_logits: torch.Tensor | None,
        encoded_lengths: torch.Tensor,
        ctc_targets: torch.Tensor,
        ctc_lengths: torch.Tensor,
        attention_targets: torch.Tensor | None,
        ctc_weight: float,
    ) -> LossOutput:
        total = encoded_lengths.new_tensor(0.0, dtype=torch.float32)
        ctc_value: torch.Tensor | None = None
        attention_value: torch.Tensor | None = None

        if self.cfg.mode in {"ctc", "hybrid"}:
            if ctc_log_probs is None:
                raise ValueError("CTC loss requested but model did not return CTC log probabilities.")
            ctc_value = self.ctc_loss(ctc_log_probs, ctc_targets, encoded_lengths, ctc_lengths)
            if not torch.isfinite(ctc_value):
                ctc_value = ctc_log_probs.new_zeros(())
            total = total + ctc_weight * ctc_value

        if self.cfg.mode in {"attention", "hybrid"}:
            if attention_logits is None or attention_targets is None:
                raise ValueError("Attention loss requested but attention outputs/targets are missing.")
            batch, length, vocab_size = attention_logits.shape
            attention_value = self.attention_loss(
                attention_logits.reshape(batch * length, vocab_size),
                attention_targets.reshape(batch * length),
            )
            if not torch.isfinite(attention_value):
                attention_value = attention_logits.new_zeros(())
            total = total + (1.0 - ctc_weight) * attention_value

        return LossOutput(
            total=total,
            ctc=None if ctc_value is None else float(ctc_value.detach().cpu()),
            attention=None if attention_value is None else float(attention_value.detach().cpu()),
            ctc_weight=ctc_weight,
        )

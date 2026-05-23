"""
models/hybrid_model.py — WiTAHybridModel.

FIX (Bug 3 — AMP double-wrapping)
-----------------------------------
The original forward() wrapped its own body in torch.autocast().  The trainer
also wrapped the loss computation in a second autocast scope.  This caused the
model's fp16 activations to exit one autocast context and re-enter another,
breaking the intended single-scope AMP flow and creating subtle precision
inconsistencies near the CTC log-softmax boundary.

Fix: torch.autocast() has been removed from model.forward().  AMP is now
managed exclusively by the trainer (which wraps both the model forward pass
AND the loss computation in a single `with autocast(...)` block).  This is
the standard PyTorch AMP pattern and the correct approach for DataParallel.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.default import Config
from .encoders.resnet3d import VideoResNet, build_encoder
from .modules.recurrent import build_recurrent_head, CTCProjection
from .decoders.attention import AttentionDecoder


class WiTAHybridModel(nn.Module):
    """
    End-to-end trainable hybrid CTC + Attention model.

    Architecture
    ------------
    clips [B, T, C, H, W]
        ↓
    VideoResNet encoder
        → features  [B, T', enc_dim]
        ↓                    ↓
    RecurrentHead          AttentionDecoder
        → rnn_out               → attn_logits
        ↓                         [B, L, attn_vocab]
    CTCProjection
        → ctc_log_probs  [B, T', ctc_vocab]

    AMP Note
    --------
    This module does NOT manage its own autocast context.  Mixed-precision
    is controlled at the trainer level so that the full forward+loss graph
    sits inside a single autocast scope — the correct PyTorch AMP pattern.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        vc = cfg.vocab
        ec = cfg.encoder
        rc = cfg.recurrent
        ac = cfg.attn

        self.encoder: VideoResNet = build_encoder(ec)
        self.recurrent, rnn_out_dim = build_recurrent_head(ec.out_dim, rc)
        self.ctc_proj = CTCProjection(
            d_in=rnn_out_dim,
            num_class=vc.ctc_vocab_size,
            cfg=rc,
        )
        self.attn_decoder = AttentionDecoder(
            encoder_dim=ec.out_dim,
            vocab_cfg=vc,
            attn_cfg=ac,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, clips: torch.Tensor) -> torch.Tensor:
        """clips [B, T, C, H, W] → features [B, T', enc_dim]."""
        return self.encoder(clips.permute(0, 2, 1, 3, 4))

    def _ctc_logits(
        self,
        features: torch.Tensor,   # [B, T', enc_dim]
        seq_lens: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """features → [B, T', ctc_vocab_size] log-softmax (batch-first)."""
        rnn_out = self.recurrent(features, seq_lens)
        logits  = self.ctc_proj(rnn_out)
        return logits.log_softmax(2)

    # ------------------------------------------------------------------
    # Forward  (no autocast here — AMP is owned by the trainer)
    # ------------------------------------------------------------------

    def forward(
        self,
        clips:        torch.Tensor,
        seq_lens:     torch.Tensor,
        tgt_tokens:   torch.Tensor | None = None,
        tgt_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns
        -------
        ctc_log_probs  : [B, T', ctc_vocab_size]
        attn_logits    : [B, L, attn_vocab_size]  (None when tgt_tokens is None)
        """
        features      = self._encode(clips)
        ctc_log_probs = self._ctc_logits(features, seq_lens)

        attn_logits: torch.Tensor | None = None
        if tgt_tokens is not None:
            attn_logits = self.attn_decoder(features, tgt_tokens, tgt_pad_mask)

        return ctc_log_probs, attn_logits

    @torch.no_grad()
    def decode_ctc_greedy(self, clips: torch.Tensor, seq_lens: torch.Tensor) -> list[list[int]]:
        """Fast greedy CTC decode; returns list of B token-index lists."""
        features  = self._encode(clips)
        log_probs = self._ctc_logits(features, seq_lens)   # [B, T, V]
        best      = log_probs.argmax(-1)                    # [B, T]
        blank     = self.cfg.vocab.blank_idx
        decoded   = []
        for seq in best:
            toks = seq.tolist()
            collapsed, prev = [], None
            for t in toks:
                if t != prev:
                    collapsed.append(t)
                prev = t
            decoded.append([t for t in collapsed if t != blank])
        return decoded

    @torch.no_grad()
    def decode_attention(self, clips: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Greedy attention decode; returns [B, L] token indices."""
        features = self._encode(clips)
        return self.attn_decoder.forward_inference(features)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> WiTAHybridModel:
    return WiTAHybridModel(cfg)

"""
models/hybrid_model.py — WiTAHybridModel.

Fully self-contained. Builds all components from Config:
  Encoder  : VideoResNet family  (models/encoders/resnet3d.py)
  Recurrent: BiLSTM / BiGRU / Transformer / None  (models/modules/recurrent.py)
  CTC head : Linear projection   (models/modules/recurrent.py)
  Attention: Transformer decoder (models/decoders/attention.py)

No import from the original WiTA repo.
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
        → ctc_logits  [T', B, ctc_vocab]

    Parameters
    ----------
    cfg : fully built Config (call cfg.build() before passing in)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        vc = cfg.vocab
        ec = cfg.encoder
        rc = cfg.recurrent
        ac = cfg.attn

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder: VideoResNet = build_encoder(ec)

        # ------------------------------------------------------------------
        # Optional recurrent head
        # ------------------------------------------------------------------
        self.recurrent, rnn_out_dim = build_recurrent_head(ec.out_dim, rc)

        # ------------------------------------------------------------------
        # CTC projection head
        # ------------------------------------------------------------------
        self.ctc_proj = CTCProjection(
            d_in=rnn_out_dim,
            num_class=vc.ctc_vocab_size,
            cfg=rc,
        )

        # ------------------------------------------------------------------
        # Attention decoder
        # ------------------------------------------------------------------
        self.attn_decoder = AttentionDecoder(
            encoder_dim=ec.out_dim,    # attn decoder reads raw encoder features
            vocab_cfg=vc,
            attn_cfg=ac,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips [B, T, C, H, W] → features [B, T', enc_dim]
        Permutes to [B, C, T, H, W] for the 3-D ResNet.
        """
        return self.encoder(clips.permute(0, 2, 1, 3, 4))

    def _ctc_logits(
        self,
        features: torch.Tensor,   # [B, T', enc_dim]
        seq_lens: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        features → [T', B, ctc_vocab_size] log-softmax for nn.CTCLoss.
        """
        rnn_out = self.recurrent(features, seq_lens)     # [B, T', rnn_out_dim]
        logits  = self.ctc_proj(rnn_out)                 # [B, T', ctc_vocab]
        return logits.permute(1, 0, 2).log_softmax(2)   # [T', B, ctc_vocab]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        clips:        torch.Tensor,              # [B, T, C, H, W]
        seq_lens:     torch.Tensor,              # [B]
        tgt_tokens:   torch.Tensor | None = None,   # [B, L] with <sos> prepended
        tgt_pad_mask: torch.Tensor | None = None,   # [B, L] bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns
        -------
        ctc_log_probs  : [T', B, ctc_vocab_size]
        attn_logits    : [B, L, attn_vocab_size]  (None when tgt_tokens is None)
        """
        features      = self._encode(clips)                        # [B, T', D]
        ctc_log_probs = self._ctc_logits(features, seq_lens)       # [T', B, V_ctc]

        attn_logits: torch.Tensor | None = None
        if tgt_tokens is not None:
            attn_logits = self.attn_decoder(features, tgt_tokens, tgt_pad_mask)

        return ctc_log_probs, attn_logits

    @torch.no_grad()
    def decode_ctc_greedy(self, clips: torch.Tensor, seq_lens: torch.Tensor) -> list[list[int]]:
        """Fast greedy CTC decode; returns list of B token-index lists."""
        features = self._encode(clips)
        log_probs = self._ctc_logits(features, seq_lens)   # [T, B, V]
        best = log_probs.argmax(-1).transpose(0, 1)        # [B, T]
        blank = self.cfg.vocab.blank_idx
        decoded = []
        for seq in best:
            toks = seq.tolist()
            collapsed = []
            prev = None
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
    """Convenience function. cfg must already be built (cfg.build())."""
    return WiTAHybridModel(cfg)

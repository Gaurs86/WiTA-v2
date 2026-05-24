"""
models/hybrid_model.py — WiTAHybridModel.

FIX (Bug 3 — AMP double-wrapping): see previous note.

FIX (Bug 4 — Architectural gradient isolation): CRITICAL NEW FIX
-----------------------------------------------------------------
Root cause identified from the epoch-1..40 log:

The original forward() was:
    features = self._encode(clips)             # [B, T', 256]
    ctc_log_probs = ctc_proj(LSTM(features))   # CTC reads from LSTM output
    attn_logits = attn_decoder(features, ...)  # Attn reads RAW encoder features!

This creates a gradient isolation trap:
  • CTCLoss gradient flows:  CTC → ctc_proj → BiLSTM → encoder
  • CE-loss gradient flows:  CE → attn_decoder → encoder   (bypasses LSTM entirely)

When CTC blank-collapses (which it does by epoch 8–10), the BiLSTM receives
≈ zero CTC gradient.  The encoder is then optimised solely by the attention
decoder — but the attention decoder reads raw 256-dim CNN features with no
temporal context from the LSTM.  The result is the attention decoder learning
English character n-gram statistics from CNN features (explaining the "English-
looking-but-wrong" predictions: "coreine", "atenenen", "carenenen"), while the
LSTM/CTC path stays permanently dead.

Fix: route the attention decoder through the LSTM output (via a lightweight
linear projection back to 256-dim), so the attention decoder's gradient
keeps the LSTM alive even when CTC is in blank mode.

Also: initialise the CTC blank-class bias to -3.0 so the blank is harder to
predict initially, which delays and sometimes prevents blank collapse entirely.

New data flow:
    encoder  [B, T', 256]
         |
    BiLSTM  [B, T', 512]
         |─────────────────────── rnn_proj Linear(512→256)
         |                               |
    ctc_proj [B, T', 28]         attn_decoder (d_model=256, unchanged)
         |                               |
    CTCLoss                          CE Loss

Both branches now share the LSTM, so both contribute gradients to it.

AMP Note
--------
This module does NOT manage its own autocast context.  Mixed-precision
is controlled at the trainer level (one unified scope covering both the
model forward and the loss computation).
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

    Architecture (after Bug 4 fix)
    --------------------------------
    clips [B, T, C, H, W]
        ↓
    VideoResNet encoder
        → enc_features  [B, T', enc_dim=256]
        ↓
    BiLSTM RecurrentHead
        → rnn_out  [B, T', rnn_dim=512]
        ↓                    ↓
    CTCProjection        rnn_proj  Linear(512→256)
        → ctc_log_probs      ↓
          [B, T', 28]   AttentionDecoder
                             → attn_logits [B, L, attn_vocab=31]
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

        # Projection from LSTM output space back to encoder_dim for the
        # attention decoder.  If rnn_out_dim == ec.out_dim, this is an
        # identity (no extra parameters).
        if rnn_out_dim != ec.out_dim:
            self.rnn_proj: nn.Module = nn.Sequential(
                nn.Linear(rnn_out_dim, ec.out_dim),
                nn.LayerNorm(ec.out_dim),
            )
        else:
            self.rnn_proj = nn.Identity()

        self.attn_decoder = AttentionDecoder(
            encoder_dim=ec.out_dim,   # unchanged — always 256
            vocab_cfg=vc,
            attn_cfg=ac,
        )

        # ── Blank-bias initialisation ──────────────────────────────────────
        # Initialise the CTC output layer's blank-class bias to a low value
        # so the model does not trivially collapse to predicting blank for
        # every frame during early training.
        self._init_blank_bias(vc.blank_idx)

    def _init_blank_bias(self, blank_idx: int) -> None:
        """Set CTC blank-class output bias to -3.0 at init time."""
        last_fc = self.ctc_proj.fc2          # always exists; see CTCProjection
        with torch.no_grad():
            last_fc.bias.data[blank_idx] = -3.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, clips: torch.Tensor) -> torch.Tensor:
        """clips [B, T, C, H, W] → enc_features [B, T', enc_dim]."""
        return self.encoder(clips.permute(0, 2, 1, 3, 4))

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
        enc_features = self._encode(clips)                    # [B, T', 256]
        rnn_out      = self.recurrent(enc_features, seq_lens) # [B, T', 512]

        # CTC branch: from LSTM output (preserves temporal alignment)
        ctc_log_probs = self.ctc_proj(rnn_out).log_softmax(2) # [B, T', 28]

        # Attention branch: project LSTM output back to 256-dim so the
        # attention decoder uses the same dimensionality as before, but
        # now receives gradients through the LSTM.
        attn_logits: torch.Tensor | None = None
        if tgt_tokens is not None:
            attn_features = self.rnn_proj(rnn_out)             # [B, T', 256]
            attn_logits   = self.attn_decoder(
                attn_features, tgt_tokens, tgt_pad_mask
            )

        return ctc_log_probs, attn_logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_ctc_greedy(
        self, clips: torch.Tensor, seq_lens: torch.Tensor
    ) -> list[list[int]]:
        """Greedy CTC decode; returns list of B token-index lists."""
        enc  = self._encode(clips)
        rnn  = self.recurrent(enc, seq_lens)
        lp   = self.ctc_proj(rnn).log_softmax(2)   # [B, T, V]
        best = lp.argmax(-1)                        # [B, T]
        blank = self.cfg.vocab.blank_idx
        decoded = []
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
    def decode_attention(
        self, clips: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Greedy attention decode; returns [B, L] token indices."""
        enc  = self._encode(clips)
        rnn  = self.recurrent(enc, seq_lens)
        feat = self.rnn_proj(rnn)
        return self.attn_decoder.forward_inference(feat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> WiTAHybridModel:
    return WiTAHybridModel(cfg)

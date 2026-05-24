"""
models/hybrid_model.py — WiTAHybridModel.

FIX SUMMARY (this revision):
------------------------------
Bug A — blank bias direction (CRITICAL)
    Previous: blank_bias = −3.0 → P(blank) ≈ 0.0018 at init.
    Words with repeated characters (needed, common, letter, etc.) require at
    least one blank between identical adjacent chars.  With P(blank)≈0, those
    samples produce loss=∞ → zero_infinity fires → gradient=0.  CTC never
    learns because it can never produce valid alignments for a large fraction
    of the vocabulary.

    Fix: blank_bias = +log(p_blank*(V−1)/(1−p_blank)) ≈ +4.1 for p_blank=0.7,
    giving P(blank)≈0.70 — matching the natural proportion (T−L)/T = (16−5)/16.

Bug B — lambda_ctc = 0.1 (CRITICAL)
    Not fixed here — it is a config value (TrainConfig.lambda_ctc_start).
    Change lambda_ctc_start from 0.1 → 0.75 and lambda_ctc_min from ~0 → 0.3.
    See losses.py and training config.

Bug C — seq_lens not scaled by temporal stride (HIGH)
    Previous: raw frame counts from the dataloader were passed directly into
    both pack_padded_sequence and CTCLoss.  If the dataset computes encoded
    lengths correctly this happens not to crash, but:
      • For non-multiples of 4 the integer arithmetic differs from actual
        encoder convolution output (e.g. raw=63 → dataset gives 63//4=15 but
        encoder actually outputs T'=16).
      • The inference decode paths (decode_ctc_greedy, decode_attention) had
        no length scaling at all.

    Fix: _compute_enc_lens() derives enc_lens from the actual T_enc/T_raw ratio
    inside the model, making it architecture-agnostic and robust to rounding.
    forward() returns enc_lens so the trainer passes it correctly to CTCLoss.

Bug D — ReLU in CTCProjection (MEDIUM)
    fc1 → ReLU → fc2 can produce dead neurons when CTC gradients are weak.
    Changed to GELU (non-zero gradient everywhere, empirically better for
    sequence-to-sequence projection heads).
"""

from __future__ import annotations
import math
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

    forward() now returns (ctc_log_probs, attn_logits, enc_lens) where
    enc_lens [B] are the correctly-scaled encoded sequence lengths that
    the trainer must pass to CTCLoss as input_lengths.
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

        if rnn_out_dim != ec.out_dim:
            self.rnn_proj: nn.Module = nn.Sequential(
                nn.Linear(rnn_out_dim, ec.out_dim),
                nn.LayerNorm(ec.out_dim),
            )
        else:
            self.rnn_proj = nn.Identity()

        self.attn_decoder = AttentionDecoder(
            encoder_dim=ec.out_dim,
            vocab_cfg=vc,
            attn_cfg=ac,
        )

        self._init_blank_bias(vc.blank_idx, p_blank=0.70)

    # ------------------------------------------------------------------
    # Blank bias — FIXED direction
    # ------------------------------------------------------------------

    def _init_blank_bias(self, blank_idx: int, p_blank: float = 0.70) -> None:
        """
        Initialise the CTC blank bias so P(blank) ≈ p_blank at model init.

        Target p_blank ≈ 0.65–0.75, matching the natural proportion of blank
        frames in a CTC alignment: (T − L) / T = (16 − 5) / 16 ≈ 0.69.

        Formula (softmax inverse):
            P(blank) = exp(b) / (exp(b) + (V−1)·1)    [other logits ≈ 0]
            b = log(p_blank · (V−1) / (1 − p_blank))

        For V=28, p_blank=0.70:
            b = log(0.70 · 27 / 0.30) = log(63) ≈ 4.14   ← POSITIVE

        Previous value was −3.0, giving P(blank) ≈ 0.002, which made words
        with repeated characters unalignable and triggered zero_infinity on
        the majority of the training set.
        """
        V = self.ctc_proj.fc2.out_features
        b = math.log(p_blank * (V - 1) / (1.0 - p_blank))
        with torch.no_grad():
            self.ctc_proj.fc2.bias.data[blank_idx] = b

    # ------------------------------------------------------------------
    # Encoded length computation — FIXED
    # ------------------------------------------------------------------

    def _compute_enc_lens(
        self,
        raw_lens: torch.Tensor,
        T_enc:    int,
        T_raw:    int,
    ) -> torch.Tensor:
        """
        Scale raw frame counts to encoded sequence lengths.

        Uses the actual ratio T_enc / T_raw rather than a hardcoded stride.
        This is architecture-agnostic and handles integer-rounding edge cases
        that a simple // 4 misses (e.g. raw=63 → encoder outputs 16, not 15).

        Returns enc_lens clamped to [1, T_enc].
        """
        if T_raw == 0 or T_enc == 0:
            return raw_lens.clamp(min=1, max=max(T_enc, 1))
        scale = T_enc / T_raw                              # e.g. 16/64 = 0.25
        enc_lens = (raw_lens.float() * scale).ceil().long()
        return enc_lens.clamp(min=1, max=T_enc)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Returns
        -------
        ctc_log_probs : [B, T', ctc_vocab_size]
        attn_logits   : [B, L, attn_vocab_size]  (None when tgt_tokens is None)
        enc_lens      : [B]  encoded sequence lengths — pass to CTCLoss as
                             input_lengths.  DO NOT use the raw seq_lens.

        The trainer must use enc_lens (not raw seq_lens) when calling
        hybrid_loss().
        """
        T_raw        = clips.shape[1]                         # raw temporal dim
        enc_features = self._encode(clips)                    # [B, T', 256]
        T_enc        = enc_features.shape[1]

        # --- Scale lengths from raw-frame space to encoded-frame space ---
        enc_lens = self._compute_enc_lens(seq_lens, T_enc, T_raw)   # [B]

        rnn_out = self.recurrent(enc_features, enc_lens)             # [B, T', 512]

        # CTC branch
        ctc_log_probs = self.ctc_proj(rnn_out).log_softmax(2)       # [B, T', 28]

        # Attention branch
        attn_logits: torch.Tensor | None = None
        if tgt_tokens is not None:
            attn_features = self.rnn_proj(rnn_out)                  # [B, T', 256]
            attn_logits   = self.attn_decoder(
                attn_features, tgt_tokens, tgt_pad_mask
            )

        return ctc_log_probs, attn_logits, enc_lens

    # ------------------------------------------------------------------
    # Inference helpers  (also use enc_lens now)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_ctc_greedy(
        self, clips: torch.Tensor, seq_lens: torch.Tensor
    ) -> list[list[int]]:
        T_raw = clips.shape[1]
        enc   = self._encode(clips)
        T_enc = enc.shape[1]
        enc_lens = self._compute_enc_lens(seq_lens, T_enc, T_raw)

        rnn  = self.recurrent(enc, enc_lens)
        lp   = self.ctc_proj(rnn).log_softmax(2)                    # [B, T, V]
        best = lp.argmax(-1)                                        # [B, T]
        blank = self.cfg.vocab.blank_idx
        decoded = []
        for b_idx, seq in enumerate(best):
            valid_len = int(enc_lens[b_idx].item())
            toks = seq[:valid_len].tolist()
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
        T_raw = clips.shape[1]
        enc   = self._encode(clips)
        T_enc = enc.shape[1]
        enc_lens = self._compute_enc_lens(seq_lens, T_enc, T_raw)

        rnn  = self.recurrent(enc, enc_lens)
        feat = self.rnn_proj(rnn)
        return self.attn_decoder.forward_inference(feat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> WiTAHybridModel:
    return WiTAHybridModel(cfg)

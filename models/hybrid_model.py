"""
models/hybrid_model.py — WiTACTCModel (VideoMAE / Video Swin + CTC).

Refactored from WiTAHybridModel (R3D + BiLSTM + Hybrid CTC/Attention)
to a clean CTC-only architecture with a pretrained VideoMAE backbone.

Architecture
------------
clips [B, T, C, H, W]
    ↓
VideoMAEEncoder (or VideoSwinEncoder / VideoResNet fallback)
    → enc_features  [B, T', enc_dim]       T'=8 for VideoMAE-base/16fr
    ↓
BiLSTM RecurrentHead  (or "none" for pure linear)
    → rnn_out       [B, T', rnn_dim]
    ↓
CTCProjection  Linear(rnn_dim → vocab)
    → ctc_logits    [B, T', ctc_vocab_size]
    ↓
log_softmax(-1)
    → ctc_log_probs [B, T', V]             → permute → [T', B, V] for CTCLoss

Removed from hybrid version
----------------------------
• AttentionDecoder  (attention branch)
• rnn_proj / attn_features routing
• All attention-decoder imports
• No tgt_tokens / tgt_pad_mask in forward()
• No hybrid loss — forward() returns (ctc_log_probs, enc_lens) only

Kept / fixed
------------
• _compute_enc_lens()  — ratio-based scaling (Bug C fix preserved)
• AMP-compatible (no internal autocast; trainer owns the scope)
• CTCProjection now uses GELU instead of ReLU (Bug D fix)
• WiTAHybridModel alias for backward compatibility
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..configs.default import Config, EncoderConfig
from .modules.recurrent import build_recurrent_head, CTCProjection


# ---------------------------------------------------------------------------
# Encoder dispatch
# ---------------------------------------------------------------------------

def _build_encoder(ec: EncoderConfig) -> nn.Module:
    """
    Build the visual backbone based on ec.arch.

    VideoMAE / Video Swin  →  models/encoders/videomae_encoder.py
    R3D / MC3 / R2+1D      →  models/encoders/resnet3d.py  (unchanged)
    """
    arch = ec.arch.lower()
    if arch in ("videomae", "video_swin"):
        from .encoders.videomae_encoder import build_video_encoder
        return build_video_encoder(ec)
    else:
        from .encoders.resnet3d import build_encoder as build_r3d
        return build_r3d(ec)


# ---------------------------------------------------------------------------
# WiTACTCModel
# ---------------------------------------------------------------------------

class WiTACTCModel(nn.Module):
    """
    End-to-end trainable CTC model with VideoMAE (or R3D) backbone.

    forward() signature is compatible with the hybrid model's forward()
    (extra tgt_tokens / tgt_pad_mask args are accepted but silently ignored)
    so existing trainer / evaluator code works without changes to call sites.

    Returns
    -------
    ctc_log_probs : [B, T', ctc_vocab_size]  — log-softmax probabilities
    enc_lens      : [B]  — encoded sequence lengths (for CTCLoss input_lengths)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        vc = cfg.vocab
        ec = cfg.encoder
        rc = cfg.recurrent

        # ── Visual backbone ──────────────────────────────────────────────
        self.encoder: nn.Module = _build_encoder(ec)
        encoder_dim: int        = ec.out_dim          # projection output dim

        # ── Optional recurrent head ──────────────────────────────────────
        self.recurrent, rnn_out_dim = build_recurrent_head(encoder_dim, rc)

        # ── CTC projection head ──────────────────────────────────────────
        self.ctc_proj = CTCProjection(
            d_in=rnn_out_dim,
            num_class=vc.ctc_vocab_size,
            cfg=rc,
        )

        print(
            f"[WiTACTCModel] encoder_dim={encoder_dim}, "
            f"rnn_out_dim={rnn_out_dim}, "
            f"ctc_vocab_size={vc.ctc_vocab_size}"
        )

    # ------------------------------------------------------------------
    # Encoded length scaling  (Bug C fix preserved from hybrid model)
    # ------------------------------------------------------------------

    def _compute_enc_lens(
        self,
        raw_lens: torch.Tensor,
        T_enc:    int,
        T_raw:    int,
    ) -> torch.Tensor:
        """
        Scale raw frame counts → encoded sequence lengths.

        Uses the actual T_enc/T_raw ratio rather than a hardcoded stride so
        the computation is architecture-agnostic and handles edge cases.

        Returns enc_lens clamped to [1, T_enc].

        VideoMAE note
        -------------
        After temporal resampling to 16 frames and tube pooling to T'=8,
        T_enc=8 for all samples regardless of their original length. The
        scaling will give:
            enc_lens[i] = ceil(raw_lens[i] * 8 / T_raw)
        which correctly reflects that longer clips contribute more content
        to each temporal slot.
        """
        if T_raw == 0 or T_enc == 0:
            return raw_lens.clamp(min=1, max=max(T_enc, 1))
        scale    = T_enc / T_raw
        enc_lens = (raw_lens.float() * scale).ceil().long()
        return enc_lens.clamp(min=1, max=T_enc)

    # ------------------------------------------------------------------
    # Internal encode helper
    # ------------------------------------------------------------------

    def _encode(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips [B, T, C, H, W]  → enc_features [B, T', enc_dim]   (raw frames)
        clips [B, T', D]        → enc_features [B, T', D]          (cached feats)

        When using CachedFeaturesDataset the loader yields [B, T', D] tensors
        (pre-extracted VideoMAE features). Detecting ndim==3 lets us skip the
        entire backbone forward pass — only BiLSTM + CTC head run per step.
        This is what makes training go from 1000s/epoch → ~5s/epoch on T4.
        """
        if clips.ndim == 3:
            # Already extracted features [B, T', D] — skip backbone entirely
            return clips

        arch = self.cfg.encoder.arch.lower()
        if arch in ("videomae", "video_swin"):
            return self.encoder(clips)                      # already [B,T,C,H,W]
        else:
            return self.encoder(clips.permute(0, 2, 1, 3, 4))  # → [B,C,T,H,W]

    # ------------------------------------------------------------------
    # Forward  (no autocast — AMP scope owned by trainer)
    # ------------------------------------------------------------------

    def forward(
        self,
        clips:        torch.Tensor,
        seq_lens:     torch.Tensor,
        tgt_tokens:   torch.Tensor | None = None,   # ignored (compat only)
        tgt_pad_mask: torch.Tensor | None = None,   # ignored (compat only)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        clips     : [B, T, C, H, W]
        seq_lens  : [B]  raw clip lengths in frames

        Returns
        -------
        ctc_log_probs : [B, T', ctc_vocab_size]
        enc_lens      : [B]   encoded lengths — pass to CTCLoss as input_lengths

        IMPORTANT: always use enc_lens (not raw seq_lens) for CTCLoss.

        Shape walkthrough (VideoMAE-base, num_frames=32)
        -------------------------------------------------
        clips         [B, T,  3, 224, 224]  e.g. T=64
        enc_features  [B, 16, 768]           T'=16 from VideoMAE+pool (32fr÷2)
        rnn_out       [B, 16, 512]           BiLSTM hidden_size=256 × 2
        ctc_logits    [B, 16, 28]            ctc_vocab_size=28
        ctc_log_probs [B, 16, 28]
        → CTCLoss wants [T', B, 28] = permute(1, 0, 2)

        With T'=16 CTC can align words up to 16 chars — covers virtually
        all English fingerspelling words in the WiTA dataset.
        (Previous T'=8 caused CTC collapse on >8-char words.)"""
        T_raw = clips.shape[1]

        # 1. Visual backbone (or pass-through if clips is pre-extracted [B,T',D])
        enc_features = self._encode(clips)                  # [B, T', D]
        T_enc        = enc_features.shape[1]

        # 2. Scale raw frame lengths → encoded frame lengths.
        #    For cached features seq_lens already equals T_enc (set by
        #    CachedFeaturesDataset) so _compute_enc_lens returns them unchanged.
        enc_lens = self._compute_enc_lens(seq_lens, T_enc, T_raw)  # [B]

        # 3. Optional recurrent head
        rnn_out = self.recurrent(enc_features, enc_lens)    # [B, T', rnn_dim]

        # 4. CTC projection + log-softmax
        ctc_logits    = self.ctc_proj(rnn_out)              # [B, T', V]
        ctc_log_probs = ctc_logits.log_softmax(-1)          # [B, T', V]

        return ctc_log_probs, enc_lens

    # ------------------------------------------------------------------
    # Backbone un-freezing (called by trainer after warm-up epochs)
    # ------------------------------------------------------------------

    def unfreeze_backbone(self) -> None:
        """
        Un-freeze all backbone parameters for end-to-end fine-tuning.
        Call after the CTC head has warmed up (e.g. epoch 10+).
        """
        ec   = self.cfg.encoder
        arch = ec.arch.lower()
        if arch in ("videomae", "video_swin"):
            for p in self.encoder.backbone.parameters():
                p.requires_grad_(True)
            print("[WiTACTCModel] Backbone UNFROZEN — full fine-tuning enabled")
        else:
            print("[WiTACTCModel] R3D backbone: no freezing was applied; "
                  "unfreeze_backbone() is a no-op")

    # ------------------------------------------------------------------
    # Greedy CTC decode (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_ctc_greedy(
        self,
        clips:    torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> list[list[int]]:
        """
        Greedy (argmax) CTC decode.

        Returns list[list[int]] — one decoded token list per batch item,
        with blank tokens and consecutive duplicates removed.
        """
        T_raw    = clips.shape[1]
        enc      = self._encode(clips)
        T_enc    = enc.shape[1]
        enc_lens = self._compute_enc_lens(seq_lens, T_enc, T_raw)

        rnn      = self.recurrent(enc, enc_lens)
        lp       = self.ctc_proj(rnn).log_softmax(-1)   # [B, T, V]
        best     = lp.argmax(-1)                         # [B, T]
        blank    = self.cfg.vocab.blank_idx

        decoded: list[list[int]] = []
        for b_idx, seq in enumerate(best):
            valid_len = int(enc_lens[b_idx].item())
            toks      = seq[:valid_len].tolist()
            collapsed, prev = [], None
            for t in toks:
                if t != prev:
                    collapsed.append(t)
                prev = t
            decoded.append([t for t in collapsed if t != blank])
        return decoded

    # ------------------------------------------------------------------
    # Kept for evaluator compat — routes to decode_ctc_greedy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_attention(
        self,
        clips:    torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stub — returns greedy CTC decode as a tensor for evaluator compat.
        The attention decoder has been removed; this is intentionally a
        CTC fallback so existing evaluate_cer(decode_mode='attn') calls
        don't crash.
        """
        seqs = self.decode_ctc_greedy(clips, seq_lens)
        # Pad to same length for tensor conversion
        if not seqs:
            return clips.new_zeros(0, 1, dtype=torch.long)
        max_len = max(len(s) for s in seqs)
        pad = self.cfg.vocab.pad_idx
        out = clips.new_full((len(seqs), max(max_len, 1)), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            if s:
                out[i, :len(s)] = clips.new_tensor(s, dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

#: Old name used by existing checkpoints / scripts — maps to WiTACTCModel.
WiTAHybridModel = WiTACTCModel


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> WiTACTCModel:
    """Construct a WiTACTCModel from a fully-built Config."""
    return WiTACTCModel(cfg)

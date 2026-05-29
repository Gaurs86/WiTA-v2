"""
models/conformer_ctc.py — Conformer encoder + ConvTranspose1d upsampler + CTC.

Stage 1 (and later stages) model.  Designed to be the **temporal model**
component of the iterative-ablation plan, used identically across:
  - Stage 1: landmark-only input (190 dims/frame)
  - Stage 2-7: per-frame visual features (various dims)
  - Stage 4+: visual + landmark concatenation

Architecture
------------
  input [B, T_in, D_in]
    → Linear(D_in → d_model)
    → LayerNorm
    → Conformer ×N_layers (PyTorch torchaudio.models.Conformer)
    → ConvTranspose1d(d, d, kernel=upsample, stride=upsample)   # T_out = upsample × T_in
    → LayerNorm
    → Linear(d_model → vocab_size)
    → log_softmax(-1)
    → ctc_log_probs [B, T_out, V]

The upsample stage is the critical fix for the CTC alignment constraint
identified in the Run-8 diagnosis: T_out must be ≥ 2·L_max + 1.  With
T_native = 32 and upsample = 2, T_out = 64 which covers every WiTA label.

Why we wrote our own Conformer encoder
--------------------------------------
torchaudio.models.Conformer exists but is only available in torchaudio
≥ 0.11 and requires a particular dependency chain.  To minimise install
surface, we implement a lean Conformer block here (Macaron FFN + MHSA +
depthwise-conv module + FFN) in ~120 lines.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _FeedForward(nn.Module):
    """Two-layer FFN with GELU + dropout, used as the Macaron pre/post FFN."""
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln   = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, mult * d)
        self.fc2  = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + 0.5 * y    # Macaron-style half-residual


class _MHSA(nn.Module):
    """Multi-head self-attention with pre-LN."""
    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln   = nn.LayerNorm(d)
        self.mha  = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        y = self.ln(x)
        y, _ = self.mha(y, y, y,
                        key_padding_mask=key_padding_mask,
                        need_weights=False)
        return x + self.drop(y)


class _ConvModule(nn.Module):
    """
    Conformer convolution module: pre-LN → pointwise+GLU → depthwise conv
    → BN → swish → pointwise → dropout → residual.
    """
    def __init__(self, d: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "conv kernel size must be odd"
        self.ln      = nn.LayerNorm(d)
        self.pw1     = nn.Conv1d(d, 2 * d, kernel_size=1)
        self.dw      = nn.Conv1d(
            d, d, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d,
        )
        self.bn      = nn.BatchNorm1d(d)
        self.pw2     = nn.Conv1d(d, d, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d]
        y = self.ln(x)
        y = y.transpose(1, 2)              # [B, d, T]
        y = self.pw1(y)                    # [B, 2d, T]
        y = F.glu(y, dim=1)                # [B, d, T]
        y = self.dw(y)                     # depthwise temporal conv
        y = self.bn(y)
        y = y * torch.sigmoid(y)           # swish
        y = self.pw2(y)                    # [B, d, T]
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + y


class ConformerBlock(nn.Module):
    """One Conformer block: ½FFN + MHSA + ConvModule + ½FFN + LN."""
    def __init__(self, d: int, n_heads: int = 4, conv_kernel: int = 15,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ff1   = _FeedForward(d, mult=ff_mult, dropout=dropout)
        self.mhsa  = _MHSA(d, n_heads=n_heads, dropout=dropout)
        self.conv  = _ConvModule(d, kernel_size=conv_kernel, dropout=dropout)
        self.ff2   = _FeedForward(d, mult=ff_mult, dropout=dropout)
        self.ln    = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        x = self.ff1(x)
        x = self.mhsa(x, key_padding_mask=key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


# ---------------------------------------------------------------------------
# ConformerCTC
# ---------------------------------------------------------------------------

class ConformerCTC(nn.Module):
    """
    Conformer encoder + temporal upsampler + linear CTC head.

    Parameters
    ----------
    input_dim    : per-frame input feature dim (190 for landmarks, 384 for
                   DINOv2-S, 575 for DINOv2+landmarks concat, etc.)
    vocab_size   : CTC output dim (28 for English WiTA)
    d_model      : Conformer hidden dim
    n_layers     : number of Conformer blocks
    n_heads      : MHSA heads
    conv_kernel  : Conformer conv kernel size (must be odd)
    ff_mult      : FFN expansion factor
    dropout      : applied throughout
    upsample     : temporal upsample factor on the encoder output
                   T_out = upsample × T_in.  For WiTA with L_max ≈ 14 and
                   T_in = 32, upsample=2 gives T_out=64 which satisfies
                   2L+1 = 29 with healthy margin.

    Returns from forward
    --------------------
    ctc_log_probs : [B, T_out, V]
    enc_lens      : [B] int — valid T_out per sample
    """

    def __init__(
        self,
        input_dim:  int,
        vocab_size: int,
        d_model:    int = 256,
        n_layers:   int = 4,
        n_heads:    int = 4,
        conv_kernel: int = 15,
        ff_mult:    int = 4,
        dropout:    float = 0.1,
        upsample:   int = 2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.d_model    = d_model
        self.upsample   = upsample
        self.vocab_size = vocab_size

        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d=d_model, n_heads=n_heads, conv_kernel=conv_kernel,
                ff_mult=ff_mult, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Temporal upsampler (transposed conv).  We use kernel=stride so the
        # output length is exactly upsample × input length and the upsampling
        # is non-overlapping (cleaner gradient than overlapping ConvTranspose).
        if upsample > 1:
            self.up = nn.ConvTranspose1d(
                d_model, d_model,
                kernel_size=upsample, stride=upsample,
            )
        else:
            self.up = nn.Identity()

        self.head_ln  = nn.LayerNorm(d_model)
        self.head_fc  = nn.Linear(d_model, vocab_size)

        logger.info(
            "[ConformerCTC] in=%d d=%d L=%d heads=%d kernel=%d upsample=%d V=%d",
            input_dim, d_model, n_layers, n_heads, conv_kernel, upsample,
            vocab_size,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        x:          torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder-only forward — returns the pre-upsample Conformer output.
        Used by the Stage 1 v3 signer-adversarial head, which needs to
        branch off the encoder representation (not the CTC log-probs).

        Returns
        -------
        h        : [B, T_in, d_model]
        pad_mask : [B, T_in] bool — True at PAD positions (None if
                   input_lens is None)
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
        B, T_in, _ = x.shape
        if input_lens is None:
            input_lens = torch.full((B,), T_in, dtype=torch.long, device=x.device)

        arange = torch.arange(T_in, device=x.device).unsqueeze(0)
        pad_mask = arange >= input_lens.unsqueeze(1)        # True at pad

        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)
        return h, pad_mask

    def decode_ctc(
        self,
        h:          torch.Tensor,
        input_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder-only forward — takes pre-upsample encoder features and
        produces CTC log-probs.  Lets the trainer share one encoder pass
        between the CTC objective and the signer-adversarial head.
        """
        h = h.transpose(1, 2)                              # [B, d, T_in]
        h = self.up(h)                                     # [B, d, T_out]
        h = h.transpose(1, 2)                              # [B, T_out, d]
        h = self.head_ln(h)
        logits = self.head_fc(h)
        log_probs = F.log_softmax(logits, dim=-1)
        T_out = h.shape[1]
        enc_lens = (input_lens.long() * self.upsample).clamp(max=T_out)
        return log_probs, enc_lens.to(torch.int32)

    def forward(
        self,
        x:          torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x           : [B, T_in, input_dim] features
        input_lens  : [B] int — valid T_in per sample (for the trainer's
                      CTCLoss).  May be None; defaults to T_in for all.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
        B, T_in, _ = x.shape
        if input_lens is None:
            input_lens = torch.full((B,), T_in, dtype=torch.long, device=x.device)

        # Build padding mask for MHSA (True at PAD).
        arange = torch.arange(T_in, device=x.device).unsqueeze(0)   # [1, T_in]
        pad_mask = arange >= input_lens.unsqueeze(1)                # [B, T_in]

        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)
        # h: [B, T_in, d]

        # Upsample along T
        h = h.transpose(1, 2)                          # [B, d, T_in]
        h = self.up(h)                                 # [B, d, T_out]
        h = h.transpose(1, 2)                          # [B, T_out, d]

        h = self.head_ln(h)
        logits = self.head_fc(h)                       # [B, T_out, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # Output lengths: input_lens × upsample, clamped to T_out
        T_out = h.shape[1]
        enc_lens = (input_lens.long() * self.upsample).clamp(max=T_out)
        return log_probs, enc_lens.to(torch.int32)

    # --- decoder ---------------------------------------------------------

    @torch.no_grad()
    def decode_ctc_greedy(
        self,
        x:          torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
        blank:      int = 0,
    ) -> list[torch.Tensor]:
        """
        Greedy argmax + standard CTC merge (collapse repeats, drop blanks).
        Returns a list of [L_i] int tensors.
        """
        log_probs, enc_lens = self.forward(x, input_lens)
        argmax = log_probs.argmax(dim=-1)              # [B, T_out]

        out: list[torch.Tensor] = []
        for b in range(argmax.shape[0]):
            seq = argmax[b, : int(enc_lens[b].item())].tolist()
            merged: list[int] = []
            prev: Optional[int] = None
            for tok in seq:
                if tok != prev and tok != blank:
                    merged.append(tok)
                prev = tok
            out.append(torch.tensor(merged, dtype=torch.int32))
        return out

    # --- trainer compatibility stub -------------------------------------

    def unfreeze_backbone(self) -> None:
        """No-op — this model has no separately-frozen backbone."""
        return

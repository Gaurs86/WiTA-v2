"""
models/attention_decoder.py — Transformer attention decoder for Stage 9.

Pairs with the Conformer encoder under joint CTC + attention training.
Same d_model as the encoder (256), but its own self+cross attention.

Vocab convention (uses the project's existing VocabConfig)
----------------------------------------------------------
The label space already reserves four special tokens after the character
indices in configs/default.py::VocabConfig:
    blank_idx        = 0
    chars            = 1 .. N     (e.g. a..z for English, N=26)
    sep_idx          = N + 1      (CTC repeat separator)
    sos_idx          = N + 2      (BOS for attention)
    eos_idx          = N + 3      (EOS for attention)
    pad_idx          = N + 4      (CE ignore_index)
    attn_vocab_size  = N + 5      (embedding size)
The attention decoder MUST be sized to attn_vocab_size so pad_idx fits.
An earlier version used ctc_vocab_size + 2 = N + 4, which made pad_idx
(N+4) out-of-range and crashed with a CUDA gather-kernel assertion.

Training is teacher-forced:
    decoder input :  [SOS, c_1, c_2, ..., c_N]
    decoder target:  [c_1, c_2, ..., c_N, EOS]
CrossEntropyLoss with ignore_index=pad_idx over the shifted target.

Greedy inference walks the decoder one step at a time until EOS or
`max_decode_len` (defaults to 2 × T_encoder, well above any WiTA label).
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
# Helpers
# ---------------------------------------------------------------------------

def _causal_mask(L: int, device) -> torch.Tensor:
    """Upper-triangular True mask for causal self-attention."""
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                      diagonal=1)


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))    # [1, max_len, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """Pre-LN block: self-attn (causal) → cross-attn (to encoder) → FFN."""

    def __init__(self, d_model: int, n_heads: int,
                 ff_mult: int, dropout: float):
        super().__init__()
        self.ln_self  = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt:           torch.Tensor,    # [B, L, d]
        memory:        torch.Tensor,    # [B, T, d]
        tgt_mask:      torch.Tensor,    # [L, L] bool, True at masked
        memory_pad:    Optional[torch.Tensor],  # [B, T] bool, True at PAD
    ) -> torch.Tensor:
        # Self-attn (causal).
        y = self.ln_self(tgt)
        y, _ = self.self_attn(y, y, y,
                              attn_mask=tgt_mask,
                              need_weights=False)
        tgt = tgt + self.drop(y)

        # Cross-attn (to encoder memory).
        y = self.ln_cross(tgt)
        y, _ = self.cross_attn(y, memory, memory,
                               key_padding_mask=memory_pad,
                               need_weights=False)
        tgt = tgt + self.drop(y)

        # FFN.
        y = self.ln_ff(tgt)
        y = self.ff(y)
        return tgt + self.drop(y)


# ---------------------------------------------------------------------------
# Attention decoder
# ---------------------------------------------------------------------------

class AttentionDecoder(nn.Module):
    """
    N-layer Transformer attention decoder.  Same d_model as the encoder.
    Outputs logits over the project's attn_vocab (configs/default.py::VocabConfig).

    Parameters
    ----------
    att_vocab_size : full attention vocab size, i.e. cfg.vocab.attn_vocab_size
    bos_idx        : SOS token, i.e. cfg.vocab.sos_idx
    eos_idx        : EOS token, i.e. cfg.vocab.eos_idx
    """

    def __init__(
        self,
        att_vocab_size: int,
        bos_idx:        int,
        eos_idx:        int,
        d_model:        int = 256,
        n_layers:       int = 3,
        n_heads:        int = 4,
        ff_mult:        int = 4,
        dropout:        float = 0.1,
        max_decode_len: int = 64,
    ):
        super().__init__()
        if not (0 <= bos_idx < att_vocab_size) or not (0 <= eos_idx < att_vocab_size):
            raise ValueError(
                f"BOS / EOS indices must be in [0, att_vocab_size).  "
                f"Got bos={bos_idx}, eos={eos_idx}, att_vocab_size={att_vocab_size}."
            )
        self.att_vocab_size = att_vocab_size
        self.bos            = bos_idx
        self.eos            = eos_idx
        self.d_model        = d_model
        self.max_decode_len = max_decode_len

        self.embed = nn.Embedding(self.att_vocab_size, d_model)
        self.pos   = _SinusoidalPositionalEncoding(d_model, max_len=max_decode_len + 4)
        self.blocks = nn.ModuleList([
            _DecoderBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, self.att_vocab_size)

        logger.info(
            "[AttentionDecoder] att_V=%d bos=%d eos=%d d=%d L=%d heads=%d",
            att_vocab_size, bos_idx, eos_idx, d_model, n_layers, n_heads,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        memory:     torch.Tensor,            # [B, T, d]
        memory_pad: Optional[torch.Tensor],  # [B, T] bool, True at PAD
        tgt_in:     torch.Tensor,            # [B, L] long, decoder input ids
    ) -> torch.Tensor:
        """
        Teacher-forced forward.  Returns logits [B, L, att_V].
        """
        L = tgt_in.size(1)
        x = self.embed(tgt_in)
        x = self.pos(x)
        causal = _causal_mask(L, x.device)
        for blk in self.blocks:
            x = blk(x, memory, causal, memory_pad)
        x = self.ln_out(x)
        return self.head(x)        # [B, L, att_V]

    # -- greedy decode -----------------------------------------------------

    @torch.no_grad()
    def greedy_decode(
        self,
        memory:     torch.Tensor,             # [B, T, d]
        memory_pad: Optional[torch.Tensor],   # [B, T] bool
        max_len:    Optional[int] = None,
    ) -> list[torch.Tensor]:
        """
        Greedy left-to-right decoding.  Stops a sequence the first time it
        emits EOS; clamps at `max_len` (defaults to self.max_decode_len).
        Returns a list of [L_i] int32 tensors (no BOS / EOS in output).
        """
        B = memory.size(0)
        device = memory.device
        max_len = max_len or self.max_decode_len

        tokens = torch.full(
            (B, 1), self.bos, dtype=torch.long, device=device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self.forward(memory, memory_pad, tokens)
            next_tok = logits[:, -1, :].argmax(dim=-1)              # [B]
            # Force-finished sequences keep emitting EOS so the matrix grows
            # cleanly; we'll truncate at the first EOS per-sequence later.
            next_tok = torch.where(
                finished, torch.full_like(next_tok, self.eos), next_tok,
            )
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos)
            if finished.all():
                break

        out: list[torch.Tensor] = []
        for b in range(B):
            seq = tokens[b, 1:].tolist()        # drop BOS
            # truncate at first EOS
            if self.eos in seq:
                seq = seq[: seq.index(self.eos)]
            out.append(torch.tensor(seq, dtype=torch.int32))
        return out


# ---------------------------------------------------------------------------
# Joint CTC + attention loss helper
# ---------------------------------------------------------------------------

def build_attention_targets(
    labels:     torch.Tensor,      # [B, L_max] long (CTC label ids)
    label_lens: torch.Tensor,      # [B] int
    bos:        int,
    eos:        int,
    pad:        int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Produce teacher-forced (decoder_in, decoder_target) tensors.

    decoder_in  [b, t] = BOS, c_1, c_2, ..., c_{L_b}      (length L_b + 1)
    decoder_tgt [b, t] = c_1, c_2, ..., c_{L_b}, EOS      (length L_b + 1)
    Padded to max(L_b)+1; padded positions filled with `pad`.
    """
    B = labels.size(0)
    Lmax = int(label_lens.max().item()) + 1     # +1 for the EOS / BOS slot
    device = labels.device

    dec_in = torch.full((B, Lmax), pad, dtype=torch.long, device=device)
    dec_tg = torch.full((B, Lmax), pad, dtype=torch.long, device=device)
    for b in range(B):
        Lb = int(label_lens[b].item())
        dec_in[b, 0] = bos
        if Lb > 0:
            dec_in[b, 1: 1 + Lb] = labels[b, :Lb].long()
            dec_tg[b, :Lb] = labels[b, :Lb].long()
        dec_tg[b, Lb] = eos
    return dec_in, dec_tg

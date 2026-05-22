"""
models/decoders/attention.py — Transformer attention decoder.

Standard seq2seq Transformer decoder for character-sequence generation.
Used as the second head in WiTAHybridModel alongside the CTC head.

Attention vocab extends the CTC vocab:
  0           : blank  (CTC)
  1 … N       : characters
  N+1 (SOS)   : start-of-sequence
  N+2 (EOS)   : end-of-sequence
  N+3 (PAD)   : padding (ignored in CrossEntropyLoss)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

from ...configs.default import AttnDecoderConfig, VocabConfig


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AttentionDecoder(nn.Module):
    """
    Transformer decoder for character sequence generation.

    Training (tgt_tokens provided):
        encoder_out   [B, T, D]
        tgt_tokens    [B, L]   — label with <sos> prepended
        tgt_pad_mask  [B, L]   — True = padded position
        → logits      [B, L, attn_vocab_size]

    Inference (tgt_tokens=None):
        encoder_out   [B, T, D]
        → token_ids   [B, seq_len]  greedy, stops at <eos>
    """

    def __init__(
        self,
        encoder_dim: int,
        vocab_cfg:   VocabConfig,
        attn_cfg:    AttnDecoderConfig,
    ):
        super().__init__()
        d_model       = encoder_dim
        vocab_size    = vocab_cfg.attn_vocab_size
        self.sos_idx  = vocab_cfg.sos_idx
        self.eos_idx  = vocab_cfg.eos_idx
        self.pad_idx  = vocab_cfg.pad_idx
        self.max_len  = attn_cfg.max_seq_len

        self.tgt_emb  = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_tgt  = PositionalEncoding(d_model, max_len=self.max_len + 4, dropout=attn_cfg.dropout)
        self.pos_mem  = PositionalEncoding(d_model, max_len=2048, dropout=attn_cfg.dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=attn_cfg.n_heads,
            dim_feedforward=attn_cfg.ff_dim,
            dropout=attn_cfg.dropout,
            batch_first=True,
            norm_first=True,     # pre-norm: stable for training from scratch
        )
        self.decoder     = nn.TransformerDecoder(layer, num_layers=attn_cfg.n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.tgt_emb.weight, std=0.02)
        if hasattr(self.tgt_emb, "padding_idx"):
            self.tgt_emb.weight.data[self.pad_idx].zero_()

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        encoder_out:  torch.Tensor,              # [B, T, D]
        tgt_tokens:   torch.Tensor,              # [B, L]
        tgt_pad_mask: torch.Tensor | None,       # [B, L]
    ) -> torch.Tensor:
        L      = tgt_tokens.shape[1]
        mem    = self.pos_mem(encoder_out)
        emb    = self.pos_tgt(self.tgt_emb(tgt_tokens))
        causal = self._causal_mask(L, encoder_out.device)
        out    = self.decoder(
            emb, mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.output_proj(out)             # [B, L, vocab_size]

    # ------------------------------------------------------------------
    # Greedy inference (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(self, encoder_out: torch.Tensor) -> torch.Tensor:
        B, device = encoder_out.shape[0], encoder_out.device
        mem  = self.pos_mem(encoder_out)
        tgt  = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_len):
            L      = tgt.shape[1]
            emb    = self.pos_tgt(self.tgt_emb(tgt))
            causal = self._causal_mask(L, device)
            out    = self.decoder(emb, mem, tgt_mask=causal)
            next_t = self.output_proj(out[:, -1]).argmax(-1, keepdim=True)
            tgt    = torch.cat([tgt, next_t], dim=1)
            done  |= next_t.squeeze(1).eq(self.eos_idx)
            if done.all():
                break

        return tgt[:, 1:]   # strip <sos>

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_out:  torch.Tensor,
        tgt_tokens:   torch.Tensor | None = None,
        tgt_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tgt_tokens is not None:
            return self.forward_train(encoder_out, tgt_tokens, tgt_pad_mask)
        return self.forward_inference(encoder_out)

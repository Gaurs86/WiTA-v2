"""
models/modules/recurrent.py — Optional recurrent head between encoder and CTC.

Provides a clean, swappable interface for:
  • BiLSTM / BiGRU  (via nn.LSTM / nn.GRU with pack/pad)
  • TransformerEncoder
  • Identity (pass-through)

Each variant accepts [B, T, D_in] and returns [B, T, D_out].
D_out depends on the arch:
  BiLSTM / BiGRU : 2 * hidden_size
  Transformer    : d_model (== hidden_size)
  None           : D_in (unchanged)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from configs.default import RecurrentConfig


class BiRNNHead(nn.Module):
    """
    Bidirectional LSTM or GRU.

    Input  : [B, T, D_in]  + seq_lens [B]
    Output : [B, T, 2*hidden_size]
    """
    def __init__(self, d_in: int, cfg: RecurrentConfig):
        super().__init__()
        cls = nn.LSTM if cfg.arch == "lstm" else nn.GRU
        self.rnn = cls(
            input_size=d_in,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = 2 * cfg.hidden_size

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        packed     = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _     = self.rnn(packed)
        out, _     = pad_packed_sequence(out, batch_first=True)
        return out   # [B, T, 2*hidden_size]


class TransformerEncoderHead(nn.Module):
    """
    Transformer encoder (self-attention over temporal dimension).

    Input  : [B, T, D_in]
    Output : [B, T, d_model]
    """
    def __init__(self, d_in: int, cfg: RecurrentConfig):
        super().__init__()
        d_model = cfg.hidden_size
        self.proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # Build key padding mask: True = padded position (ignore in attention)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=mask)


class IdentityHead(nn.Module):
    """Pass-through: no recurrent module."""
    def __init__(self, d_in: int, cfg: RecurrentConfig):
        super().__init__()
        self.out_dim = d_in

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# FC projection head (maps recurrent output → CTC logits)
# ---------------------------------------------------------------------------

class CTCProjection(nn.Module):
    """
    Linear projection from recurrent output to CTC vocab size.
    Mirrors GestureTranslator's fc1 + fc2 (or just fc2 when no recurrent).
    """
    def __init__(self, d_in: int, num_class: int, cfg: RecurrentConfig):
        super().__init__()
        if d_in == num_class:
            # Direct projection (no hidden fc)
            self.fc1 = None
            self.fc2 = nn.Linear(d_in, num_class)
        else:
            self.fc1 = nn.Linear(d_in, cfg.fc_hidden)
            self.fc2 = nn.Linear(cfg.fc_hidden, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1 is not None:
            x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_recurrent_head(
    d_in:      int,
    cfg:       RecurrentConfig,
) -> tuple[nn.Module, int]:
    """
    Returns (recurrent_module, output_dim).
    recurrent_module.forward(x, seq_lens) → [B, T, out_dim]
    """
    arch = cfg.arch.lower()
    if arch in ("lstm", "gru"):
        head = BiRNNHead(d_in, cfg)
    elif arch == "transformer":
        head = TransformerEncoderHead(d_in, cfg)
    elif arch == "none":
        head = IdentityHead(d_in, cfg)
    else:
        raise ValueError(f"Unknown recurrent arch '{arch}'")
    return head, head.out_dim

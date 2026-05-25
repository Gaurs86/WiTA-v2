"""
models/modules/recurrent.py — Optional recurrent head between encoder and CTC.

Provides a clean, swappable interface for:
  • BiLSTM / BiGRU  (via nn.LSTM / nn.GRU with pack/pad)
  • TransformerEncoder
  • Identity (pass-through — use when VideoMAE + Linear is sufficient)

Each variant accepts [B, T, D_in] and returns [B, T, D_out].
D_out depends on the arch:
  BiLSTM / BiGRU : 2 * hidden_size
  Transformer    : d_model (== hidden_size)
  None           : D_in (unchanged)

Bug D fix (from hybrid model)
------------------------------
CTCProjection previously used ReLU between fc1 and fc2, which can produce
dead neurons when CTC gradients are weak (early training, frozen backbone).
Changed to GELU: non-zero gradient everywhere, empirically better for
sequence-to-sequence projection heads (cf. BERT, ViT feed-forward layers).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...configs.default import RecurrentConfig


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
        packed = pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out   # [B, T, 2*hidden_size]


class TransformerEncoderHead(nn.Module):
    """
    Transformer encoder (self-attention over temporal dimension).

    Input  : [B, T, D_in]
    Output : [B, T, d_model]
    """
    def __init__(self, d_in: int, cfg: RecurrentConfig):
        super().__init__()
        d_model    = cfg.hidden_size
        self.proj  = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        layer      = nn.TransformerEncoderLayer(
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
        x    = self.proj(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        return self.encoder(x, src_key_padding_mask=mask)


class IdentityHead(nn.Module):
    """
    Pass-through: no recurrent processing.
    Useful for VideoMAE + direct Linear head (fastest, fewest parameters).
    """
    def __init__(self, d_in: int, cfg: RecurrentConfig):
        super().__init__()
        self.out_dim = d_in

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# CTC Projection head
# ---------------------------------------------------------------------------

class CTCProjection(nn.Module):
    """
    Two-layer projection: recurrent output → CTC logits.

    fc1 (Linear) → GELU → fc2 (Linear)

    Bug D fix: uses GELU instead of ReLU.
    ReLU can produce dead neurons when CTC gradients are sparse (early
    training, frozen backbone).  GELU has non-zero gradient everywhere
    and is consistent with the VideoMAE backbone's feed-forward layers.
    """

    def __init__(self, d_in: int, num_class: int, cfg: RecurrentConfig):
        super().__init__()
        if d_in == num_class:
            # Direct projection (no hidden layer needed)
            self.fc1 = None
            self.fc2 = nn.Linear(d_in, num_class)
        else:
            self.fc1 = nn.Linear(d_in, cfg.fc_hidden)
            self.fc2 = nn.Linear(cfg.fc_hidden, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1 is not None:
            x = F.gelu(self.fc1(x))   # ← GELU (was ReLU in hybrid version)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_recurrent_head(
    d_in: int,
    cfg:  RecurrentConfig,
) -> tuple[nn.Module, int]:
    """
    Build the recurrent head and return (module, output_dim).

    module.forward(x: [B, T, D_in], seq_lens: [B]) → [B, T, out_dim]
    """
    arch = cfg.arch.lower()
    if arch in ("lstm", "gru"):
        head = BiRNNHead(d_in, cfg)
    elif arch == "transformer":
        head = TransformerEncoderHead(d_in, cfg)
    elif arch == "none":
        head = IdentityHead(d_in, cfg)
    else:
        raise ValueError(
            f"Unknown recurrent arch '{arch}'. "
            f"Choose from: 'lstm', 'gru', 'transformer', 'none'."
        )
    return head, head.out_dim

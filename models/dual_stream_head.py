"""
models/dual_stream_head.py — Stage 4 fusion architectures.

Two variants per plan §2 Task C:

  EarlyFusion (single Conformer over concat features)
  ---------------------------------------------------
    landmark_feat: [T, 190]
    dino_feat:    [T, 1153]
    x = cat([landmark_feat, dino_feat], dim=-1)   # [T, 1343]
    x = LayerNorm(x); x = Linear(1343 → 256)(x)
    x = Conformer4(x)
    x = upsample(x); x = head(x)

  → reuses ConformerCTC(input_dim=1343, input_layernorm=True).  No new
    class needed; the EarlyFusion path lives in stage4_train.py.

  LateFusion (two parallel Conformers, concat outputs)
  ----------------------------------------------------
    x_l = LayerNorm(190);  Linear(190 → 128);  Conformer3_d128 -> [T, 128]
    x_d = LayerNorm(1153); Linear(1153 → 128); Conformer3_d128 -> [T, 128]
    x   = cat([x_l, x_d], dim=-1)                                # [T, 256]
    x   = upsample(x); x = head(x)

  → implemented here as `LateFusionConformerCTC`.

Contract reminder
-----------------
Stage 4 differs from Stage 3 only in the addition of the landmark stream.
Optimiser, schedule, seed, augmentation, ConvTranspose1d upsample, T_native,
CV folds — all identical to Stage 3.  See plan §5.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_ctc import ConformerBlock

logger = logging.getLogger(__name__)


class LateFusionConformerCTC(nn.Module):
    """
    Two parallel Conformer encoders, one per stream, concatenated at the
    end and projected to CTC log-probs.

    Stream shapes (defaults match the prompt):
      landmark stream :  in=190,  proj→128, 3-layer Conformer at d=128
      dino stream     :  in=1153, proj→128, 3-layer Conformer at d=128
      fused           :  cat([128,128]) = 256 → ConvTranspose1d upsample
                          → LayerNorm → Linear(256→V) → log_softmax

    Both proj_in modules use a leading LayerNorm so each stream's raw
    statistics are tamed independently before projection.

    Parameters
    ----------
    landmark_dim, dino_dim : per-frame input dim per stream.
    vocab_size             : CTC head output dim.
    d_per_stream           : hidden dim of each per-stream Conformer.
    n_layers_per_stream    : Conformer blocks per stream.
    fused_dim              : d_per_stream * 2 by default (concat).
    upsample               : ConvTranspose1d stride/kernel (T_out = upsample * T_in).
    """

    def __init__(
        self,
        landmark_dim:         int,
        dino_dim:             int,
        vocab_size:           int,
        *,
        d_per_stream:         int = 128,
        n_layers_per_stream:  int = 3,
        n_heads:              int = 4,
        conv_kernel:          int = 15,
        ff_mult:              int = 4,
        dropout:              float = 0.2,
        upsample:             int = 2,
    ):
        super().__init__()
        self.landmark_dim = landmark_dim
        self.dino_dim     = dino_dim
        self.upsample     = upsample
        self.vocab_size   = vocab_size
        fused_dim = 2 * d_per_stream

        self.proj_l = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, d_per_stream),
            nn.LayerNorm(d_per_stream),
        )
        self.proj_d = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, d_per_stream),
            nn.LayerNorm(d_per_stream),
        )
        self.stream_l = nn.ModuleList([
            ConformerBlock(d=d_per_stream, n_heads=n_heads,
                           conv_kernel=conv_kernel, ff_mult=ff_mult,
                           dropout=dropout)
            for _ in range(n_layers_per_stream)
        ])
        self.stream_d = nn.ModuleList([
            ConformerBlock(d=d_per_stream, n_heads=n_heads,
                           conv_kernel=conv_kernel, ff_mult=ff_mult,
                           dropout=dropout)
            for _ in range(n_layers_per_stream)
        ])

        if upsample > 1:
            self.up = nn.ConvTranspose1d(
                fused_dim, fused_dim,
                kernel_size=upsample, stride=upsample,
            )
        else:
            self.up = nn.Identity()

        self.head_ln = nn.LayerNorm(fused_dim)
        self.head_fc = nn.Linear(fused_dim, vocab_size)

        logger.info(
            "[LateFusionConformerCTC] landmark_dim=%d dino_dim=%d d=%d L_per=%d "
            "fused=%d upsample=%d V=%d",
            landmark_dim, dino_dim, d_per_stream, n_layers_per_stream,
            fused_dim, upsample, vocab_size,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        landmark_feats: torch.Tensor,    # [B, T, 190]
        dino_feats:     torch.Tensor,    # [B, T, 1153]
        input_lens:     Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if landmark_feats.shape[:2] != dino_feats.shape[:2]:
            raise ValueError(
                f"stream shape mismatch: landmark {landmark_feats.shape}, "
                f"dino {dino_feats.shape}"
            )
        B, T, _ = landmark_feats.shape
        device  = landmark_feats.device
        if input_lens is None:
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)

        arange   = torch.arange(T, device=device).unsqueeze(0)
        pad_mask = arange >= input_lens.unsqueeze(1)

        x_l = self.proj_l(landmark_feats)
        for blk in self.stream_l:
            x_l = blk(x_l, key_padding_mask=pad_mask)
        x_d = self.proj_d(dino_feats)
        for blk in self.stream_d:
            x_d = blk(x_d, key_padding_mask=pad_mask)

        x = torch.cat([x_l, x_d], dim=-1)              # [B, T, fused]
        x = x.transpose(1, 2)                          # [B, fused, T]
        x = self.up(x)                                 # [B, fused, T_out]
        x = x.transpose(1, 2)                          # [B, T_out, fused]
        x = self.head_ln(x)
        logits = self.head_fc(x)                       # [B, T_out, V]
        log_probs = F.log_softmax(logits, dim=-1)

        T_out = x.shape[1]
        enc_lens = (input_lens.long() * self.upsample).clamp(max=T_out)
        return log_probs, enc_lens.to(torch.int32)

    @torch.no_grad()
    def decode_ctc_greedy(
        self,
        landmark_feats: torch.Tensor,
        dino_feats:     torch.Tensor,
        input_lens:     Optional[torch.Tensor] = None,
        blank:          int = 0,
    ) -> list[torch.Tensor]:
        log_probs, enc_lens = self.forward(landmark_feats, dino_feats, input_lens)
        argmax = log_probs.argmax(dim=-1)
        out: list[torch.Tensor] = []
        for b in range(argmax.shape[0]):
            seq = argmax[b, : int(enc_lens[b].item())].tolist()
            merged: list[int] = []; prev: Optional[int] = None
            for tok in seq:
                if tok != prev and tok != blank:
                    merged.append(tok)
                prev = tok
            out.append(torch.tensor(merged, dtype=torch.int32))
        return out

"""
models/heads/clip_cross_modal_head.py — CLIP cross-modal CTC head.

Architecture
------------
visual features [B, T, D_v]   (D_v = SigLIP vision dim, e.g. 1152)
    ↓
temporal adapter  ('lstm' | 'conv' | 'transformer' | 'none')
    → [B, T, D_h]
    ↓
linear projection back to text-embed dim
    → [B, T, D_v]
    ↓
similarity vs frozen prototypes
    → [B, T, V]   logits = visual @ prototypes.T / tau
    ↓
log_softmax(-1)
    → ctc_log_probs [B, T, V]

Prototype building
------------------
At init, the head's caller passes a [V, D_v] tensor of frozen prototypes
obtained from SigLIP's text encoder applied to per-character prompt strings
(see build_prototypes() helper).  The prototypes are registered as a buffer
so they ride along with state_dict but never receive gradients.

Vocabulary layout
-----------------
Matches StrLabelConverter / VocabConfig:
  0          : CTC blank
  1 .. N     : characters
  N+1        : '-' consecutive-repeat separator
ctc_vocab_size = N+2

The blank and separator slots need prototypes too — see build_prototypes().
"""

from __future__ import annotations
import math
import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal adapter
# ---------------------------------------------------------------------------

class _TemporalAdapter(nn.Module):
    """
    Tiny per-clip temporal aggregator over frame features.

    Variants:
      lstm        : 1-layer BiLSTM, hidden = adapter_hidden (output 2*hidden)
      conv        : stack of 1D depthwise+pointwise convs over time
      transformer : 2-layer TransformerEncoder, d_model = adapter_hidden
      none        : identity passthrough (just a linear projection)
    """

    def __init__(
        self,
        arch:            Literal["lstm", "conv", "transformer", "none"],
        in_dim:          int,
        adapter_hidden:  int = 512,
        num_layers:      int = 1,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.arch    = arch
        self.in_dim  = in_dim

        if arch == "lstm":
            self.lstm = nn.LSTM(
                input_size  = in_dim,
                hidden_size = adapter_hidden // 2,   # ×2 from bidirectional
                num_layers  = num_layers,
                bidirectional = True,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )
            self.out_dim = adapter_hidden
        elif arch == "conv":
            layers: list[nn.Module] = []
            ch = in_dim
            for _ in range(num_layers):
                layers += [
                    nn.Conv1d(ch, adapter_hidden, kernel_size=3,
                              padding=1, groups=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                ch = adapter_hidden
            self.conv = nn.Sequential(*layers)
            self.out_dim = adapter_hidden
        elif arch == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model         = in_dim,
                nhead           = 8,
                dim_feedforward = 4 * in_dim,
                dropout         = dropout,
                activation      = "gelu",
                batch_first     = True,
                norm_first      = True,
            )
            self.tx = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.out_dim = in_dim
        elif arch == "none":
            self.out_dim = in_dim
        else:
            raise ValueError(f"unknown adapter arch: {arch}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : [B, T, D]
        mask : [B, T]  bool, True at PAD positions (transformer only)
        ↩   : [B, T, out_dim]
        """
        if self.arch == "lstm":
            out, _ = self.lstm(x)
            return out
        if self.arch == "conv":
            # [B, T, D] → [B, D, T]
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return y
        if self.arch == "transformer":
            return self.tx(x, src_key_padding_mask=mask)
        return x


# ---------------------------------------------------------------------------
# CLIPCrossModalHead
# ---------------------------------------------------------------------------

class CLIPCrossModalHead(nn.Module):
    """
    Cross-modal CTC head: visual features → similarity logits over frozen
    text-derived character prototypes.

    Parameters
    ----------
    visual_dim     : input dim of frame features (SigLIP vision out, e.g. 1152)
    prototypes     : [V, visual_dim] tensor of frozen prototypes
                     (build with build_prototypes()).
    adapter_arch   : 'lstm' | 'conv' | 'transformer' | 'none'
    adapter_hidden : hidden dim inside the adapter
    init_tau       : initial value for temperature (log space).  CLIP-style
                     0.07 is the canonical inverse-temperature initialization;
                     we store log_tau so it stays positive.
    learnable_tau  : if True, log_tau is trainable; if False, frozen.
    """

    def __init__(
        self,
        visual_dim:     int,
        prototypes:     torch.Tensor,
        adapter_arch:   Literal["lstm", "conv", "transformer", "none"] = "lstm",
        adapter_hidden: int   = 512,
        adapter_layers: int   = 1,
        dropout:        float = 0.1,
        init_tau:       float = 0.07,
        learnable_tau:  bool  = True,
    ):
        super().__init__()
        if prototypes.dim() != 2 or prototypes.shape[1] != visual_dim:
            raise ValueError(
                f"prototypes must be [V, {visual_dim}], got {tuple(prototypes.shape)}"
            )
        V = prototypes.shape[0]
        self.visual_dim = visual_dim
        self.vocab_size = V

        # Frozen prototypes — buffer so it travels with state_dict.
        self.register_buffer("prototypes", prototypes.float().contiguous())

        # Temporal adapter
        self.adapter = _TemporalAdapter(
            arch           = adapter_arch,
            in_dim         = visual_dim,
            adapter_hidden = adapter_hidden,
            num_layers     = adapter_layers,
            dropout        = dropout,
        )

        # Project adapter output back to visual_dim so we can dot with prototypes.
        # Direct projection (no normalization) — the linear absorbs the scale.
        self.proj = nn.Linear(self.adapter.out_dim, visual_dim)

        # Temperature stored as log so exp() keeps it positive.
        # CLIP init: tau=0.07 → log(1/0.07)≈2.66 used as inv-tau; here we
        # multiply logits by 1/tau so we store inv_tau in log space.
        log_inv_tau = torch.tensor(math.log(1.0 / init_tau), dtype=torch.float32)
        if learnable_tau:
            self.log_inv_tau = nn.Parameter(log_inv_tau)
        else:
            self.register_buffer("log_inv_tau", log_inv_tau)

        logger.info(
            "[CLIPCrossModalHead] visual_dim=%d vocab=%d adapter=%s "
            "adapter_out=%d learnable_tau=%s",
            visual_dim, V, adapter_arch, self.adapter.out_dim, learnable_tau,
        )

    def forward(
        self,
        visual_feats: torch.Tensor,
        mask:         torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        visual_feats : [B, T, visual_dim]  (frozen SigLIP features)
        mask         : [B, T] bool, True at PAD positions  (transformer only)
        returns      : ctc_log_probs [B, T, V]
        """
        h = self.adapter(visual_feats, mask=mask)        # [B, T, H]
        z = self.proj(h)                                  # [B, T, D]
        # Logits: [B, T, D] @ [D, V] → [B, T, V]
        logits = z @ self.prototypes.t()
        logits = logits * self.log_inv_tau.exp()
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Prototype builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_prototypes(
    chars:        str,
    blank_idx:    int,
    sep_idx:      int,
    ctc_vocab_size: int,
    model_name:   str = "google/siglip-so400m-patch14-384",
    char_template:  str = "the letter {ch}",
    blank_template: str = "no character",
    sep_template:   str = "a brief pause between letters",
    device:       str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build a [ctc_vocab_size, D] tensor of prototypes via SigLIP text encoder.

    Layout (matches VocabConfig):
      idx 0           : blank (encoded from `blank_template`)
      idx 1..N        : chars (encoded from `char_template.format(ch=...)`)
      idx N+1         : separator (encoded from `sep_template`)

    Returns the prototypes as float32 on CPU.  Caller decides where to move them.
    """
    try:
        from transformers import SiglipTextModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("transformers>=4.40 required for SigLIP") from e

    text_model = SiglipTextModel.from_pretrained(model_name).to(device).eval()
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    prompts: list[str] = [""] * ctc_vocab_size
    prompts[blank_idx] = blank_template
    for i, ch in enumerate(chars):
        prompts[i + 1] = char_template.format(ch=ch)
    prompts[sep_idx]   = sep_template

    inputs = tokenizer(
        prompts, return_tensors="pt", padding="max_length", truncation=True,
    ).to(device)
    out = text_model(**inputs)
    feats = out.pooler_output  # [V, D]

    # Free GPU memory immediately — text model is single-use here.
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feats.float().cpu().contiguous()

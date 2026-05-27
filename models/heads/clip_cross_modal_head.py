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
    init_tau       : initial value for temperature.  In the L2-normalized
                     similarity setting, cosine sims are in [-1, 1] and we
                     multiply by 1/tau.  init_tau=1.0 → logit range ≈ [-1, 1]
                     (well-conditioned at init).  init_tau=0.07 (CLIP-style)
                     gives logit range ≈ [-14, +14], which is overly sharp
                     for a randomly-initialized projection layer and was the
                     cause of the Run-7 mode collapse.  Default is 1.0; allow
                     the learnable parameter to find a sharper value if needed.
    learnable_tau  : if True, log_inv_tau is trainable; if False, frozen.
    """

    def __init__(
        self,
        visual_dim:     int,
        prototypes:     torch.Tensor,
        adapter_arch:   Literal["lstm", "conv", "transformer", "none"] = "lstm",
        adapter_hidden: int   = 512,
        adapter_layers: int   = 1,
        dropout:        float = 0.1,
        init_tau:       float = 1.0,
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
        visual_feats : [B, T, visual_dim]  (frozen V-L features)
        mask         : [B, T] bool, True at PAD positions  (transformer only)
        returns      : ctc_log_probs [B, T, V]

        Numerical stability (Run-7 post-mortem fix)
        -------------------------------------------
        Both sides are L2-normalized before the dot product — this is how
        CLIP/SigLIP/X-CLIP were trained and how they are used at inference.
        Without normalization, the dot-product magnitude depends on the
        random init of `self.proj`, which combined with a sharp temperature
        (inv_tau ≈ 14 from init_tau=0.07) saturates the softmax at init →
        vanishing CTC gradient → mode collapse.

        With both sides L2-normed, the dot product lives in [-1, 1] and
        inv_tau controls a well-conditioned cosine similarity.
        """
        h = self.adapter(visual_feats, mask=mask)        # [B, T, H]
        z = self.proj(h)                                  # [B, T, D]

        # L2-normalize both sides → cosine similarity in [-1, 1].
        z_n     = F.normalize(z, dim=-1, eps=1e-6)
        proto_n = F.normalize(self.prototypes, dim=-1, eps=1e-6)   # [V, D]

        # Logits: [B, T, D] @ [D, V] → [B, T, V],  range [-inv_tau, +inv_tau]
        logits = z_n @ proto_n.t()
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
    model_name:   str = "google/siglip-so400m-patch14-224",
    char_template:  str = "the letter {ch}",
    blank_template: str = "no character",
    sep_template:   str = "a brief pause between letters",
    device:       str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build a [ctc_vocab_size, D] tensor of prototypes via the chosen model's
    text encoder.  Dispatches between SigLIP and X-CLIP based on model_name.

    Layout (matches VocabConfig):
      idx 0           : blank (encoded from `blank_template`)
      idx 1..N        : chars (encoded from `char_template.format(ch=...)`)
      idx N+1         : separator (encoded from `sep_template`)

    Supported backbones
    -------------------
      siglip   : "google/siglip-*"   → SiglipTextModel.pooler_output
      xclip    : "microsoft/xclip-*" → XCLIPModel.get_text_features (projected)

    Returns prototypes as float32 on CPU.  Caller decides where to move them.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("transformers>=4.40 required") from e

    is_xclip  = "xclip"  in model_name.lower()
    is_siglip = "siglip" in model_name.lower()

    # Build the per-class prompts (same for both backbones).
    prompts: list[str] = [""] * ctc_vocab_size
    prompts[blank_idx] = blank_template
    for i, ch in enumerate(chars):
        prompts[i + 1] = char_template.format(ch=ch)
    prompts[sep_idx]   = sep_template

    print(f"[build_prototypes] Loading text encoder: {model_name} "
          f"(first run downloads weights — be patient)", flush=True)

    # Same robust unwrapping used in the X-CLIP encoder — some transformers
    # versions return BaseModelOutputWithPooling from get_*_features instead
    # of a raw tensor.
    def _to_tensor(out):
        if torch.is_tensor(out):
            return out
        pooler = getattr(out, "pooler_output", None)
        if pooler is not None:
            return pooler
        last = getattr(out, "last_hidden_state", None)
        if last is not None:
            return last.mean(dim=1) if last.dim() == 3 else last
        if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
            return out[1]
        if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            return out[0]
        raise TypeError(
            f"Could not extract a tensor from text-encoder output of type "
            f"{type(out).__name__}"
        )

    if is_xclip:
        # X-CLIP exposes get_text_features on the full XCLIPModel only.
        from transformers import XCLIPModel
        full = XCLIPModel.from_pretrained(model_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        # Some transformers versions return BaseModelOutputWithPooling
        # instead of a tensor — handle both.
        try:
            raw = full.get_text_features(**inputs)
            feats = _to_tensor(raw)
        except Exception as e:
            # Manual fallback: text_model → projection.
            logger.warning(
                "get_text_features failed (%s) — using manual fallback.", e,
            )
            text_out = full.text_model(**inputs)
            text_pooled = _to_tensor(text_out)
            feats = full.text_projection(text_pooled)
        del full
    elif is_siglip:
        from transformers import SiglipTextModel
        text_model = SiglipTextModel.from_pretrained(model_name).to(device).eval()
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True,
        ).to(device)
        feats = _to_tensor(text_model(**inputs))     # [V, hidden_size]
        del text_model
    else:
        raise ValueError(
            f"Unsupported model_name for prototype building: {model_name}. "
            "Expected a SigLIP or X-CLIP HF model id."
        )

    print(f"[build_prototypes] Prototypes built: shape={tuple(feats.shape)}",
          flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feats.float().cpu().contiguous()

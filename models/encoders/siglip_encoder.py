"""
models/encoders/siglip_encoder.py — Frozen SigLIP visual encoder.

Wraps HF SiglipVisionModel as a per-frame visual encoder. Always frozen —
features are intended to be pre-extracted and cached. See
`datasets/feature_cache.py` for the extraction pipeline.

Input/output shapes
-------------------
forward(pixel_values):
  pixel_values : [B, 3, H, W]  (H=W=image_size, SigLIP-So400m/384 → 384×384)
                  float32, ALREADY normalized with SigLIP's mean/std (0.5/0.5/0.5)
  returns      : [B, D]        D=1152 for so400m, 768 for base, 1024 for large

forward_video(pixel_values):
  pixel_values : [B, T, 3, H, W]  — encodes per-frame, returns [B, T, D]

Normalization
-------------
SigLIP uses mean=std=0.5 (NOT ImageNet stats).  The encoder expects already-
normalized input.  Use the `default_normalize` function in this file or
HF's AutoProcessor to preprocess images outside this module.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SigLIP image normalization constants  (different from ImageNet)
# ---------------------------------------------------------------------------

SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD:  tuple[float, float, float] = (0.5, 0.5, 0.5)


def default_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a [0,1]-ranged tensor with SigLIP stats.
    Input  : [..., 3, H, W]  float, range [0, 1]
    Output : same shape, normalized to roughly [-1, 1]
    """
    mean = torch.as_tensor(SIGLIP_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std  = torch.as_tensor(SIGLIP_STD,  dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# SigLIPVisionEncoder
# ---------------------------------------------------------------------------

class SigLIPVisionEncoder(nn.Module):
    """
    Frozen SigLIP vision encoder.

    Parameters
    ----------
    model_name : HF Hub model ID (e.g. "google/siglip-so400m-patch14-384").
    image_size : expected H==W for input (must match the checkpoint, 384 for so400m).
    use_pooler : if True, returns SiglipVisionModel.pooler_output [B, D].
                 if False, returns mean-pooled last_hidden_state.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        image_size: int = 384,
        use_pooler: bool = True,
    ):
        super().__init__()
        try:
            from transformers import SiglipVisionModel
        except ImportError as e:
            raise ImportError(
                "transformers>=4.40 required for SigLIP. "
                "pip install --upgrade transformers"
            ) from e

        self.model_name = model_name
        self.image_size = image_size
        self.use_pooler = use_pooler

        self.backbone: nn.Module = SiglipVisionModel.from_pretrained(model_name)
        self.out_dim: int = self.backbone.config.hidden_size

        # Always frozen — caching expects deterministic outputs.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        logger.info(
            "[SigLIPVisionEncoder] %s loaded — out_dim=%d, image_size=%d, frozen",
            model_name, self.out_dim, image_size,
        )

    def train(self, mode: bool = True) -> "SigLIPVisionEncoder":
        # Keep backbone in eval mode regardless — frozen.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of frames.

        pixel_values : [B, 3, H, W]  pre-normalized with SigLIP stats
        returns      : [B, D]
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected [B, 3, H, W], got shape {tuple(pixel_values.shape)}"
            )
        out = self.backbone(pixel_values=pixel_values)
        if self.use_pooler and getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def forward_video(
        self,
        pixel_values: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode a video clip per-frame.

        pixel_values : [B, T, 3, H, W]  pre-normalized
        chunk_size   : how many frames to process at a time (memory control)
        returns      : [B, T, D]
        """
        if pixel_values.dim() != 5:
            raise ValueError(
                f"Expected [B, T, 3, H, W], got shape {tuple(pixel_values.shape)}"
            )
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * T, C, H, W)
        outs: list[torch.Tensor] = []
        for i in range(0, flat.shape[0], chunk_size):
            outs.append(self.forward(flat[i : i + chunk_size]))
        feats = torch.cat(outs, dim=0)
        return feats.reshape(B, T, self.out_dim)

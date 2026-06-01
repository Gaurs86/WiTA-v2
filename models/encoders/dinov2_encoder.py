"""
models/encoders/dinov2_encoder.py — Frozen DINOv2 per-frame encoder.

Stage 2 of the iterative ablation plan.  Wraps HF `facebook/dinov2-small`
as a per-frame feature extractor.  Always frozen — features are cached
once via `datasets/dinov2_cache.py` and the Conformer head trains on the
cache.

Why DINOv2-S/14 specifically
----------------------------
- Self-supervised ViT — no Kinetics/action-recognition baggage that
  poisoned X-CLIP for fine motion.
- Small: 22M params, ViT-S/14, hidden=384.  Fast enough that per-frame
  forward over the full 3477-clip dataset takes ~20-30 min on T4.
- Strong dense features (per-patch tokens carry fine spatial detail).
  Stage 2 mean-pools across patches; Stage 3 will use fingertip-region
  pooling instead.

Input/output shapes
-------------------
forward(pixel_values):
  pixel_values : [B, 3, H, W]   H=W=image_size (224), ImageNet-normalized
  returns      : [B, 384]       mean-pool over patch tokens (CLS excluded)

forward_clip(pixel_values):
  pixel_values : [T, 3, H, W]   single clip
  returns      : [T, 384]
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization constants (DINOv2 uses standard ImageNet stats)
# ---------------------------------------------------------------------------

DINOV2_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DINOV2_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)


def default_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a [0,1]-ranged tensor with ImageNet stats.
    Input  : [..., 3, H, W]
    Output : same shape
    """
    mean = torch.as_tensor(DINOV2_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std  = torch.as_tensor(DINOV2_STD,  dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Pooling modes
# ---------------------------------------------------------------------------

POOL_MEAN_PATCH = "mean_patch"   # Stage 2: mean over all 256 patch tokens
POOL_CLS        = "cls"          # alternative: CLS token only
POOL_MEAN_ALL   = "mean_all"     # mean over CLS + patches


# ---------------------------------------------------------------------------
# DINOv2VisionEncoder
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 per-frame encoder.

    Parameters
    ----------
    model_name : HF Hub model ID (default `facebook/dinov2-small`, 22M params,
                 384-d hidden).  Alternatives:
                   - facebook/dinov2-base  (86M, 768-d)
                   - facebook/dinov2-large (300M, 1024-d)
    image_size : input H==W (default 224; DINOv2 trained at 224 and also
                 supports 518 via positional embedding interpolation).
    pool       : how to reduce token dim to 384.
                   "mean_patch" — mean over patch tokens (Stage 2 default)
                   "cls"        — CLS token only
                   "mean_all"   — mean over CLS + patches
    """

    def __init__(
        self,
        model_name:       str = "facebook/dinov2-small",
        image_size:       int = 224,
        pool:             str = POOL_MEAN_PATCH,
        unfreeze_last_n:  int = 0,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers required for DINOv2. pip install transformers"
            ) from e

        self.model_name      = model_name
        self.image_size      = image_size
        self.pool            = pool
        self.unfreeze_last_n = int(unfreeze_last_n)

        print(f"[DINOv2Encoder] Loading: {model_name} "
              f"(first run downloads weights — be patient)", flush=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.out_dim: int = self.backbone.config.hidden_size

        # 1) Freeze everything by default.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2) Selectively unfreeze the last N transformer blocks if requested.
        #    HF DinoV2Model: backbone.encoder.layer is the list of blocks.
        if self.unfreeze_last_n > 0:
            layers = self.backbone.encoder.layer
            n_layers = len(layers)
            if self.unfreeze_last_n > n_layers:
                raise ValueError(
                    f"unfreeze_last_n={self.unfreeze_last_n} > total blocks "
                    f"({n_layers}) in {model_name}"
                )
            unfreeze_idx = list(range(n_layers - self.unfreeze_last_n, n_layers))
            for i in unfreeze_idx:
                for p in layers[i].parameters():
                    p.requires_grad = True
            # Also unfreeze the post-encoder LayerNorm so gradients can flow
            # cleanly out of the last block.
            if hasattr(self.backbone, "layernorm"):
                for p in self.backbone.layernorm.parameters():
                    p.requires_grad = True
            print(f"[DINOv2Encoder] Unfroze last {self.unfreeze_last_n} block(s): "
                  f"indices {unfreeze_idx} + post-LN", flush=True)
        else:
            # Fully frozen: legacy behaviour.  eval() prevents BN/dropout
            # surprises during inference-only use.
            self.backbone.eval()

        status = "frozen" if self.unfreeze_last_n == 0 else f"unfrozen last {self.unfreeze_last_n} blocks"
        print(
            f"[DINOv2Encoder] {model_name} loaded — "
            f"out_dim={self.out_dim}, image_size={image_size}, pool={pool}, {status}",
            flush=True,
        )
        logger.info(
            "[DINOv2Encoder] %s loaded — out_dim=%d, image_size=%d, pool=%s, %s",
            model_name, self.out_dim, image_size, pool, status,
        )

    def train(self, mode: bool = True) -> "DINOv2Encoder":
        super().train(mode)
        if self.unfreeze_last_n == 0:
            # Fully frozen path: keep backbone in eval to suppress dropout.
            self.backbone.eval()
        else:
            # Partial-unfreeze path: backbone follows the caller's mode so
            # dropout/LN behave correctly for the trainable blocks during
            # training, and revert to eval during validation.
            self.backbone.train(mode)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Per-frame encode.

        pixel_values : [B, 3, H, W]  ImageNet-normalized
        returns      : [B, hidden_size]

        Gradients flow when `unfreeze_last_n > 0`; otherwise wrap callers
        in `torch.no_grad()` for free speed.  (We removed the unconditional
        decorator so Stage 10 end-to-end training works.)
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected [B, 3, H, W], got {tuple(pixel_values.shape)}"
            )
        out = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, 1+N_patches, D]   index 0 = CLS
        tokens = out.last_hidden_state
        if self.pool == POOL_MEAN_PATCH:
            return tokens[:, 1:].mean(dim=1)
        if self.pool == POOL_CLS:
            return tokens[:, 0]
        if self.pool == POOL_MEAN_ALL:
            return tokens.mean(dim=1)
        raise ValueError(f"unknown pool: {self.pool}")

    def forward_clip(
        self,
        pixel_values: torch.Tensor,
        chunk_size:   int = 16,
    ) -> torch.Tensor:
        """
        Encode a single clip in [T, 3, H, W] → [T, hidden_size].
        Uses `chunk_size` per backbone call to bound activation memory.
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected [T, 3, H, W], got {tuple(pixel_values.shape)}"
            )
        T = pixel_values.shape[0]
        outs: list[torch.Tensor] = []
        for i in range(0, T, chunk_size):
            outs.append(self.forward(pixel_values[i : i + chunk_size]))
        return torch.cat(outs, dim=0)

    def forward_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return PER-PATCH token features (CLS excluded), un-pooled.

        Used by Stage 3 (fingertip-region pooling) and later stages that
        need spatial conditioning on the token grid.

        pixel_values : [B, 3, H, W]
        returns      : [B, N_patches, hidden_size]
                       where N_patches = (H // patch_size) ** 2 = 256 for
                       dinov2-small at 224×224 (patch14 → 16×16 grid).
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected [B, 3, H, W], got {tuple(pixel_values.shape)}"
            )
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state[:, 1:]    # drop CLS

    def forward_patches_clip(
        self,
        pixel_values: torch.Tensor,
        chunk_size:   int = 16,
    ) -> torch.Tensor:
        """
        Patch-token version of forward_clip.
        pixel_values : [T, 3, H, W]
        returns      : [T, N_patches, D]
        """
        T = pixel_values.shape[0]
        outs: list[torch.Tensor] = []
        for i in range(0, T, chunk_size):
            outs.append(self.forward_patches(pixel_values[i : i + chunk_size]))
        return torch.cat(outs, dim=0)

    @property
    def patch_size(self) -> int:
        """Spatial patch size (14 for dinov2-* models)."""
        return int(self.backbone.config.patch_size)

    @property
    def grid_size(self) -> int:
        """Token grid side: image_size // patch_size (16 for 224/14)."""
        return self.image_size // self.patch_size

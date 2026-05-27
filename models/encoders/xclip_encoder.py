"""
models/encoders/xclip_encoder.py — Frozen X-CLIP video-language encoder.

Wraps HF XCLIPModel as a sliding-window video encoder.  Unlike SigLIP which
encodes a single frame, X-CLIP encodes a SHORT CLIP (8 frames by default)
and applies temporal multi-frame attention internally — so each "feature"
captures motion across those 8 frames.

Pipeline per full clip
----------------------
clip [T_raw, 3, H, W]   (e.g. 64 raw frames @ 224×224)
    ↓ slide window size=8, stride=4
segments [N_seg, 8, 3, H, W]   (N_seg = (T_raw - 8) / 4 + 1)
    ↓ X-CLIP get_video_features per segment
features [N_seg, projection_dim]   (projection_dim=512 for xclip-base)

The N_seg features are then fed to the cross-modal head as a temporal
sequence (analogous to T'=16 with VideoMAE).  CTC operates over these.

Why X-CLIP for WiTA
-------------------
Per-frame SigLIP features all encoded the same scene ("person sitting") with
no motion information — that's why training mode-collapsed.  X-CLIP's
multi-frame attention forces each segment feature to encode motion within
that segment, recovering the fingertip-trajectory signal that's actually
discriminative for WiTA.

Normalization
-------------
X-CLIP processor uses CLIP's mean/std (NOT ImageNet, NOT SigLIP).  The
encoder expects already-normalized input.  Use the `default_normalize`
function in this file or HF's XCLIPProcessor to preprocess.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# X-CLIP / CLIP image normalization constants
# ---------------------------------------------------------------------------

XCLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275,  0.40821073)
XCLIP_STD:  tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def default_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a [0,1]-ranged tensor with X-CLIP (CLIP) stats.
    Input  : [..., 3, H, W]  float, range [0, 1]
    Output : same shape, X-CLIP-normalized
    """
    mean = torch.as_tensor(XCLIP_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std  = torch.as_tensor(XCLIP_STD,  dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# XCLIPVideoEncoder
# ---------------------------------------------------------------------------

class XCLIPVideoEncoder(nn.Module):
    """
    Frozen X-CLIP video encoder with sliding-window segmentation.

    Parameters
    ----------
    model_name  : HF Hub model ID (e.g. "microsoft/xclip-base-patch16-zero-shot").
    image_size  : expected H==W for input (224 for all base/large X-CLIP).
    num_frames  : frames per X-CLIP segment (8 for base, must match model config).
    stride      : frame stride between consecutive windows.  Smaller stride →
                  more output segments → more CTC timesteps but more compute.
                  Recommended: 4 (50% overlap, gives ~15 segs for 64-frame clip).
    """

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        image_size: int = 224,
        num_frames: int = 8,
        stride:     int = 4,
    ):
        super().__init__()
        try:
            from transformers import XCLIPModel
        except ImportError as e:
            raise ImportError(
                "transformers>=4.40 required for X-CLIP. "
                "pip install --upgrade transformers"
            ) from e

        self.model_name = model_name
        self.image_size = image_size
        self.num_frames = num_frames
        self.stride     = stride

        print(f"[XCLIPVideoEncoder] Loading: {model_name} "
              f"(first run downloads weights — be patient)", flush=True)
        # Full XCLIPModel — we need it for get_video_features which lives on
        # the joint model (not on XCLIPVisionModel).
        self.backbone = XCLIPModel.from_pretrained(model_name)
        # Projected video feature dim — what get_video_features returns.
        self.out_dim: int = self.backbone.config.projection_dim

        # Verify the checkpoint matches our segmentation assumptions.
        ckpt_frames = getattr(self.backbone.config.vision_config, "num_frames", None)
        if ckpt_frames is not None and ckpt_frames != num_frames:
            logger.warning(
                "X-CLIP checkpoint expects num_frames=%d but encoder configured "
                "for %d. Adjusting to checkpoint value.",
                ckpt_frames, num_frames,
            )
            self.num_frames = ckpt_frames

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        print(
            f"[XCLIPVideoEncoder] {model_name} loaded — "
            f"projection_dim={self.out_dim}, num_frames={self.num_frames}, "
            f"stride={self.stride}, frozen",
            flush=True,
        )
        logger.info(
            "[XCLIPVideoEncoder] %s loaded — out_dim=%d, num_frames=%d, stride=%d, frozen",
            model_name, self.out_dim, self.num_frames, self.stride,
        )

    def train(self, mode: bool = True) -> "XCLIPVideoEncoder":
        # Keep backbone in eval mode regardless — frozen.
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------ #
    # Segmentation                                                       #
    # ------------------------------------------------------------------ #

    def segment_window_starts(self, T_raw: int) -> list[int]:
        """
        Compute the start indices of valid X-CLIP windows over a T_raw-frame
        clip.  Each window is num_frames long, advanced by stride.

        For shorter clips we always emit at least one window starting at 0
        (with frame-wise padding handled by the caller).
        """
        if T_raw <= self.num_frames:
            return [0]
        last_start = T_raw - self.num_frames
        starts = list(range(0, last_start + 1, self.stride))
        if starts[-1] != last_start:
            starts.append(last_start)   # ensure tail is covered
        return starts

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_tensor(out) -> torch.Tensor:
        """
        Extract a [B, D] tensor from whatever XCLIPModel.get_video_features
        returns.  Different transformers versions return either:
          - torch.Tensor directly                  (canonical)
          - BaseModelOutputWithPooling             (some versions)
          - tuple (last_hidden_state, pooled, ...) (return_dict=False path)
        """
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
            f"Could not extract a tensor from XCLIPModel.get_video_features "
            f"output of type {type(out).__name__}"
        )

    @torch.no_grad()
    def encode_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Encode a single batch of segments through X-CLIP.

        segment : [B, num_frames, 3, H, W]  X-CLIP-normalized
        returns : [B, projection_dim]
        """
        if segment.dim() != 5:
            raise ValueError(
                f"Expected [B, num_frames, 3, H, W], got {tuple(segment.shape)}"
            )
        try:
            out = self.backbone.get_video_features(pixel_values=segment)
            return self._to_tensor(out)
        except Exception as e:
            # Manual fallback: run the X-CLIP pipeline ourselves.
            # vision_model → visual_projection → multiframe integration → pool.
            # This matches the canonical get_video_features implementation
            # in transformers/models/x_clip/modeling_x_clip.py.
            logger.warning(
                "get_video_features failed (%s) — falling back to manual pipeline.", e,
            )
            return self._manual_video_features(segment)

    @torch.no_grad()
    def _manual_video_features(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Manual X-CLIP video feature pipeline.  Mirrors HuggingFace's reference
        get_video_features but doesn't rely on its tensor-vs-output-object
        return contract.
        """
        B, T, C, H, W = segment.shape
        flat = segment.reshape(B * T, C, H, W)

        # 1. Per-frame ViT
        vis_out = self.backbone.vision_model(pixel_values=flat)
        vis_pooled = self._to_tensor(vis_out)        # [B*T, hidden]

        # 2. Visual projection to common dim
        projected = self.backbone.visual_projection(vis_pooled)  # [B*T, proj]

        # 3. Reshape and apply Multiframe Integration Transformer
        cls_feats = projected.view(B, T, -1)
        mit_out   = self.backbone.mit(cls_feats)
        return self._to_tensor(mit_out)              # [B, proj]

    @torch.no_grad()
    def encode_clip(
        self,
        clip:      torch.Tensor,
        valid_len: Optional[int] = None,
        chunk:     int = 4,
    ) -> torch.Tensor:
        """
        Encode a single clip into a sequence of segment features.

        clip      : [T_raw, 3, H, W]  X-CLIP-normalized
        valid_len : number of non-padding frames (clip[:valid_len] is real).
                    If None, treats the whole clip as valid.
        chunk     : how many segments to push through X-CLIP at once (memory knob).

        returns   : [N_seg, projection_dim]
        """
        if clip.dim() != 4:
            raise ValueError(f"Expected [T, 3, H, W], got {tuple(clip.shape)}")

        T_raw = clip.shape[0] if valid_len is None else min(valid_len, clip.shape[0])
        T_full = clip.shape[0]
        starts = self.segment_window_starts(T_raw)

        # Build all segments by slicing.  If a window extends past T_full
        # (only possible when T_raw <= num_frames < T_full), repeat last frame.
        segments: list[torch.Tensor] = []
        for s in starts:
            e = s + self.num_frames
            if e <= T_full:
                seg = clip[s:e]
            else:
                # Repeat-pad with the last valid frame.
                pad = clip[T_raw - 1 : T_raw].expand(e - T_full + (T_full - s), *clip.shape[1:])
                seg = torch.cat([clip[s:T_full], pad[: e - T_full]], dim=0)
            segments.append(seg)
        x = torch.stack(segments, dim=0)   # [N_seg, num_frames, 3, H, W]

        # Chunk forward for memory.
        outs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], chunk):
            outs.append(self.encode_segment(x[i : i + chunk]))
        return torch.cat(outs, dim=0)      # [N_seg, projection_dim]

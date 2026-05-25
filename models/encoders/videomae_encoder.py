"""
models/encoders/videomae_encoder.py — VideoMAE & Video Swin backbones for WiTA.

Architecture overview
---------------------
Input clips [B, T, C, H, W]
    ↓  temporal resample → num_frames (16)
    ↓  spatial  resize   → img_size   (224)
    ↓  VideoMAE backbone
Patch tokens [B, N_total, hidden_size]        # e.g. [B, 1568, 768]
    ↓  mean-pool spatial patches per tube
Temporal features [B, T', hidden_size]        # T' = num_frames // tubelet_size = 8
    ↓  optional linear projection → cfg.out_dim
Output [B, T', out_dim]

Shape walkthrough (defaults: MCG-NJU/videomae-base, 16 frames, 224px)
-----------------------------------------------------------------------
Input          : [B, T,  3, H,   W  ]   e.g. [2, 64, 3, 224, 224]
After resample : [B, 16, 3, 224, 224]
After VideoMAE : [B, 1568, 768]          1568 = (16/2) * (224/16)^2 = 8 * 196
After pooling  : [B, 8,   768]           T' = 8  (temporal tokens only)
After proj     : [B, 8,   out_dim]       e.g. [B, 8, 768] if out_dim=768

Important: T' = 8 means CTC needs target lengths ≤ 8 characters.
           For longer sequences use cfg.videomae_num_frames=32 (→ T'=16)
           by fine-tuning with interpolated positional embeddings.

VideoSwinEncoder is an alternative that naturally supports longer temporal
sequences via 3D shifted windows (pytorchvideo dependency).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configs.default import EncoderConfig


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

try:
    from transformers import VideoMAEModel
    from transformers import VideoMAEConfig as HFVideoMAEConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import pytorchvideo  # noqa: F401
    _HAS_PYTORCHVIDEO = True
except ImportError:
    _HAS_PYTORCHVIDEO = False


# ---------------------------------------------------------------------------
# VideoMAE Encoder
# ---------------------------------------------------------------------------

class VideoMAEEncoder(nn.Module):
    """
    HuggingFace VideoMAEModel → temporal feature sequence [B, T', D].

    Parameters (all from EncoderConfig)
    ------------------------------------
    videomae_model_name : HF model hub ID (default: "MCG-NJU/videomae-base")
    videomae_num_frames : how many frames to resample clips to (default: 16)
    tubelet_size        : tube height in frames (default: 2, matches pretraining)
    patch_size          : spatial patch size in pixels (default: 16)
    img_size            : spatial resolution fed to backbone (default: 224)
    pretrained          : load pretrained Hub weights (default: True)
    freeze_backbone     : freeze backbone parameters (default: True)
    out_dim             : output feature dim; identity if == hidden_size

    T' computation
    --------------
    T' = videomae_num_frames // tubelet_size
       = 16 // 2 = 8  (default)

    Increase videomae_num_frames (e.g. to 32) for more temporal resolution,
    but be aware that VideoMAE was pretrained on 16 frames; you may need to
    interpolate positional embeddings for other counts.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers>=4.26 required for VideoMAE. "
                "pip install 'transformers>=4.26'"
            )

        self.model_name    = cfg.videomae_model_name
        self.num_frames    = cfg.videomae_num_frames
        self.img_size      = cfg.img_size
        self.tubelet_size  = cfg.tubelet_size
        self.patch_size    = cfg.patch_size
        self.freeze        = cfg.freeze_backbone

        # ── Load or randomly-init backbone ──────────────────────────────────
        if cfg.pretrained:
            self.backbone = VideoMAEModel.from_pretrained(self.model_name)
            print(f"[VideoMAEEncoder] Loaded pretrained weights: {self.model_name}")
        else:
            hf_cfg        = HFVideoMAEConfig.from_pretrained(self.model_name)
            self.backbone = VideoMAEModel(hf_cfg)
            print(f"[VideoMAEEncoder] Random init (config from: {self.model_name})")

        # ── Derived dimensions ───────────────────────────────────────────────
        self.hidden_size = self.backbone.config.hidden_size
        self.T_prime     = self.num_frames // self.tubelet_size
        n_h              = self.img_size // self.patch_size
        self.n_spatial   = n_h * n_h                               # spatial patches per tube
        self.n_patches   = self.T_prime * self.n_spatial           # total patches

        print(
            f"[VideoMAEEncoder] num_frames={self.num_frames}, "
            f"T'={self.T_prime}, n_spatial={self.n_spatial}, "
            f"hidden_size={self.hidden_size}, out_dim={cfg.out_dim}"
        )

        # ── Freeze backbone if requested ─────────────────────────────────────
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("[VideoMAEEncoder] Backbone parameters FROZEN")
        else:
            print("[VideoMAEEncoder] Backbone parameters trainable")

        # ── Optional output projection ───────────────────────────────────────
        if cfg.out_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, cfg.out_dim)
            print(f"[VideoMAEEncoder] Projection {self.hidden_size} → {cfg.out_dim}")
        else:
            self.proj = nn.Identity()

        self.out_dim = cfg.out_dim

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _resample_temporal(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sample self.num_frames from clips along the time axis.

        Uses torch.linspace index selection (nearest-neighbour) to avoid
        any trainable parameters or gradient flow through the resampling.

        clips : [B, T, C, H, W]
        """
        B, T, C, H, W = clips.shape
        if T == self.num_frames:
            return clips
        idx = torch.linspace(0, T - 1, self.num_frames,
                              device=clips.device).long()  # [num_frames]
        return clips[:, idx]  # [B, num_frames, C, H, W]

    def _resize_spatial(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Bilinearly resize (H, W) → (img_size, img_size).
        Merges batch and time dims for a single interpolation call.

        clips : [B, T, C, H, W]
        """
        B, T, C, H, W = clips.shape
        if H == self.img_size and W == self.img_size:
            return clips
        x = clips.view(B * T, C, H, W)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        return x.view(B, T, C, self.img_size, self.img_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips   : [B, T, C, H, W]  raw input (any T, any spatial res)
        returns : [B, T', out_dim]  temporal feature sequence
        """
        print(f"[DEBUG VideoMAEEncoder] input  shape: {clips.shape}")

        # 1. Temporal resampling → [B, num_frames, C, H, W]
        clips = self._resample_temporal(clips)
        # 2. Spatial resize      → [B, num_frames, C, img_size, img_size]
        clips = self._resize_spatial(clips)

        print(f"[DEBUG VideoMAEEncoder] preprocessed shape: {clips.shape}")

        # 3. VideoMAE backbone
        #    HF convention: pixel_values [B, T, C, H, W]  (matches our layout)
        outputs = self.backbone(pixel_values=clips)
        hidden  = outputs.last_hidden_state               # [B, N_total, D]

        print(f"[DEBUG VideoMAEEncoder] backbone output shape: {hidden.shape}")

        # 4. Pool spatial patches per temporal tube → [B, T', D]
        B, N, D = hidden.shape
        n_spatial = N // self.T_prime                     # re-derive at runtime
        temporal_feat = hidden.view(B, self.T_prime, n_spatial, D).mean(dim=2)

        print(f"[DEBUG VideoMAEEncoder] temporal features shape: {temporal_feat.shape}")
        print(f"[DEBUG VideoMAEEncoder] T'={temporal_feat.shape[1]}")

        # 5. Optional projection
        temporal_feat = self.proj(temporal_feat)          # [B, T', out_dim]

        return temporal_feat


# ---------------------------------------------------------------------------
# Video Swin Encoder (pytorchvideo alternative)
# ---------------------------------------------------------------------------

class VideoSwinEncoder(nn.Module):
    """
    Video Swin Transformer backbone via pytorchvideo.

    Outputs [B, T', D] by applying AdaptiveAvgPool over the spatial
    dimensions of the Swin feature map, preserving the temporal dimension.

    Requires: pip install pytorchvideo

    Model names (passed as cfg.videomae_model_name):
      "swin_t"   → Swin-T Kinetics-400  (out_dim≈768 after pool)
      "swin_s"   → Swin-S
      "swin_b"   → Swin-B

    NOTE: Video Swin processes clips of arbitrary length. The temporal
    downsampling factor is 2x (from the stem), so T' = T_in // 2 (approx).
    This gives higher temporal resolution than VideoMAE (T'=8) and is
    better suited for longer sign-language clips.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if not _HAS_PYTORCHVIDEO:
            raise ImportError(
                "pytorchvideo required for Video Swin. "
                "pip install pytorchvideo"
            )

        from pytorchvideo.models import create_model  # type: ignore

        model_name = cfg.videomae_model_name if cfg.videomae_model_name else "swin_t"
        self.swin  = create_model(
            model_name,
            pretrained=cfg.pretrained,
            head=None,               # Remove classification head
        )

        if cfg.freeze_backbone:
            for p in self.swin.parameters():
                p.requires_grad_(False)
            print(f"[VideoSwinEncoder] Backbone FROZEN ({model_name})")

        # Spatial pool to get temporal-only features
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Probe the actual output channel count
        _test  = torch.zeros(1, 3, 8, 224, 224)
        with torch.no_grad():
            _out = self.swin(_test)               # [1, C, T', 1, 1] or similar
        _C     = _out.shape[1]
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.proj = (
            nn.Linear(_C, cfg.out_dim) if _C != cfg.out_dim else nn.Identity()
        )
        self.out_dim = cfg.out_dim
        print(f"[VideoSwinEncoder] backbone_dim={_C}, out_dim={cfg.out_dim}")

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips   : [B, T, C, H, W]
        returns : [B, T', out_dim]
        """
        print(f"[DEBUG VideoSwinEncoder] input shape: {clips.shape}")

        # Video Swin expects [B, C, T, H, W]
        x = clips.permute(0, 2, 1, 3, 4)                # [B, C, T, H, W]
        x = self.swin(x)                                 # [B, C', T', H', W']

        print(f"[DEBUG VideoSwinEncoder] swin output shape: {x.shape}")

        # Pool spatial → [B, C', T', 1, 1] → [B, T', C']
        x = self.spatial_pool(x)
        x = x.squeeze(-1).squeeze(-1)                   # [B, C', T']
        x = x.permute(0, 2, 1)                          # [B, T', C']

        print(f"[DEBUG VideoSwinEncoder] temporal features shape: {x.shape}")

        x = self.proj(x)                                 # [B, T', out_dim]
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_video_encoder(cfg: EncoderConfig) -> nn.Module:
    """Dispatch to VideoMAEEncoder or VideoSwinEncoder based on cfg.arch."""
    arch = cfg.arch.lower()
    if arch == "videomae":
        return VideoMAEEncoder(cfg)
    elif arch == "video_swin":
        return VideoSwinEncoder(cfg)
    else:
        raise ValueError(
            f"build_video_encoder: unknown arch '{arch}'. "
            f"Expected 'videomae' or 'video_swin'."
        )

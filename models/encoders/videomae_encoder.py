"""
models/encoders/videomae_encoder.py — VideoMAE & Video Swin backbones for WiTA.

Architecture overview
---------------------
Input clips [B, T, C, H, W]
    ↓  temporal resample + spatial resize
    ↓  backbone (VideoMAE or Video Swin)
Temporal features [B, T', D]
    ↓  optional linear projection → cfg.out_dim
Output [B, T', out_dim]

VideoMAEEncoder (arch="videomae")
----------------------------------
Shape walkthrough (defaults: MCG-NJU/videomae-base, 32 frames, 224px)
    Input          : [B, T,  3, H,   W  ]   e.g. [2, 64, 3, 224, 224]
    After resample : [B, 32, 3, 224, 224]
    After VideoMAE : [B, 1568, 768]          1568 = (32/2) * (224/16)^2 = 16 * 196
    After pooling  : [B, 16, 768]            T' = 16
    After proj     : [B, 16, out_dim]

VideoSwinEncoder (arch="video_swin")
-------------------------------------
Uses torchvision.models.video.swin3d_t/s/b — no pytorchvideo required.
Shape walkthrough (defaults: swin_t, 32 frames, 224px)
    Input          : [B, T,  3, H,   W  ]   e.g. [2, 64, 3, 224, 224]
    After resample : [B, 32, 3, 224, 224]
    After permute  : [B, 3,  32, 224, 224]  (BCTHW for torchvision)
    After backbone : [B, 768, 16, 7, 7]     T'=32//2=16, spatial=224//32=7
    After pool     : [B, 16, 768]            spatial mean → temporal sequence
    After proj     : [B, 16, out_dim]

Both encoders produce T'=16 for num_frames=32 — identical CTC input length.
The caching pipeline (cache_features.py) works with both architectures
unchanged since both expose the same forward(clips) → [B, T', out_dim] API.
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

        # ── Positional embedding interpolation ───────────────────────────────
        # VideoMAE-base was pretrained on 16 frames → 1568 pos embeddings
        # (T'_pretrain=8, n_spatial=196).  Any other num_frames requires the
        # temporal dimension of the position embedding table to be resized.
        # Without this, forward() crashes with a tensor-size mismatch.
        pretrain_frames = self.backbone.config.num_frames          # typically 16
        if self.num_frames != pretrain_frames:
            self._interpolate_pos_embed(pretrain_frames)
            # Patch the config so HF's internal shape-checks also pass
            self.backbone.config.num_frames = self.num_frames

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
    # Positional embedding interpolation
    # ------------------------------------------------------------------

    def _interpolate_pos_embed(self, pretrain_frames: int) -> None:
        """
        Resize the backbone's temporal positional embeddings so they match
        self.num_frames, enabling fine-tuning at a different frame count
        than what the model was pretrained on.

        Layout of VideoMAE position embeddings
        ----------------------------------------
        pos_embed : [1, T'_pretrain * n_spatial, hidden_size]
        e.g. pretrained 16 fr → [1, 1568, 768]  (T'=8, n_spatial=196)

        Strategy
        ---------
        1. Reshape to [1, T'_pretrain, n_spatial, hidden_size]
        2. Permute  to [1, hidden_size, T'_pretrain, n_spatial]
        3. F.interpolate (bilinear) along dim-2 to T'_new
        4. Permute back and flatten → [1, T'_new * n_spatial, hidden_size]
        5. Replace as nn.Parameter (frozen/trainable matches backbone state)

        This is identical to the interpolation used in the official VideoMAE
        fine-tuning scripts (util/pos_embed.py → interpolate_pos_embed).
        """
        T_prime_old = pretrain_frames // self.tubelet_size   # e.g. 8
        T_prime_new = self.T_prime                           # e.g. 16 for 32 frames

        if T_prime_old == T_prime_new:
            return   # nothing to do

        pe_old = self.backbone.embeddings.position_embeddings.data
        # pe_old : [1, T'_old * n_spatial, D]
        D = pe_old.shape[-1]

        # Reshape → [1, D, T'_old, n_spatial] for interpolation
        pe = (
            pe_old
            .reshape(1, T_prime_old, self.n_spatial, D)   # [1, T'_old, S, D]
            .permute(0, 3, 1, 2)                           # [1, D, T'_old, S]
            .float()
        )

        # Interpolate temporal dimension only
        pe_new = F.interpolate(
            pe,
            size=(T_prime_new, self.n_spatial),
            mode="bilinear",
            align_corners=False,
        )                                                  # [1, D, T'_new, S]

        # Back to [1, T'_new * n_spatial, D]
        pe_new = (
            pe_new
            .permute(0, 2, 3, 1)                          # [1, T'_new, S, D]
            .reshape(1, T_prime_new * self.n_spatial, D)  # [1, T'_new*S, D]
            .to(dtype=self.backbone.embeddings.position_embeddings.dtype)
        )

        # Replace as a parameter; requires_grad matches whatever freeze_backbone set
        self.backbone.embeddings.position_embeddings = nn.Parameter(
            pe_new,
            requires_grad=self.backbone.embeddings.position_embeddings.requires_grad,
        )

        print(
            f"[VideoMAEEncoder] Positional embeddings interpolated: "
            f"T'={T_prime_old} ({pretrain_frames} frames) → "
            f"T'={T_prime_new} ({self.num_frames} frames)  "
            f"shape: {list(pe_old.shape)} → {list(pe_new.shape)}"
        )

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
        # 1. Temporal resampling → [B, num_frames, C, H, W]
        clips = self._resample_temporal(clips)
        # 2. Spatial resize      → [B, num_frames, C, img_size, img_size]
        clips = self._resize_spatial(clips)

        # 3. VideoMAE backbone
        #    HF convention: pixel_values [B, T, C, H, W]  (matches our layout)
        outputs = self.backbone(pixel_values=clips)
        hidden  = outputs.last_hidden_state               # [B, N_total, D]

        # 4. Pool spatial patches per temporal tube → [B, T', D]
        B, N, D = hidden.shape
        n_spatial = N // self.T_prime                     # re-derive at runtime
        temporal_feat = hidden.view(B, self.T_prime, n_spatial, D).mean(dim=2)

        # 5. Optional projection
        temporal_feat = self.proj(temporal_feat)          # [B, T', out_dim]

        return temporal_feat


# ---------------------------------------------------------------------------
# Video Swin Encoder (torchvision backend — no pytorchvideo required)
# ---------------------------------------------------------------------------

# Supported arch variants: name → (factory_fn, weights_cls, backbone_dim)
_SWIN_VARIANTS: dict[str, tuple] = {
    "swin_t": ("swin3d_t", "Swin3D_T_Weights", 768),
    "swin_s": ("swin3d_s", "Swin3D_S_Weights", 768),
    "swin_b": ("swin3d_b", "Swin3D_B_Weights", 1024),
}


class VideoSwinEncoder(nn.Module):
    """
    Video Swin Transformer backbone (torchvision ≥ 0.15) → [B, T', D].

    Uses torchvision.models.video.swin3d_t/s/b — no pytorchvideo dependency.
    Pretrained weights: Kinetics-400 (KINETICS400_V1).

    Architecture flow
    -----------------
    Input clips [B, T, C, H, W]
        ↓  temporal resample → video_swin_num_frames  (default 32)
        ↓  spatial  resize   → video_swin_img_size    (default 224)
        ↓  permute → [B, C, T, H, W]  (torchvision convention)
        ↓  swin3d backbone (avgpool replaced with Identity)
    Feature map [B, D, T', H', W']   D=768, T'=16, H'=W'=7 for swin_t/32fr/224px
        ↓  spatial mean pool → [B, D, T']
        ↓  permute → [B, T', D]
        ↓  optional linear projection → cfg.out_dim
    Output [B, T', out_dim]

    Shape walkthrough (swin_t defaults: 32 frames, 224px)
    -------------------------------------------------------
    Input          : [B, T,  3, H,   W  ]   any T, any spatial
    After resample : [B, 32, 3, 224, 224]
    After permute  : [B, 3,  32, 224, 224]  (torchvision BCTHW)
    After backbone : [B, 768, 16, 7, 7]     T'=32//2=16, spatial=224//32=7
    After mean pool: [B, 768, 16]
    After permute  : [B, 16, 768]           → CTC input sequence
    After proj     : [B, 16, out_dim]

    T' = video_swin_num_frames // video_swin_patch_size[0]
       = 32 // 2 = 16  (same as VideoMAE with 32 frames — cache compatible)

    Requirement: (video_swin_num_frames // 2) % video_swin_window_size[0] == 0
                 Default: 16 % 8 == 0  ✓

    Unfreeze interface
    ------------------
    self.backbone  — the swin3d model (matches VideoMAEEncoder attribute name
                     so hybrid_model.unfreeze_backbone() works unchanged)
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()

        try:
            import torchvision.models.video as tvm  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchvision >= 0.15 is required for VideoSwinEncoder. "
                "pip install 'torchvision>=0.15'"
            )

        arch = cfg.video_swin_arch.lower()
        if arch not in _SWIN_VARIANTS:
            raise ValueError(
                f"VideoSwinEncoder: unknown video_swin_arch='{arch}'. "
                f"Choose from {list(_SWIN_VARIANTS.keys())}."
            )

        self.num_frames  = cfg.video_swin_num_frames
        self.img_size    = cfg.video_swin_img_size
        self.patch_t     = cfg.video_swin_patch_size[0]   # temporal stride = 2
        self.T_prime     = self.num_frames // self.patch_t # e.g. 32 // 2 = 16
        self.freeze      = cfg.freeze_backbone

        factory_name, weights_cls_name, backbone_dim = _SWIN_VARIANTS[arch]

        # ── Validate window size divides T' evenly ───────────────────────
        window_t = cfg.video_swin_window_size[0]
        if self.T_prime % window_t != 0:
            raise ValueError(
                f"VideoSwinEncoder: T'={self.T_prime} "
                f"(num_frames={self.num_frames} // patch_t={self.patch_t}) "
                f"is not divisible by window_size[0]={window_t}. "
                f"Adjust video_swin_num_frames so that "
                f"(num_frames // {self.patch_t}) % {window_t} == 0."
            )

        # ── Build backbone ───────────────────────────────────────────────
        import torchvision.models.video as tvm
        factory_fn  = getattr(tvm, factory_name)
        weights_cls = getattr(tvm, weights_cls_name)

        if cfg.pretrained:
            weights = getattr(weights_cls, "KINETICS400_V1")
            self.backbone = factory_fn(
                weights=weights,
            )
            print(f"[VideoSwinEncoder] Loaded Kinetics-400 pretrained: {arch}")
        else:
            self.backbone = factory_fn(
                weights=None,
            )
            print(f"[VideoSwinEncoder] Random init: {arch}")

        # ── Neutralise the classification head ──────────────────────────
        # Replace avgpool with spatial-preserving pool so temporal dim survives.
        # Replace flatten + head with Identity to get raw feature maps out.
        #
        # torchvision Swin3D forward (simplified):
        #   x = patch_embed(x)           # NDHWC
        #   for layer in layers: x = layer(x)
        #   x = norm(x)                  # NDHWC
        #   x = x.permute(0,4,1,2,3)    # → NCDHW
        #   x = avgpool(x)               # [B, C, 1, 1, 1] normally
        #   x = flatten(x)               # [B, C]
        #   x = head(x)                  # [B, num_classes]
        #
        # After our overrides backbone returns [B, D, T', H', W']:
        self.backbone.avgpool  = nn.Identity()
        self.backbone.flatten  = nn.Identity()
        self.backbone.head     = nn.Identity()

        # ── Freeze backbone if requested ─────────────────────────────────
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print(f"[VideoSwinEncoder] Backbone parameters FROZEN")
        else:
            print(f"[VideoSwinEncoder] Backbone parameters trainable")

        # ── Optional output projection ───────────────────────────────────
        if cfg.out_dim != backbone_dim:
            self.proj = nn.Linear(backbone_dim, cfg.out_dim)
            print(f"[VideoSwinEncoder] Projection {backbone_dim} → {cfg.out_dim}")
        else:
            self.proj = nn.Identity()

        self.out_dim      = cfg.out_dim
        self.backbone_dim = backbone_dim

        print(
            f"[VideoSwinEncoder] arch={arch}, num_frames={self.num_frames}, "
            f"T'={self.T_prime}, backbone_dim={backbone_dim}, out_dim={cfg.out_dim}"
        )

    # ------------------------------------------------------------------
    # Preprocessing helpers  (mirrors VideoMAEEncoder interface)
    # ------------------------------------------------------------------

    def _resample_temporal(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sample self.num_frames from clips along the time axis.
        clips : [B, T, C, H, W]
        """
        B, T, C, H, W = clips.shape
        if T == self.num_frames:
            return clips
        idx = torch.linspace(0, T - 1, self.num_frames,
                             device=clips.device).long()
        return clips[:, idx]

    def _resize_spatial(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Bilinearly resize (H, W) → (img_size, img_size).
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
        # 1. Temporal resampling → [B, num_frames, C, H, W]
        clips = self._resample_temporal(clips)
        # 2. Spatial resize      → [B, num_frames, C, img_size, img_size]
        clips = self._resize_spatial(clips)

        # 3. torchvision expects [B, C, T, H, W]
        x = clips.permute(0, 2, 1, 3, 4).contiguous()   # [B, C, T, H, W]

        # 4. Backbone forward
        #    With avgpool/flatten/head = Identity, returns [B, D, T', H', W']
        x = self.backbone(x)                             # [B, D, T', H', W']

        # 5. Spatial mean pool → [B, D, T']
        x = x.mean(dim=[-2, -1])                         # [B, D, T']

        # 6. Rearrange to sequence format → [B, T', D]
        x = x.permute(0, 2, 1)                           # [B, T', D]

        # 7. Optional projection
        x = self.proj(x)                                  # [B, T', out_dim]

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

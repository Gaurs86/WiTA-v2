"""
models/stage12_model.py — Stage 12 composite: VideoMAE+LoRA backbone
spatially-mean-pooled to tube features, temporally upsampled to T=32,
concatenated with the existing landmark stream, projected, then the
Stage-9a Conformer encoder + joint CTC + attention decoder head.

Reuses unchanged from prior stages:
  ConformerCTC.encode + ConformerCTC.decode_ctc  (encoder + CTC head + upsample)
  AttentionDecoder                              (joint CE branch)

New here:
  VideoMAELoRA              wraps MCG-NJU/videomae-base + LoRA adapters.
  Stage12FeatureFusion      projects landmarks, concatenates with video,
                            mean-pools VideoMAE spatial patches, temporally
                            upsamples to match landmarks.
  Stage12Model              full forward returning (ctc_log_probs, attn_logits)

The backbone is trainable only on its LoRA params plus the post-LN; the
rest of VideoMAE is frozen.  ~1.5M trainable / 86M total.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_ctc       import ConformerCTC
from .attention_decoder   import AttentionDecoder, build_attention_targets

logger = logging.getLogger(__name__)


# ImageNet normalisation constants used by VideoMAE.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class VideoMAELoRA(nn.Module):
    """
    VideoMAE-base with LoRA adapters on the attention q/k/v projections.

    Forward
    -------
    pixel_values : [B, T=16, 3, H=224, W=224]   ImageNet-normalised float32 (or fp16)
    returns      : [B, T_tubes=8, D=768]        spatially mean-pooled tube features

    All non-LoRA params are frozen; ~1.5M trainable.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        lora_r:     int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        try:
            from transformers import VideoMAEModel
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "Stage 12 needs `pip install transformers peft` (and `torchvision` "
                "for normalisation constants)."
            ) from e
        self.model_name = model_name
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        for p in self.backbone.parameters():
            p.requires_grad = False
        # IMPORTANT: use `inject_adapter_in_model` rather than
        # `get_peft_model(...)`.  The PeftModel wrapper's forward
        # explicitly passes input_ids=None / attention_mask=None to the
        # base model -- VideoMAEModel.forward doesn't accept either,
        # so wrapping it raises TypeError on the first forward pass.
        # In-place injection inserts LoRA layers into q/k/v while
        # leaving the backbone's class + forward signature untouched.
        target = ["query", "key", "value"]
        lcfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target, bias="none",
            # task_type intentionally omitted -- this is a vision encoder.
        )
        inject_adapter_in_model(lcfg, self.backbone)
        # Unfreeze the post-encoder LN so gradients can flow cleanly out.
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        print(f"[VideoMAELoRA] trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)", flush=True)
        self.out_dim = self.backbone.config.hidden_size
        # Tube spatial grid: 14x14 for 224/16; tube temporal: T/2 for T=16.
        self.spatial_patches  = (self.backbone.config.image_size // self.backbone.config.patch_size) ** 2
        self.temporal_tubes_for_t16 = 16 // self.backbone.config.tubelet_size

    @staticmethod
    def normalise(video_uint8: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] uint8 -> ImageNet-normalised float (fp32 or whatever amp wants)."""
        x = video_uint8.float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, video_uint8: torch.Tensor) -> torch.Tensor:
        """
        video_uint8 : [B, T=16, 3, 224, 224] uint8
        returns     : [B, T_tubes=8, D=768]  spatially mean-pooled
        """
        if video_uint8.dim() != 5:
            raise ValueError(f"expected [B, T, 3, H, W], got {tuple(video_uint8.shape)}")
        x = self.normalise(video_uint8)
        out = self.backbone(pixel_values=x).last_hidden_state    # [B, T_tubes*P, D]
        B, N, D = out.shape
        # N = T_tubes * spatial_patches (e.g. 8 * 196 = 1568 for VideoMAE-base T=16).
        spatial = self.spatial_patches
        if N % spatial != 0:
            raise RuntimeError(
                f"last_hidden_state has {N} tokens but spatial_patches={spatial} "
                f"(model_name={self.model_name}).  Check tubelet_size + patch_size."
            )
        t_tubes = N // spatial
        out = out.view(B, t_tubes, spatial, D).mean(dim=2)        # [B, t_tubes, D]
        return out


# ---------------------------------------------------------------------------
# Fusion stack
# ---------------------------------------------------------------------------

class Stage12FeatureFusion(nn.Module):
    """
    VideoMAE tubes (upsampled from T_tubes -> T_native) + landmarks ->
    [B, T_native, d_model] for the Conformer.

    Inputs
    ------
    video_features    : [B, T_tubes, D_video=768]
    landmark_features : [B, T_native, D_landmark=190]

    Output: [B, T_native, d_model]
    """

    def __init__(
        self,
        d_video:    int = 768,
        d_landmark: int = 190,
        d_model:    int = 256,
        d_landmark_proj: int = 128,
        T_native:   int = 32,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.T_native = T_native
        self.landmark_proj = nn.Sequential(
            nn.LayerNorm(d_landmark),
            nn.Linear(d_landmark, d_landmark_proj),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_in = d_video + d_landmark_proj
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        video_features:    torch.Tensor,        # [B, T_tubes, D_v]
        landmark_features: torch.Tensor,        # [B, T_native, D_l]
    ) -> torch.Tensor:
        # Upsample video to T_native frames (linear in time).
        v = video_features.transpose(1, 2)
        v = F.interpolate(v, size=self.T_native, mode="linear", align_corners=False)
        v = v.transpose(1, 2)                                       # [B, T_native, D_v]
        # Project landmarks.
        l = self.landmark_proj(landmark_features)                   # [B, T_native, D_lp]
        x = torch.cat([v, l], dim=-1)                               # [B, T_native, D_v + D_lp]
        return self.fusion_proj(x)                                  # [B, T_native, d_model]


# ---------------------------------------------------------------------------
# Full Stage 12 model
# ---------------------------------------------------------------------------

class Stage12Model(nn.Module):
    """
    Composite: VideoMAE+LoRA -> fusion -> Conformer -> CTC head + attention
    decoder head.
    """

    def __init__(
        self,
        ctc_vocab_size:    int,
        attn_vocab_size:   int,
        sos_idx:           int,
        eos_idx:           int,
        *,
        videomae_model_name: str = "MCG-NJU/videomae-base",
        lora_r:        int = 16,
        lora_alpha:    int = 32,
        lora_dropout:  float = 0.1,
        d_model:       int = 256,
        n_layers:      int = 4,
        n_heads:       int = 4,
        conv_kernel:   int = 15,
        dropout:       float = 0.2,
        upsample:      int = 2,
        T_native:      int = 32,
        dec_n_layers:  int = 2,
        dec_n_heads:   int = 4,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone = VideoMAELoRA(
            model_name=videomae_model_name,
            lora_r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.fusion = Stage12FeatureFusion(
            d_video=self.backbone.out_dim,
            d_landmark=190, d_model=d_model,
            T_native=T_native, dropout=dropout,
        )
        # ConformerCTC takes a [B, T, d_model] input; we already projected.
        # We pass d_model as its input_dim so its proj_in is just a LN+identity-ish.
        self.encoder = ConformerCTC(
            input_dim       = d_model,
            vocab_size      = ctc_vocab_size,
            d_model         = d_model,
            n_layers        = n_layers,
            n_heads         = n_heads,
            conv_kernel     = conv_kernel,
            dropout         = dropout,
            upsample        = upsample,
            input_layernorm = False,   # fusion already LN'd the input
        )
        self.decoder = AttentionDecoder(
            att_vocab_size = attn_vocab_size,
            bos_idx        = sos_idx,
            eos_idx        = eos_idx,
            d_model        = d_model,
            n_layers       = dec_n_layers,
            n_heads        = dec_n_heads,
            ff_mult        = 4,
            dropout        = dropout,
        )

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fused_features(
        self,
        video:    torch.Tensor,        # [B, T=16, 3, 224, 224] uint8
        landmark: torch.Tensor,        # [B, T_native=32, 190]
    ) -> torch.Tensor:
        v = self.backbone(video)                              # [B, T_tubes, D_v]
        return self.fusion(v, landmark)                       # [B, T_native, d_model]

    def encode(
        self,
        video:      torch.Tensor,
        landmark:   torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
    ):
        feats = self.fused_features(video, landmark)
        h, pad_mask = self.encoder.encode(feats, input_lens)
        return h, pad_mask

    def forward(
        self,
        video:      torch.Tensor,
        landmark:   torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
        attn_tgt_in: Optional[torch.Tensor] = None,
    ):
        """
        Returns (ctc_log_probs, enc_lens, attn_logits_or_None, h, pad_mask).
        attn_logits is None if attn_tgt_in is None (e.g. at inference).
        """
        h, pad_mask = self.encode(video, landmark, input_lens)
        log_probs, enc_lens = self.encoder.decode_ctc(
            h, input_lens if input_lens is not None
            else torch.full((video.size(0),), h.size(1),
                            dtype=torch.long, device=h.device),
        )
        attn_logits = None
        if attn_tgt_in is not None:
            attn_logits = self.decoder(h, pad_mask, attn_tgt_in)
        return log_probs, enc_lens, attn_logits, h, pad_mask

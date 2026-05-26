"""
configs/default.py — WiTA v2 Configuration (VideoMAE + CTC refactor)

Changes from hybrid version
----------------------------
EncoderConfig
  • arch now includes "videomae" and "video_swin" (default: "videomae")
  • Added: videomae_model_name, videomae_num_frames, tubelet_size,
           patch_size, freeze_backbone
  • img_size moved here from DataConfig (VideoMAE needs 224; R3D used 112)
  • out_dim default changed to 768 (VideoMAE-B hidden size)
  • Added: video_swin_* fields for Video Swin Transformer (torchvision backend)
    - video_swin_arch          : "swin_t" | "swin_s" | "swin_b"
    - video_swin_num_frames    : frames to resample to (T' = num_frames // 2)
    - video_swin_img_size      : spatial resolution (224)
    - video_swin_patch_size    : (T, H, W) patch strides — (2, 4, 4) for all variants
    - video_swin_window_size   : (T, H, W) attention window
    - video_swin_drop_path_rate: stochastic depth rate
  • T_prime property now returns arch-correct temporal token count

TrainConfig
  • Removed: lambda_ctc_start, lambda_ctc_min, label_smoothing
             (attention-decoder-only fields)
  • lr default → 1e-4 (more conservative with large frozen backbone)
  • batch_size default → 1 (VideoMAE at 224px is VRAM-heavy)
  • Added: unfreeze_after_epoch — backbone unfreezing schedule

AttnDecoderConfig kept for reference but NOT used by WiTACTCModel.

VocabConfig layout (unchanged)
--------------------------------
0          : CTC blank
1 .. N     : characters
N+1        : '-' consecutive-repeat separator
N+2        : <SOS>  (attention only — unused in CTC model)
N+3        : <EOS>
N+4        : <PAD>
attn_vocab_size = N+5
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import os
import torch


# ---------------------------------------------------------------------------
# Language / Vocabulary
# ---------------------------------------------------------------------------

ALPHABET  = "abcdefghijklmnopqrstuvwxyz"
HANGUL    = (
    "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅃㅉㄸㄲㅆ"
    "ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅀㅄ"
    "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢᴥ "
)
ALPHA_HAN = ALPHABET + HANGUL

VOCAB_MAP: dict[str, str] = {
    "english": ALPHABET,
    "korean":  HANGUL,
    "both":    ALPHA_HAN,
}


@dataclass
class VocabConfig:
    """
    Vocabulary configuration.  Layout unchanged from hybrid version.

    Index layout
    ------------
    0          : CTC blank
    1 .. N     : characters
    N+1        : '-' consecutive-repeat separator  (StrLabelConverter)
    N+2        : <SOS>   (kept for evaluator/decoder compatibility)
    N+3        : <EOS>
    N+4        : <PAD>
    attn_vocab_size = N+5
    """
    lang: Literal["english", "korean", "both"] = "english"

    chars:           str = field(default="", init=False, repr=False)
    blank_idx:       int = field(default=0,  init=False)
    sep_idx:         int = field(default=0,  init=False)
    ctc_vocab_size:  int = field(default=0,  init=False)
    sos_idx:         int = field(default=0,  init=False)
    eos_idx:         int = field(default=0,  init=False)
    pad_idx:         int = field(default=0,  init=False)
    attn_vocab_size: int = field(default=0,  init=False)

    def build(self) -> "VocabConfig":
        self.chars           = VOCAB_MAP[self.lang]
        N                    = len(self.chars)
        self.blank_idx       = 0
        self.sep_idx         = N + 1
        self.ctc_vocab_size  = N + 2
        self.sos_idx         = N + 2
        self.eos_idx         = N + 3
        self.pad_idx         = N + 4
        self.attn_vocab_size = N + 5
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    hf_repo_id:    str             = "yewon816/WiTA"
    download_dir:  str             = "/kaggle/working/downloads"
    hf_cache_dir:  str             = os.path.expanduser("~/.cache/huggingface")
    max_zips:      Optional[int]   = None
    lang:          Literal["english", "korean", "both"] = "english"

    # NOTE: VideoMAE requires 224×224. If using R3D encoder, 112 is fine.
    # img_size is now in EncoderConfig; DataConfig.img_size is the resize
    # target BEFORE the encoder — set equal to EncoderConfig.img_size.
    img_size:      int             = 224    # changed from 112 for VideoMAE
    max_frames:    int             = 64
    img_mean:      tuple           = (0.485, 0.456, 0.406)
    img_std:       tuple           = (0.229, 0.224, 0.225)

    train_split:   float           = 0.90
    seed:          int             = 42


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    mirror_prob:     float = 0.30
    rotation_deg:    float = 5.0
    brightness:      float = 0.50
    contrast:        float = 0.50
    saturation:      float = 0.50
    hue:             float = 0.50
    grayscale_prob:  float = 0.10

    temporal_crop_ratio:  tuple = (0.75, 1.0)
    temporal_min_frames:  int   = 8
    drop_frames_prob:     float = 0.10


# ---------------------------------------------------------------------------
# Encoder  (VideoMAE / Video Swin / R3D)
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """
    Unified encoder config covering both VideoMAE and R3D backends.

    VideoMAE fields (used when arch in {"videomae", "video_swin"})
    --------------------------------------------------------------
    videomae_model_name : HF Hub model ID
    videomae_num_frames : frames to resample to before VideoMAE (default 16)
    tubelet_size        : temporal patch size in frames (default 2)
    patch_size          : spatial patch size in pixels (default 16)
    img_size            : spatial resolution fed to VideoMAE (default 224)
    freeze_backbone     : if True, backbone weights are frozen (recommended
                          for first ~5 epochs to stabilise the CTC head)

    R3D fields (used when arch in {"r3d", "mc3", "rmc3", "r2plus1d", "r2d"})
    -------------------------------------------------------------------------
    num_res_layers, pooling, track_running_stats  (unchanged from v1)

    Shared fields
    -------------
    out_dim    : output feature dim; VideoMAE default = 768 (ViT-B hidden)
    pretrained : load pretrained weights from Hub / torchvision
    """

    # ── Backbone selection ────────────────────────────────────────────────
    arch: Literal[
        "videomae", "video_swin",
        "r3d", "mc3", "rmc3", "r2plus1d", "r2d"
    ] = "videomae"

    # ── VideoMAE-specific ─────────────────────────────────────────────────
    videomae_model_name: str  = "MCG-NJU/videomae-base"
    videomae_num_frames: int  = 32      # frames resampled to → T' = 32//2 = 16
    tubelet_size:        int  = 2       # VideoMAE tube height → T' = num_frames//2
    patch_size:          int  = 16      # spatial patch size (ViT-style)
    img_size:            int  = 224     # spatial resolution for VideoMAE
    freeze_backbone:     bool = True    # freeze backbone initially

    # ── Video Swin-specific (torchvision backend, no pytorchvideo needed) ─
    #
    # video_swin_arch variants and their output dims:
    #   "swin_t"  → 768   (28M params, Kinetics-400 pretrained)  ← recommended
    #   "swin_s"  → 768   (49M params)
    #   "swin_b"  → 1024  (87M params)
    #
    # video_swin_patch_size (2, 4, 4):
    #   Temporal stride=2 → T' = video_swin_num_frames // 2
    #   For num_frames=32 → T'=16  (vs VideoMAE T'=16 at 32 frames — identical!)
    #
    # video_swin_window_size (8, 7, 7):
    #   T' (16) must be divisible by window_T (8) → 16 % 8 == 0  ✓
    #   If you change num_frames, verify: (num_frames // 2) % window_T == 0
    #
    # video_swin_drop_path_rate:
    #   Stochastic depth during fine-tuning. 0.2 = swin_t default.
    #   Backbone frozen → this is dormant during Phase 1 caching.
    #
    video_swin_arch:           str                    = "swin_t"
    video_swin_num_frames:     int                    = 32    # T' = 32 // 2 = 16
    video_swin_img_size:       int                    = 224
    video_swin_patch_size:     tuple                  = (2, 4, 4)   # (T, H, W)
    video_swin_window_size:    tuple                  = (8, 7, 7)   # (T, H, W)
    video_swin_drop_path_rate: float                  = 0.2

    # ── R3D-specific ──────────────────────────────────────────────────────
    num_res_layers:      int  = 1
    pooling:             Literal["average", "max"] = "average"
    track_running_stats: bool = True

    # ── Shared ───────────────────────────────────────────────────────────
    # VideoMAE-B hidden size = 768.  Video Swin-T/S = 768, Swin-B = 1024.
    # Set out_dim to project to a smaller dim (e.g. 512) or leave at 768
    # for an identity projection.
    out_dim:    int  = 768
    pretrained: bool = True

    @property
    def T_prime(self) -> int:
        """Number of temporal tokens produced by the encoder (arch-aware)."""
        if self.arch == "video_swin":
            return self.video_swin_num_frames // self.video_swin_patch_size[0]
        # VideoMAE (and R3D — approximate, actual T' depends on clip length)
        return self.videomae_num_frames // self.tubelet_size

    @property
    def effective_img_size(self) -> int:
        """Spatial resolution clips are resized to before the backbone."""
        if self.arch == "video_swin":
            return self.video_swin_img_size
        return self.img_size


# ---------------------------------------------------------------------------
# Recurrent head
# ---------------------------------------------------------------------------

@dataclass
class RecurrentConfig:
    arch:         Literal["lstm", "gru", "transformer", "none"] = "lstm"
    hidden_size:  int   = 256
    num_layers:   int   = 2
    nhead:        int   = 8
    ff_dim:       int   = 1024
    dropout:      float = 0.1
    fc_hidden:    int   = 256


# ---------------------------------------------------------------------------
# Attention Decoder  (kept for evaluator/decode compatibility; not trained)
# ---------------------------------------------------------------------------

@dataclass
class AttnDecoderConfig:
    n_layers:    int   = 4
    n_heads:     int   = 8
    ff_dim:      int   = 2048
    dropout:     float = 0.1
    max_seq_len: int   = 22


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    CTC-only training configuration.

    Key changes from hybrid version
    --------------------------------
    • batch_size default  → 1  (VideoMAE at 224px needs ~4–8 GB/sample)
    • lr default          → 1e-4  (lower: large frozen backbone + small head)
    • Removed: lambda_ctc_start/min, label_smoothing  (attention-only)
    • Added:   unfreeze_after_epoch  (unfreeze backbone after N epochs)
    """

    # ── Data loading ──────────────────────────────────────────────────────
    batch_size:   int  = 16         # safe with cached features on T4 (no backbone)
    accum_steps:  int  = 1          # no accumulation needed at batch=16

    num_workers:  int  = 2
    pin_memory:   bool = False
    persistent_workers: bool = True

    # ── Optimiser ─────────────────────────────────────────────────────────
    num_epochs:   int   = 40
    lr:           float = 1e-4      # lower than hybrid (large pretrained backbone)
    weight_decay: float = 1e-4
    beta1:        float = 0.90
    beta2:        float = 0.98
    grad_clip:    float = 5.0
    optimizer:    Literal["adamw", "adam", "sgd", "rmsprop", "lamb"] = "adamw"

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler:    Literal["onecycle", "warmup_multistep", "steplr", "none"] = "onecycle"
    warmup_pct:   float = 0.05
    final_div_factor: float = 300.0
    scheduler_gamma:  float = 0.1
    scheduler_step:   int   = 5

    # ── Backbone unfreezing schedule ─────────────────────────────────────
    # After this many epochs, backbone weights are unfrozen for fine-tuning.
    # Set to a large value (e.g. 999) to keep backbone frozen permanently.
    # With feature caching the backbone is never called during training,
    # so unfreeze_after_epoch is irrelevant — set high to avoid the call.
    unfreeze_after_epoch: int = 999

    # ── Logging / checkpointing ───────────────────────────────────────────
    log_interval:   int          = 10
    val_limit:      Optional[int] = 50
    qual_every_n:   int          = 5
    qual_n:         int          = 20

    checkpoint_dir:  str          = "/kaggle/working/checkpoints"
    resume_path:     Optional[str] = None
    save_frequency:  int          = 5

    seed: int = 42


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    vocab:     VocabConfig       = field(default_factory=VocabConfig)
    data:      DataConfig        = field(default_factory=DataConfig)
    aug:       AugConfig         = field(default_factory=AugConfig)
    encoder:   EncoderConfig     = field(default_factory=EncoderConfig)
    recurrent: RecurrentConfig   = field(default_factory=RecurrentConfig)
    attn:      AttnDecoderConfig = field(default_factory=AttnDecoderConfig)
    train:     TrainConfig       = field(default_factory=TrainConfig)

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def build(self) -> "Config":
        self.vocab.lang = self.data.lang
        self.vocab.build()
        # Keep DataConfig.img_size in sync with the arch-appropriate resolution.
        # effective_img_size returns video_swin_img_size for video_swin arch,
        # and img_size for VideoMAE / R3D — so this is always correct.
        self.data.img_size = self.encoder.effective_img_size
        return self

    def log_dir(self) -> str:
        return os.path.join(self.train.checkpoint_dir, "..", "logs")

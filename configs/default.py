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
    # Horizontal flip mirrors glyphs (b↔d, p↔q). Disabled for char recognition.
    mirror_prob:     float = 0.0
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
        "r3d", "mc3", "rmc3", "r2plus1d", "r2d",
        "siglip",
        "xclip",
    ] = "videomae"

    # ── X-CLIP video-language (used when arch == "xclip") ─────────────────
    # Sliding 8-frame windows are encoded by X-CLIP and cached.  Cross-modal
    # head uses X-CLIP text encoder to build per-character prototypes.
    # Unlike per-frame SigLIP, each segment feature encodes MOTION within
    # the 8 frames — this is the fix for the Run-7 mode collapse.
    # NOTE: do NOT use the "-zero-shot" variant — its checkpoint expects
    # num_frames=32, which the encoder auto-bumps to, making feature
    # extraction ~4× slower with no benefit for our small-adapter training.
    # patch16 = 14×14 spatial tokens (fine), patch32 = 7×7 (coarse but ~4× faster).
    xclip_model_name:     str   = "microsoft/xclip-base-patch16"
    xclip_num_frames:     int   = 8       # must match the checkpoint
    xclip_stride:         int   = 4       # sliding-window stride over the raw clip
    xclip_char_template:  str   = "writing the letter {ch} in the air"
    xclip_blank_template: str   = "no writing"
    xclip_sep_template:   str   = "a brief pause between letters"

    # ── Hand-region cropping (Path A) ─────────────────────────────────────
    # Crops to the writing hand using MediaPipe Hands BEFORE feeding to
    # X-CLIP.  Concentrates fingertip motion inside the 224×224 input so the
    # 16×16 patch tokens cover ~7×7 source pixels instead of ~16×16.
    # Used at both training AND deployment time — same code path either way.
    enable_hand_crop:        bool  = True
    hand_crop_padding:       float = 0.3   # extra margin around detected bbox
    hand_crop_min_conf:      float = 0.3   # MediaPipe detection threshold

    # The head fields are shared with SigLIP (siglip_temporal_arch etc.).

    # ── SigLIP cross-modal (used when arch == "siglip") ───────────────────
    # Frozen vision encoder + frozen text-derived character prototypes.
    # Only the temporal adapter + projection are trained.
    # See models/clip_ctc_model.py and models/heads/clip_cross_modal_head.py.
    #
    # Default model is the 224-input So400m variant — matches the WiTA frame
    # resolution (224×224) natively, no upscaling needed.
    # Alternatives:
    #   "google/siglip-base-patch16-224"     — smaller (87M, dim=768), faster
    #   "google/siglip-so400m-patch14-384"   — larger input, requires upscale
    #   "google/siglip2-so400m-patch14-224"  — newer SigLIP2 (needs transformers>=4.50)
    siglip_model_name:      str   = "google/siglip-so400m-patch14-224"
    siglip_char_template:   str   = "the letter {ch}"
    siglip_blank_template:  str   = "no character"
    siglip_sep_template:    str   = "a brief pause between letters"
    siglip_temporal_arch:   Literal["lstm", "conv", "transformer", "none"] = "lstm"
    # 256 (was 512) — full-data ablation on 3129 clips showed train loss
    # → 0.0003 at hidden=512, indicating ~10× more capacity than data supports.
    siglip_adapter_hidden:  int   = 256
    siglip_adapter_layers:  int   = 1
    # NOTE: PyTorch's nn.LSTM ignores `dropout` when num_layers=1. The dropout
    # field is wired through but only fires for conv/transformer adapters or
    # when num_layers>=2. Keep at 0.1 as a sane default.
    siglip_dropout:         float = 0.1
    # 1.0 (was 0.07): in the L2-normalized similarity setting (commit b78c54c)
    # cosine sims are in [-1,1], so inv_tau=1.0 keeps logits well-conditioned
    # at init. init_tau=0.07 → inv_tau≈14 was the cause of the Run-7 mode
    # collapse before normalization was added.
    siglip_init_tau:        float = 1.0
    siglip_learnable_tau:   bool  = True

    # ── VideoMAE-specific ─────────────────────────────────────────────────
    videomae_model_name: str  = "MCG-NJU/videomae-base"
    # T' = num_frames // tubelet_size. T'=8 (16 frames) is too short for many
    # English labels after sep-token insertion ("suggestion"→11 tokens), causing
    # CTCLoss to emit inf and be silenced by zero_infinity=True. T'=16 (32
    # frames) covers the dataset. Pos-embed interpolation handles the mismatch.
    videomae_num_frames: int  = 32      # frames resampled to (model's T input)
    tubelet_size:        int  = 2       # VideoMAE tube height → T' = 16//2 = 8
    patch_size:          int  = 16      # spatial patch size (ViT-style)
    img_size:            int  = 224     # spatial resolution for VideoMAE
    freeze_backbone:     bool = True    # freeze backbone initially

    # ── R3D-specific ──────────────────────────────────────────────────────
    num_res_layers:      int  = 1
    pooling:             Literal["average", "max"] = "average"
    track_running_stats: bool = True

    # ── Shared ───────────────────────────────────────────────────────────
    # VideoMAE-B hidden size = 768. If projecting to smaller dim set out_dim.
    out_dim:    int  = 768
    pretrained: bool = True

    @property
    def T_prime(self) -> int:
        """Number of temporal tokens produced by VideoMAE encoder."""
        return self.videomae_num_frames // self.tubelet_size


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
    # Effective batch = batch_size * accum_steps = 8 (unchanged from before).
    # batch_size=4 on 2× T4 with DataParallel splits to 2/GPU → ~3.5 GB/GPU
    # under AMP + gradient checkpointing, well inside the 15 GB budget.
    # Engages both GPUs and roughly halves wall-clock vs batch_size=1.
    batch_size:   int  = 4          # was 1; engages DataParallel on 2× T4
    accum_steps:  int  = 2          # was 8; keeps effective batch = 8

    num_workers:  int  = 2
    pin_memory:   bool = False
    persistent_workers: bool = True

    # ── Optimiser ─────────────────────────────────────────────────────────
    num_epochs:   int   = 40
    lr:           float = 1e-4      # lower than hybrid (large pretrained backbone)
    # 5e-4 (was 1e-4): the cross-modal head heavily overfits the small
    # WiTA dataset (3129 train clips) at the original 1e-4. Higher weight
    # decay slows train-loss collapse to 0 and may improve val CER.
    weight_decay: float = 5e-4
    beta1:        float = 0.90
    # VideoMAE/ViT fine-tuning recipe uses 0.999 (not 0.98).
    beta2:        float = 0.999
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
    unfreeze_after_epoch: int = 10

    # Discriminative LR: backbone uses `backbone_lr_mult * lr`. Standard for
    # pretrained ViT fine-tuning. Applied via a second optimizer param group.
    backbone_lr_mult: float = 0.1

    # ── Logging / checkpointing ───────────────────────────────────────────
    log_interval:   int          = 10
    val_limit:      Optional[int] = 50
    qual_every_n:   int          = 5
    qual_n:         int          = 20

    checkpoint_dir:  str          = "/kaggle/working/checkpoints"
    resume_path:     Optional[str] = None
    save_frequency:  int          = 5

    # ── SigLIP cached-feature training (cfg.encoder.arch == "siglip") ─────
    # Path to the .pt file produced by datasets/feature_cache.py.
    # When set, training loads cached features instead of running the
    # SigLIP vision encoder online.
    siglip_feature_cache: Optional[str] = "/kaggle/working/siglip_features.pt"

    # ── X-CLIP cached-feature training (cfg.encoder.arch == "xclip") ──────
    # Path to the .pt file produced by datasets/video_feature_cache.py.
    # Default points at the hand-crop variant — set
    # cfg.encoder.enable_hand_crop=False AND change this path to ablate
    # without hand cropping.
    xclip_feature_cache: Optional[str] = "/kaggle/working/xclip_features_handcrop.pt"

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
        # Keep DataConfig.img_size in sync with EncoderConfig.img_size
        self.data.img_size = self.encoder.img_size
        return self

    def log_dir(self) -> str:
        return os.path.join(self.train.checkpoint_dir, "..", "logs")

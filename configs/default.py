"""
configs/default.py — WiTA v2 Configuration

Dataclass-based configuration. Zero global mutable state.
Every module receives a Config (or sub-config) object explicitly;
nothing reads module-level singletons.

Usage
-----
    from configs.default import Config
    cfg = Config()                         # all defaults
    cfg = Config(data=DataConfig(lang="korean"), train=TrainConfig(epochs=60))

Kaggle one-liner override:
    cfg = Config()
    cfg.train.batch_size = 4
    cfg.train.accum_steps = 4
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
    Vocabulary configuration.

    The CTC blank is always index 0 (StrLabelConverter convention).
    Characters occupy indices 1..N.
    Attention-decoder special tokens are appended at N+1, N+2, N+3.
    """
    lang: Literal["english", "korean", "both"] = "english"

    # Populated by build() — do not set manually.
    chars:          str = field(default="", init=False, repr=False)
    blank_idx:      int = field(default=0,  init=False)
    ctc_vocab_size: int = field(default=0,  init=False)   # len(chars) + 1 (blank)
    sos_idx:        int = field(default=0,  init=False)
    eos_idx:        int = field(default=0,  init=False)
    pad_idx:        int = field(default=0,  init=False)
    attn_vocab_size:int = field(default=0,  init=False)

    def build(self) -> "VocabConfig":
        self.chars          = VOCAB_MAP[self.lang]
        N                   = len(self.chars)
        self.blank_idx      = 0
        self.ctc_vocab_size = N + 1          # blank(0)  + chars(1..N)
        self.sos_idx        = N + 1
        self.eos_idx        = N + 2
        self.pad_idx        = N + 3
        self.attn_vocab_size = N + 4
        return self


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    hf_repo_id:    str             = "yewon816/WiTA"
    download_dir:  str             = "/kaggle/working/downloads"
    hf_cache_dir:  str             = os.path.expanduser("~/.cache/huggingface")
    max_zips:      Optional[int]   = None   # None = all; int = debug subset
    lang:          Literal["english", "korean", "both"] = "english"

    # Frame pre-processing
    img_size:      int             = 112
    max_frames:    int             = 64
    img_mean:      tuple           = (0.485, 0.456, 0.406)
    img_std:       tuple           = (0.229, 0.224, 0.225)

    # Split & reproducibility
    train_split:   float           = 0.90
    seed:          int             = 42


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    # Spatial (PIL-level, applied uniformly across all frames)
    mirror_prob:     float = 0.30
    rotation_deg:    float = 5.0         # ±5° matches WiTA baseline
    brightness:      float = 0.50
    contrast:        float = 0.50
    saturation:      float = 0.50
    hue:             float = 0.50
    grayscale_prob:  float = 0.10

    # Temporal (tensor-level)
    temporal_crop_ratio: tuple = (0.75, 1.0)
    temporal_min_frames: int   = 8
    drop_frames_prob:    float = 0.10


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """
    3-D ResNet backbone configuration.
    arch: which variant to build.
    num_res_layers: blocks per stage (1 → R3D-10, 2 → R3D-18).
    out_dim: final feature dimension after the internal FC layer.
    pooling: spatial-temporal pooling strategy ('average' | 'max').
    pretrained: load ImageNet weights for matching torchvision models.
    track_running_stats: BatchNorm3d flag (False helps tiny-batch training).
    """
    arch:               Literal["r3d", "mc3", "rmc3", "r2plus1d", "r2d"] = "r3d"
    num_res_layers:     int   = 1
    out_dim:            int   = 256
    pooling:            Literal["average", "max"] = "average"
    pretrained:         bool  = False
    track_running_stats:bool  = True


# ---------------------------------------------------------------------------
# Recurrent head (optional, sits between encoder and CTC projection)
# ---------------------------------------------------------------------------

@dataclass
class RecurrentConfig:
    """
    Optional BiLSTM / BiGRU / TransformerEncoder head.
    Set arch='none' to skip (direct encoder→CTC projection).
    """
    arch:         Literal["lstm", "gru", "transformer", "none"] = "lstm"
    hidden_size:  int = 256    # per-direction for RNNs; d_model for Transformer
    num_layers:   int = 2
    nhead:        int = 8      # Transformer only
    ff_dim:       int = 1024   # Transformer only
    dropout:      float = 0.1
    fc_hidden:    int = 256    # intermediate FC after RNN  (→ num_class)


# ---------------------------------------------------------------------------
# Attention Decoder
# ---------------------------------------------------------------------------

@dataclass
class AttnDecoderConfig:
    n_layers:    int   = 4
    n_heads:     int   = 8
    ff_dim:      int   = 2048
    dropout:     float = 0.1
    max_seq_len: int   = 22    # max_word_len + 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Batch / accumulation
    batch_size:   int   = 4     # micro-batch per forward pass
    accum_steps:  int   = 4     # effective batch = batch_size * accum_steps

    num_workers:  int   = 2
    pin_memory:   bool  = False  # False when num_workers > 0 (Kaggle RAM safety)
    persistent_workers: bool = True

    # Optimisation
    num_epochs:   int   = 40
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    beta1:        float = 0.90
    beta2:        float = 0.98
    grad_clip:    float = 5.0
    optimizer:    Literal["adamw", "adam", "sgd", "rmsprop", "lamb"] = "adamw"

    # Scheduler
    scheduler:    Literal["onecycle", "warmup_multistep", "steplr", "none"] = "onecycle"
    warmup_pct:   float = 0.05
    final_div_factor: float = 300.0
    scheduler_gamma: float = 0.1
    scheduler_step:  int   = 5

    # Hybrid loss
    lambda_ctc_start: float = 0.50   # annealed to lambda_ctc_min over epochs
    lambda_ctc_min:   float = 0.20
    label_smoothing:  float = 0.10

    # Logging
    log_interval:  int  = 10    # batches between log lines
    val_limit:     Optional[int] = 50   # max val batches per pass; None = full
    qual_every_n:  int  = 5     # epochs between qualitative sample tables
    qual_n:        int  = 20    # samples in qualitative table

    # Checkpointing
    checkpoint_dir: str         = "/kaggle/working/checkpoints"
    resume_path:    Optional[str] = None
    save_frequency: int         = 5     # save every N epochs (+ best)

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration object.  Pass around instead of reading globals.

    Example
    -------
        cfg = Config()
        cfg.vocab.build()   # called automatically if you use Config.build()
    """
    vocab:   VocabConfig       = field(default_factory=VocabConfig)
    data:    DataConfig        = field(default_factory=DataConfig)
    aug:     AugConfig         = field(default_factory=AugConfig)
    encoder: EncoderConfig     = field(default_factory=EncoderConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    attn:    AttnDecoderConfig = field(default_factory=AttnDecoderConfig)
    train:   TrainConfig       = field(default_factory=TrainConfig)

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def build(self) -> "Config":
        """
        Finalise derived fields (vocab indices, device).
        Call once after all overrides are applied.
        """
        self.vocab.lang = self.data.lang
        self.vocab.build()
        return self

    def log_dir(self) -> str:
        return os.path.join(self.train.checkpoint_dir, "..", "logs")

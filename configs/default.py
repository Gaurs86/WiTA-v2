"""
configs/default.py — WiTA v2 Configuration

FIX (Bug 1 — VocabConfig index collision)
------------------------------------------
The original build() set:

    ctc_vocab_size = N + 1   (blank=0, chars=1..N)
    sos_idx        = N + 1

StrLabelConverter.encode() inserts a consecutive-repeat separator '-' at
index N+1 whenever a label contains a doubled letter (e.g. 'suggestion'
→ [s,u,g, SEP(N+1), g,e,s,t,i,o,n]).  This creates two distinct bugs:

BUG A — out-of-range CTC target:
  ctc_vocab_size = N+1 means valid CTC output indices are 0..N.
  The separator has index N+1, which is >= ctc_vocab_size.
  PyTorch's nn.CTCLoss(zero_infinity=True) silently zeroes the loss AND
  the gradient for any sample that contains a repeated letter.  The CTC
  head receives zero alignment signal for those samples (~40% of the
  English word vocabulary contains doubled letters).

BUG B — sos_idx collision:
  sos_idx was also set to N+1, identical to the separator index.
  prepare_attn_targets() prepends SOS to every label.  For words with
  doubled letters the resulting tgt_input contains a second token with
  value N+1 mid-sequence — indistinguishable from SOS.  The attention
  decoder cannot learn a consistent sequence-start boundary and collapses.

Fix: include the separator in the CTC vocabulary and shift all attention
special tokens up by one:

    ctc_vocab_size = N + 2   ← blank(0) + chars(1..N) + separator(N+1)
    sos_idx        = N + 2   ← no longer collides with separator
    eos_idx        = N + 3
    pad_idx        = N + 4
    attn_vocab_size= N + 5
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

    Index layout (after fix)
    ------------------------
    0          : CTC blank
    1 .. N     : characters
    N+1        : '-' consecutive-repeat separator  (StrLabelConverter convention)
    N+2        : <SOS>  (was N+1 — COLLISION with separator, now fixed)
    N+3        : <EOS>
    N+4        : <PAD>
    attn_vocab_size = N+5
    """
    lang: Literal["english", "korean", "both"] = "english"

    chars:           str = field(default="", init=False, repr=False)
    blank_idx:       int = field(default=0,  init=False)
    sep_idx:         int = field(default=0,  init=False)   # '-' separator (NEW — exposed)
    ctc_vocab_size:  int = field(default=0,  init=False)
    sos_idx:         int = field(default=0,  init=False)
    eos_idx:         int = field(default=0,  init=False)
    pad_idx:         int = field(default=0,  init=False)
    attn_vocab_size: int = field(default=0,  init=False)

    def build(self) -> "VocabConfig":
        self.chars          = VOCAB_MAP[self.lang]
        N                   = len(self.chars)
        self.blank_idx      = 0
        self.sep_idx        = N + 1        # StrLabelConverter's '-' separator
        self.ctc_vocab_size = N + 2        # blank(0) + chars(1..N) + sep(N+1)
        self.sos_idx        = N + 2        # shifted up — no longer == sep_idx
        self.eos_idx        = N + 3
        self.pad_idx        = N + 4
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

    img_size:      int             = 112
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

    temporal_crop_ratio: tuple = (0.75, 1.0)
    temporal_min_frames: int   = 8
    drop_frames_prob:    float = 0.10


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    arch:               Literal["r3d", "mc3", "rmc3", "r2plus1d", "r2d"] = "r3d"
    num_res_layers:     int   = 1
    out_dim:            int   = 256
    pooling:            Literal["average", "max"] = "average"
    pretrained:         bool  = False
    track_running_stats:bool  = True


# ---------------------------------------------------------------------------
# Recurrent head
# ---------------------------------------------------------------------------

@dataclass
class RecurrentConfig:
    arch:         Literal["lstm", "gru", "transformer", "none"] = "lstm"
    hidden_size:  int = 256
    num_layers:   int = 2
    nhead:        int = 8
    ff_dim:       int = 1024
    dropout:      float = 0.1
    fc_hidden:    int = 256


# ---------------------------------------------------------------------------
# Attention Decoder
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
    batch_size:   int   = 4
    accum_steps:  int   = 4

    num_workers:  int   = 2
    pin_memory:   bool  = False
    persistent_workers: bool = True

    num_epochs:   int   = 40
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    beta1:        float = 0.90
    beta2:        float = 0.98
    grad_clip:    float = 5.0
    optimizer:    Literal["adamw", "adam", "sgd", "rmsprop", "lamb"] = "adamw"

    scheduler:    Literal["onecycle", "warmup_multistep", "steplr", "none"] = "onecycle"
    warmup_pct:   float = 0.05
    final_div_factor: float = 300.0
    scheduler_gamma: float = 0.1
    scheduler_step:  int   = 5

    # Recommended: start low so the attention decoder gets sufficient gradient
    # signal from the outset. Do not exceed 0.5 — at high lambda the corrupted
    # CTC loss for repeated-letter words dominates and destabilises training.
    lambda_ctc_start: float = 0.20   # keep low: high λ amplifies blank-collapse damage
    lambda_ctc_min:   float = 0.10
    label_smoothing:  float = 0.10

    log_interval:  int  = 10
    val_limit:     Optional[int] = 50
    qual_every_n:  int  = 5
    qual_n:        int  = 20

    checkpoint_dir: str         = "/kaggle/working/checkpoints"
    resume_path:    Optional[str] = None
    save_frequency: int         = 5

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
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def build(self) -> "Config":
        self.vocab.lang = self.data.lang
        self.vocab.build()
        return self

    def log_dir(self) -> str:
        return os.path.join(self.train.checkpoint_dir, "..", "logs")

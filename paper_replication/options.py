"""
options.py — paper's AirTypingOptions + Stage 13B joint-decoder flags.

Stage 13B additions (vs the original Kim et al. file):
  * --loss_type            'ctc' or 'joint'
  * --use_joint_decoder    enable the attention decoder branch
  * --lambda_ctc           CTC weight in joint loss
  * --attn_decoder_layers  decoder depth (Stage 9a winner: 2)
  * --attn_decoder_heads   decoder heads
  * --label_smoothing      attention CE label smoothing
"""

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)


def _str2bool(x):
    """Accept 'True'/'true'/'1' or 'False'/'false'/'0' for bool argparse flags."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {x!r}")


class AirTypingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Air Typing Options")

        # EXPERIMENT Name
        self.parser.add_argument("--model_name", type=str, default="wita-resnet3d",
                                 help="model name")

        # PATHS
        self.parser.add_argument("--data_path_train", type=str,
                                 default='/root/dataset/wita/RawData/data/english/train')
        self.parser.add_argument("--data_path_val",   type=str,
                                 default='/root/dataset/wita/RawData/data/english/val')
        self.parser.add_argument("--data_path_test",  type=str,
                                 default='/root/dataset/wita/RawData/data/english/test')
        self.parser.add_argument("--save_dir",        type=str, default=file_dir,
                                 help="directory to save logs, models, etc.")
        self.parser.add_argument("--load_dir",        type=str, default='',
                                 help="directory of models to load (default: don't load)")

        # MODEL options
        self.parser.add_argument("--model_type",     type=str, default='r3d',
                                 help="['r2d', 'r3d', 'mc3', 'rmc3', 'twoplusone']")
        self.parser.add_argument("--num_res_layer",  type=int, default=1)
        self.parser.add_argument("--pretrained",     type=_str2bool, default=False,
                                 help="load torchvision pretrained weights for backbone")
        self.parser.add_argument("--recurrent_type", type=str, default="none",
                                 help="[gru, lstm, transformers, none]")
        self.parser.add_argument("--input_size",     type=int, default=256)
        self.parser.add_argument("--hidden_size",    type=int, default=256)
        self.parser.add_argument("--num_layers",     type=int, default=2)
        self.parser.add_argument("--num_encoder_layer", type=int, default=6)
        self.parser.add_argument("--d_model",        type=int, default=256)
        self.parser.add_argument("--nhead",          type=int, default=8)
        self.parser.add_argument("--hidden_size_fc", type=int, default=128)
        self.parser.add_argument("--img_size",       type=int, default=112)
        self.parser.add_argument("--pooling_type",   type=str, default="average",
                                 help="[max, average]")

        # TRAINING options
        self.parser.add_argument('--track_running',  type=_str2bool, default=True)
        self.parser.add_argument('--seed_number',    type=int, default=0)
        self.parser.add_argument("--save_frequency", type=int, default=50)
        self.parser.add_argument("--log_interval",   type=int, default=100)
        self.parser.add_argument("--tensorboard_path", type=str, default='')

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",        type=int,   default=1)
        self.parser.add_argument("--learning_rate",     type=float, default=1e-3)
        self.parser.add_argument("--num_epochs",        type=int,   default=175)
        self.parser.add_argument("--optimizer_type",    type=str,   default='adam',
                                 help="['rmsprop','sgd','lamb','adam','adamW']")
        self.parser.add_argument("--scheduler_type",    type=str,   default='warmup',
                                 help="['warmup','steplr','none']")
        self.parser.add_argument("--scheduler_step_size", type=int, default=5)
        self.parser.add_argument("--scheduler_gamma",   type=float, default=0.9)

        # DATASET options
        self.parser.add_argument("--data_type",    type=str,        default='english',
                                 help="[english, korean, eng_kor]")
        self.parser.add_argument("--data_augment", type=_str2bool,  default=False)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",     action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=2)

        # ================================================================
        # Stage 13B additions — joint CTC + attention decoder
        # ================================================================
        self.parser.add_argument("--loss_type",            type=str, default='ctc',
                                 help="['ctc','joint'] -- 'joint' enables the attention head loss")
        self.parser.add_argument("--use_joint_decoder",    type=_str2bool, default=False,
                                 help="build the Transformer attention decoder branch")
        self.parser.add_argument("--lambda_ctc",           type=float, default=0.5,
                                 help="weight for CTC in joint loss; attn weight = 1 - lambda_ctc")
        self.parser.add_argument("--attn_decoder_layers",  type=int, default=2,
                                 help="number of attention decoder layers (Stage 9a winner: 2)")
        self.parser.add_argument("--attn_decoder_heads",   type=int, default=4)
        self.parser.add_argument("--attn_decoder_dropout", type=float, default=0.1)
        self.parser.add_argument("--label_smoothing",      type=float, default=0.1)
        self.parser.add_argument("--attn_max_len",         type=int, default=32,
                                 help="max decoded length for greedy attention decode at eval")
        self.parser.add_argument("--max_frames",           type=int, default=0,
                                 help="cap clip length to this many frames via uniform "
                                      "temporal sampling (0 = no cap).  Bounds GPU memory "
                                      "for very long clips; CTC T_out = max_frames/4.")
        self.parser.add_argument("--use_amp",              type=_str2bool, default=True,
                                 help="mixed-precision training (fp16 autocast + GradScaler). "
                                      "~2x faster + ~half memory on a single GPU.  CTC/CE "
                                      "losses still computed in fp32 for stability.")
        self.parser.add_argument("--cache_dir",            type=str, default='',
                                 help="dir of pre-decoded uint8 frame caches (built by "
                                      "build_cache.py).  CER-neutral: stores the exact "
                                      "decoded+resized+capped frames; augmentation still "
                                      "runs per-epoch.  Eliminates the JPEG-decode bottleneck.")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

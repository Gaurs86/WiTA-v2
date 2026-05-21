"""
datasets/collate.py — Sequence-length helpers and pad_collate.

Re-implements all calc_seq_len_* functions from WiTA baseline utils.py.
These compute the temporal length after the 3-D ResNet encoder's temporal
down-sampling, which is required as input_lengths for nn.CTCLoss.

Every function signature and formula is identical to the baseline so that
checkpoints remain compatible.

Additionally provides make_pad_collate() which returns a pad_collate function
configured for a specific (model_arch, lang, num_res_layers) combination.
"""

from __future__ import annotations
import math
from typing import Callable

import torch
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Sequence-length helpers  (exact copies from WiTA baseline utils.py)
# ---------------------------------------------------------------------------

def calc_seq_len(n: int) -> int:
    """r3d / r2plus1d (English & Korean with 1 layer)."""
    n = math.floor((n - 3 + 2) / 2) + 1
    n = math.floor((n - 3 + 2) / 2) + 1
    return n


def calc_seq_len_mc3(n: int) -> int:
    """mc3 (English)."""
    n = math.floor((n - 3 + 2) / 2) + 1
    n = math.floor((n - 2) / 2) + 1
    return n


def calc_seq_len_rmc3(n: int) -> int:
    """rmc3 (English)."""
    n = math.floor((n - 2) / 2) + 1
    n = math.floor((n - 3 + 2) / 2) + 1
    return n


def calc_seq_len_2d_eng(n: int) -> int:
    """r2d (English)."""
    return math.floor((n - 2) / 2) + 1


def calc_seq_len_2d_kor(n: int) -> int:
    """r2d (Korean)."""
    n = n + 2
    return math.floor((n - 2) / 2) + 1


def seq_len_r3d_kor(n: int) -> int:
    """r3d (Korean, 2 layers)."""
    n = n + 2
    n = math.floor((n - 3 + 2) / 2) + 1
    n = math.floor((n - 3 + 2) / 2) + 1
    return n


def seq_len_mc3_kor(n: int) -> int:
    """mc3 (Korean, 2 layers)."""
    n = n + 2
    n = math.floor((n - 3 + 2) / 2) + 1
    n = math.floor((n - 2) / 2) + 1
    return n


def seq_len_rmc3_kor(n: int) -> int:
    """rmc3 (Korean, 2 layers)."""
    n = n + 2
    n = math.floor((n - 2) / 2) + 1
    n = math.floor((n - 3 + 2) / 2) + 1
    return n


# ---------------------------------------------------------------------------
# Seq-len dispatcher
# ---------------------------------------------------------------------------

def get_seq_len_fn(
    arch:           str,
    lang:           str,
    num_res_layers: int,
) -> Callable[[int], int]:
    """
    Return the correct seq-len function for a given (arch, lang, layers) combo.
    Mirrors the if/elif tree in baseline train.py pad_collate.
    """
    english_or_kor1 = lang == "english" or (lang == "korean" and num_res_layers == 1)

    if english_or_kor1:
        return {
            "r3d":      calc_seq_len,
            "r2plus1d": calc_seq_len,
            "rmc3":     calc_seq_len_rmc3,
            "mc3":      calc_seq_len_mc3,
            "r2d":      calc_seq_len_2d_eng,
        }.get(arch, calc_seq_len)

    else:  # korean, 2 layers
        return {
            "r3d":      seq_len_r3d_kor,
            "r2plus1d": calc_seq_len,
            "rmc3":     seq_len_rmc3_kor,
            "mc3":      seq_len_mc3_kor,
            "r2d":      calc_seq_len_2d_kor,
        }.get(arch, seq_len_r3d_kor)


# ---------------------------------------------------------------------------
# pad_collate factory
# ---------------------------------------------------------------------------

def make_pad_collate(
    arch:           str = "r3d",
    lang:           str = "english",
    num_res_layers: int = 1,
) -> Callable:
    """
    Return a pad_collate function bound to the given model / data config.

    The returned function:
      • Pads video tensors  [T, C, H, W]  along T  → [B, T_max, C, H, W]
      • Pads label tensors  [L]            along L  → [B, L_max]
      • Computes CTC input_lengths via the correct seq_len helper
      • Returns (clips_pad, labels_pad, input_lens, label_lens)

    Mirrors baseline pad_collate in train.py exactly.

    pin_memory note
    ---------------
    Pinning is handled at the DataLoader level, not here.
    """
    seq_len_fn = get_seq_len_fn(arch, lang, num_res_layers)

    def _pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        clips, labels = zip(*batch)

        # Clips: each is [T, C, H, W] — pad along T
        clips_pad  = pad_sequence(clips,  batch_first=True, padding_value=0.0)  # [B, T, C, H, W]
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=0)    # [B, L]

        input_lens = torch.LongTensor([seq_len_fn(c.shape[0]) for c in clips])
        label_lens = torch.LongTensor([l.shape[0]             for l in labels])

        return clips_pad, labels_pad, input_lens, label_lens

    return _pad_collate

"""
datasets/collate.py — Sequence-length helpers and pad_collate.

Re-implements all calc_seq_len_* functions from WiTA baseline utils.py.
These compute the temporal length after the 3-D ResNet encoder's temporal
down-sampling, which is required as input_lengths for nn.CTCLoss.

FIX (Bug 2)
-----------
The original code padded label tensors with value 0 (the CTC blank index),
but VocabConfig.pad_idx = N+3 (29 for English).  The attention-decoder mask
`tgt_key_padding = tgt_input.eq(vocab.pad_idx)` therefore never fired,
leaving blank-padded positions unmasked.  This caused two problems:
  1. The attention decoder attended to blank (0) tokens as if they were
     real content, degrading every cross-attention computation.
  2. CrossEntropyLoss was computed over blank-padded target positions (the
     loss is supposed to ignore pad_idx via ignore_index=pad_idx, but the
     padded tokens were 0=blank, not pad_idx, so they were NOT ignored).

Fix: make_pad_collate() now accepts `pad_idx` and passes it as
`padding_value` to pad_sequence for labels.
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
    pad_idx:        int = 0,    # ← NEW: must be VocabConfig.pad_idx (N+3)
) -> Callable:
    """
    Return a pad_collate function bound to the given model / data config.

    Parameters
    ----------
    pad_idx : int
        The padding token index from VocabConfig (= N+3, NOT 0).
        Used as padding_value when stacking label tensors so that the
        attention-decoder key-padding mask fires correctly.

    Returns (clips_pad, labels_pad, input_lens, label_lens).
    """
    seq_len_fn = get_seq_len_fn(arch, lang, num_res_layers)

    def _pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        clips, labels = zip(*batch)

        # Clips: each is [T, C, H, W] — pad along T with 0.0 (black frame)
        clips_pad  = pad_sequence(clips, batch_first=True, padding_value=0.0)

        # Labels: pad with pad_idx (N+3), NOT with blank (0).
        # This ensures tgt_key_padding = tgt_input.eq(pad_idx) fires correctly
        # inside prepare_attn_targets, and CrossEntropyLoss ignores these positions
        # via ignore_index=pad_idx.
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)

        input_lens = torch.LongTensor([seq_len_fn(c.shape[0]) for c in clips])
        label_lens = torch.LongTensor([l.shape[0]             for l in labels])

        return clips_pad, labels_pad, input_lens, label_lens

    return _pad_collate

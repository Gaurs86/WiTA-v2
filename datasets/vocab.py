"""
datasets/vocab.py — Self-contained vocabulary and label conversion.

Re-implements StrLabelConverter from the WiTA baseline (utils.py) with:
  • Identical encoding/decoding logic — existing checkpoints remain compatible
  • Type hints and proper docstrings
  • No dependency on the original repo

StrLabelConverter conventions (preserved from baseline)
-------------------------------------------------------
  Index 0        : CTC blank (reserved — never assigned to a character)
  Index 1 .. N   : characters from the alphabet, 1-based
  Index N+1      : '-' separator for consecutive identical characters
                   (used by encode() to insert a separator between repeated chars
                    so CTCLoss can learn them as distinct targets)
  The alphabet stored in self.alphabet is: user_chars + '-'

Korean labels must be hgtk-decomposed *before* calling encode().
"""

from __future__ import annotations
import collections
from typing import Union

import torch
import editdistance


# ---------------------------------------------------------------------------
# Character sets (same strings as WiTA baseline utils.py)
# ---------------------------------------------------------------------------

ALPHABET  = "abcdefghijklmnopqrstuvwxyz"
HANGUL    = (
    "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅃㅉㄸㄲㅆ"
    "ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅀㅄ"
    "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢᴥ "
)
ALPHA_HAN = ALPHABET + HANGUL

VOCAB_BY_LANG: dict[str, str] = {
    "english": ALPHABET,
    "korean":  HANGUL,
    "both":    ALPHA_HAN,
}


# ---------------------------------------------------------------------------
# StrLabelConverter
# ---------------------------------------------------------------------------

class StrLabelConverter:
    """
    Convert between character strings and integer label tensors.

    Compatible with the WiTA baseline's utils.StrLabelConverter — encode()
    and decode() produce/consume identical tensors so saved checkpoints and
    gt.txt parsing remain unchanged.

    Parameters
    ----------
    alphabet    : character set (e.g. ALPHABET for English)
    ignore_case : lowercase input before encoding (default True)
    """

    def __init__(self, alphabet: str, ignore_case: bool = True):
        self._ignore_case = ignore_case
        if ignore_case:
            alphabet = alphabet.lower()

        # Append '-' as the consecutive-repeat separator (same as baseline)
        self.alphabet = alphabet + "-"

        # dict maps char → index (1-based; 0 reserved for blank)
        self.dict: dict[str, int] = {
            char: i + 1 for i, char in enumerate(self.alphabet)
        }

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(
        self, text: Union[str, list[str]]
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        """
        Encode text to integer indices.

        For a **single string**: consecutive identical characters are
        separated by the '-' token (index of '-' in self.dict).  This
        mirrors the baseline exactly and is required for CTCLoss to treat
        them as distinct targets.

        For a **list of strings**: concatenate and return lengths.

        Returns
        -------
        text_tensor : IntTensor [total_length]
        length      : IntTensor [batch_size]
        """
        if isinstance(text, str):
            text_list: list[int] = []
            prev = ""
            for char in text:
                if self._ignore_case:
                    char = char.lower()
                if char == prev:
                    text_list.append(self.dict["-"])   # separator
                text_list.append(self.dict[char])
                prev = char
            return torch.IntTensor(text_list), torch.IntTensor([len(text_list)])

        elif isinstance(text, (list, tuple)):
            length = [len(s) for s in text]
            joined = "".join(text)
            enc, _ = self.encode(joined)
            return enc, torch.IntTensor(length)

        raise TypeError(f"Expected str or list[str], got {type(text)}")

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(
        self,
        t:      torch.IntTensor,    # [L] or [total_L]
        length: torch.IntTensor,    # [1] or [B]
        raw:    bool = False,
    ) -> Union[str, list[str]]:
        """
        Decode integer indices back to a string (or list of strings).

        raw=False : collapse consecutive repeats and strip blank (0).
                    This is the standard CTC greedy-decode post-processing.
        raw=True  : return raw character sequence without collapsing.
        """
        if length.numel() == 1:
            L = int(length[0])
            assert t.numel() == L, (
                f"text length {t.numel()} != declared length {L}"
            )
            if raw:
                return "".join(self.alphabet[int(i) - 1] for i in t)
            chars = []
            for i in range(L):
                idx = int(t[i])
                if idx != 0 and not (i > 0 and int(t[i - 1]) == idx):
                    chars.append(self.alphabet[idx - 1])
            return "".join(chars)

        # Batch mode
        assert t.numel() == int(length.sum()), (
            f"total text length {t.numel()} != sum of lengths {length.sum()}"
        )
        texts, start = [], 0
        for L in length:
            L = int(L)
            texts.append(
                self.decode(t[start : start + L], torch.IntTensor([L]), raw=raw)
            )
            start += L
        return texts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_converter(lang: str = "english") -> StrLabelConverter:
    """Build a StrLabelConverter for the given language."""
    if lang not in VOCAB_BY_LANG:
        raise ValueError(f"Unknown lang '{lang}'. Choose from {list(VOCAB_BY_LANG)}")
    alphabet = VOCAB_BY_LANG[lang]
    ignore_case = lang != "korean"   # Korean jamo are case-insensitive but not ASCII
    return StrLabelConverter(alphabet, ignore_case=ignore_case)


# ---------------------------------------------------------------------------
# CER / WER (from WiTA baseline utils.py — re-implemented here)
# ---------------------------------------------------------------------------

def cer(ref: str, hyp: str) -> tuple[int, int]:
    """
    Character Error Rate between reference and hypothesis strings.

    Strips <START> / <EOS> markers (same as baseline).

    Returns
    -------
    (edit_distance, ref_length)
    — NOT the ratio.  The caller divides: cer_ratio = err / max(length, 1)
    """
    import re
    ref = re.sub(r"^<START>|<EOS>$", "", ref)
    hyp = re.sub(r"^<START>|<EOS>$", "", hyp)
    return editdistance.eval(ref, hyp), len(ref)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """
    Word Error Rate (edit distance on word tokens).

    Returns (edit_distance_on_words, n_ref_words).
    """
    import re
    ref = re.sub(r"^<START>|<EOS>$", "", ref).split()
    hyp = re.sub(r"^<START>|<EOS>$", "", hyp).split()
    return editdistance.eval(ref, hyp), len(ref)

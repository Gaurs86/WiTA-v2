"""evaluation/metrics.py — CER / WER metric helpers."""
from __future__ import annotations
from ..datasets.vocab import cer, wer          # self-contained implementations
from ..datasets.vocab import StrLabelConverter
import torch


def decode_ctc_indices(indices: list[int], converter: StrLabelConverter) -> str:
    """Convert raw CTC token indices (1-based) → string via StrLabelConverter."""
    valid = [i for i in indices if 0 < i <= len(converter.alphabet) - 1]
    if not valid:
        return ""
    return converter.decode(
        torch.IntTensor(valid),
        torch.IntTensor([len(valid)]),
    )


def decode_attn_indices(indices: list[int], converter: StrLabelConverter,
                        sos_idx: int, eos_idx: int, pad_idx: int) -> str:
    """Convert attention decoder indices → string, stripping specials."""
    specials = {0, sos_idx, eos_idx, pad_idx}
    clean = [i for i in indices if i not in specials and 0 < i <= len(converter.alphabet) - 1]
    if not clean:
        return ""
    return converter.decode(
        torch.IntTensor(clean),
        torch.IntTensor([len(clean)]),
    )

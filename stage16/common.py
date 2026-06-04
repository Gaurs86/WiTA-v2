"""
stage16/common.py — shared pieces for the decoupled CLIP-features +
temporal-CTC + LM-decode pipeline (Stage 16).

Vocab (STANDARD CTC): blank=0, 'a'..'z' = 1..26 -> V=27.
NOTE: unlike the paper's StrLabelConverter (which uses a '-' repeat separator,
V=28), Stage 16 trains a FRESH model and uses standard CTC -- repeated chars
are handled by the blank, so no '-' token.  This keeps pyctcdecode + KenLM
integration clean (the beam output is plain text, so a word LM works).

Data layout (paper 8:1:1 person split), read directly from the Kaggle mount:
  <root>/eng_<split>_<subset>/<subset>/<signer_dir>/<clip_idx>/<frame>.jpg
  gt.txt in <signer_dir>/, label for a clip = gt.txt line [int(clip_idx)].
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import editdistance
except ImportError:
    editdistance = None


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOCAB = 27          # STANDARD CTC: blank=0, a..z=1..26


class CharConverter:
    """Standard-CTC char converter (blank=0, a..z=1..26).  Repeated chars are
    handled by the CTC blank, not a '-' separator, so the beam output is plain
    text (clean for pyctcdecode + a word/char KenLM)."""

    def __init__(self, alphabet: str = ALPHABET):
        self.alphabet = alphabet                        # 26 chars -> idx 1..26
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.alphabet)}

    def encode(self, text: str) -> list[int]:
        text = text.lower()
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_ctc(self, ids) -> str:
        out, prev = [], -1
        for i in ids:
            i = int(i)
            if i != 0 and i != prev and 0 < i <= len(self.alphabet):
                out.append(self.alphabet[i - 1])
            prev = i
        return "".join(out)

    def ids_to_text(self, ids) -> str:
        """Map raw label ids -> text WITHOUT CTC collapse (for reconstructing a
        ground-truth string from encoded targets).  decode_ctc must NOT be used
        for this: it collapses consecutive repeats, so 'letter' -> 'leter'."""
        return "".join(self.alphabet[int(i) - 1] for i in ids
                       if 0 < int(i) <= len(self.alphabet))

    @property
    def pyctc_labels(self) -> list[str]:
        """pyctcdecode labels: index 0 = blank ('' empty string), then a..z.
        Length must equal V=27."""
        return [""] + list(self.alphabet)               # ['', a..z] -> 27


def gt_string(label: str) -> str:
    return label.lower()


# ---------------------------------------------------------------------------
# Paper-split clip walker
# ---------------------------------------------------------------------------

def find_data_root(default="/kaggle/input/datasets/gaurs86/wita-full-english-122signers"):
    """Locate the dataset root (no recursive glob -- Stage 11 lesson)."""
    if os.path.isdir(os.path.join(default, "eng_train_lex")):
        return default
    base = "/kaggle/input"
    if os.path.isdir(base):
        for c in os.listdir(base):
            p1 = os.path.join(base, c)
            if not os.path.isdir(p1):
                continue
            if os.path.isdir(os.path.join(p1, "eng_train_lex")):
                return p1
            for c2 in os.listdir(p1):
                p2 = os.path.join(p1, c2)
                if os.path.isdir(p2) and os.path.isdir(os.path.join(p2, "eng_train_lex")):
                    return p2
    raise FileNotFoundError("eng_train_lex not found; pass the dataset root explicitly.")


def walk_clips(root, split, subsets=("lex", "nonlex")):
    """Yield dicts: {dir, label, subset, signer, clip_id} for a split."""
    root = Path(root)
    out = []
    for subset in subsets:
        base = root / f"eng_{split}_{subset}" / subset
        if not base.exists():
            alt = root / f"eng_{split}_{subset}"
            base = alt if alt.exists() else base
        if not base.exists():
            continue
        for signer_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            gt = signer_dir / "gt.txt"
            if not gt.exists():
                continue
            try:
                lines = open(gt, "r", encoding="utf-8", errors="replace").read().splitlines()
            except Exception:
                continue
            for clip_dir in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
                try:
                    idx = int(clip_dir.name)
                except ValueError:
                    continue
                if idx >= len(lines):
                    continue
                label = lines[idx].strip()
                if not label:
                    continue
                out.append({
                    "dir": str(clip_dir), "label": label, "subset": subset,
                    "signer": signer_dir.name, "clip_id": f"{signer_dir.name}_{idx}",
                })
    return out


def clip_cache_name(signer, clip_id):
    """Cache filename stem: <SIGNER>__<clip_id>.npy (clip_id already includes signer)."""
    return f"{signer}__{clip_id}"


def cer_pair(ref, hyp):
    if editdistance is None:
        raise ImportError("pip install editdistance")
    e = editdistance.eval(ref, hyp)
    return min(e, len(ref)), len(ref)

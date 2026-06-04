"""
lexicon_decode.py — lexicon-constrained CTC decoding for the lex subset.

The lex subset is a CLOSED VOCABULARY of frequent English words.  Greedy
CTC decode makes character-level errors ("removable" -> "removble").  This
script instead scores every word in the lexicon under the CTC model and
picks the most probable valid word, fixing those errors.  No retraining.

Method (per clip):
  1. forward the best-val checkpoint -> CTC log-probs [T_out, V].
  2. GREEDY baseline: collapse-repeat argmax decode.
  3. LEXICON decode (lex subset only): argmax over the lexicon of the CTC
     sequence log-probability of each candidate word (exact CTC forward).
  nonlex (random strings) always uses greedy -- a lexicon doesn't apply.

The lexicon is built from TRAIN lex labels only (the frequent-word list).
Coverage of the eval split's lex words by that lexicon is reported, so the
closed-vocabulary assumption is auditable (no test-label leakage).

Discipline: --eval_split val is a free diagnostic (no marker); test is the
one-shot headline (marker-gated).  Reports GREEDY vs LEXICON CER side by
side, per subset, so the improvement from lexicon decoding is explicit.
"""

import os
import sys
import json
import math
import numpy as np
import torch
import editdistance
from pathlib import Path

import utils
from data    import AirTypingDataset
from model   import GestureTranslator
from options import AirTypingOptions
from utils   import calc_seq_len


# ---------------------------------------------------------------------------
# CTC forward (log domain) — log P(target | log_probs)
# ---------------------------------------------------------------------------

def _logsumexp2(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_logp(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """log P(target | log_probs) via the CTC forward algorithm.

    log_probs : [T, V] log-softmax.
    target    : list of label ids (1..V-1), already including the paper's
                '-' repeat-separator where the StrLabelConverter inserts it.
    """
    T = log_probs.shape[0]
    if len(target) == 0:
        return float(log_probs[:, blank].sum())          # all-blank path
    ext = [blank]
    for c in target:
        ext += [int(c), blank]
    S = len(ext)
    # Minimum frames to emit the target: len + one extra per adjacent repeat
    # (a blank must separate identical adjacent labels).
    repeats = sum(1 for i in range(1, len(target)) if target[i] == target[i - 1])
    if T < len(target) + repeats:
        return -math.inf
    NEG = -math.inf
    prev = [NEG] * S
    prev[0] = float(log_probs[0, blank])
    if S > 1:
        prev[1] = float(log_probs[0, ext[1]])
    for t in range(1, T):
        cur = [NEG] * S
        lpt = log_probs[t]
        start = max(0, S - 2 * (T - t))
        for s in range(start, S):
            v = prev[s]
            if s > 0:
                v = _logsumexp2(v, prev[s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                v = _logsumexp2(v, prev[s - 2])
            cur[s] = v + float(lpt[ext[s]])
        prev = cur
    return _logsumexp2(prev[S - 1], prev[S - 2] if S > 1 else NEG)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def build_lexicon(train_root, converter, wordfreq_topk=0):
    """Lexicon = TRAIN lex words (+ optionally the wordfreq_topk most frequent
    English words, an external prior with NO test-label leakage).  Returns a
    dict keyed by first character -> list of (word, encoded_ids, len) for fast
    candidate pruning, plus the flat word set for coverage reporting."""
    words = set()
    base = Path(train_root)
    lex = base / "lex"
    if not lex.exists():
        lex = base
    for gt in lex.rglob("gt.txt"):
        try:
            for line in open(gt, "r", encoding="utf-8", errors="replace"):
                w = line.strip().lower()
                if w:
                    words.add(w)
        except Exception:
            continue
    n_train = len(words)
    if wordfreq_topk > 0:
        try:
            from wordfreq import top_n_list
            for w in top_n_list("en", wordfreq_topk):
                w = w.strip().lower()
                if w.isalpha() and 1 <= len(w) <= 20:
                    words.add(w)
            print(f"[lexicon] train words={n_train}  + wordfreq top {wordfreq_topk} "
                  f"-> {len(words)} total", flush=True)
        except ImportError:
            print("[lexicon] wordfreq not installed (pip install wordfreq); "
                  "using TRAIN words only.", flush=True)
    by_fc = {}
    for w in sorted(words):
        enc, _ = converter.encode(w)
        ids = [int(x) for x in enc.tolist()]
        by_fc.setdefault(w[0], []).append((w, ids, len(w)))
    return by_fc, words


def greedy_decode(log_probs, converter, blank=0):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != blank and i != prev and 0 < i <= len(converter.alphabet):
            out.append(converter.alphabet[i - 1])
        prev = i
    return "".join(out).replace("-", "")


def lexicon_decode(log_probs, by_fc, greedy_str, converter, blank=0, len_window=4):
    """Lexicon-constrained decode with SOFT FALLBACK.

    Candidates = greedy string itself + lexicon words sharing the greedy's
    first character and within len_window of its length.  Pick the highest
    CTC-probability candidate.  Because greedy is always a candidate, the
    result is never less probable than greedy -> lexicon decoding can only
    help, never hurt (monotone).
    """
    gl = len(greedy_str)
    # Greedy is always a candidate (soft fallback).
    g_enc, _ = converter.encode(greedy_str)
    best_w = greedy_str
    best_s = ctc_logp(log_probs, [int(x) for x in g_enc.tolist()], blank)
    if gl == 0:
        return best_w
    for w, ids, wl in by_fc.get(greedy_str[0], ()):
        if abs(wl - gl) > len_window:
            continue
        s = ctc_logp(log_probs, ids, blank)
        if s > best_s:
            best_s, best_w = s, w
    return best_w


# ---------------------------------------------------------------------------

def run(opts, split, by_fc, lex_words):
    device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
    converter = utils.StrLabelConverter(utils.ALPHABET)

    data = AirTypingDataset(opts, opts.data_path_test)     # points at val or test
    model = GestureTranslator(opts).to(device).eval()
    ckpt = os.path.join(opts.load_dir, "model.pth")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"[lexicon_decode] loaded {ckpt}  | lexicon={len(lex_words)} words", flush=True)

    agg = {m: {sub: {"e": 0, "l": 0, "n": 0} for sub in ("lex", "nonlex")}
           for m in ("greedy", "lexicon")}
    n = len(data.video_list)
    miss_cov = 0          # lex GT words not in the lexicon
    n_override = 0        # times lexicon changed the greedy output (lex only)
    lw = int(getattr(opts, "len_window", 4))
    with torch.no_grad():
        for idx in range(n):
            vpath = data.video_list[idx]
            subset = "lex" if "/lex/" in vpath and "/nonlex/" not in vpath else "nonlex"
            gt = data.labels[idx].lower()
            video = data[idx][0].unsqueeze(0).to(device)
            x_lens = torch.LongTensor([calc_seq_len(video.size(1))]).to(device)
            logits, _ = model(video, x_lens)
            log_probs = logits[0].float().log_softmax(-1).cpu().numpy()   # [T_out, V]

            g = greedy_decode(log_probs, converter)
            if subset == "lex":
                if gt not in lex_words:
                    miss_cov += 1
                lx = lexicon_decode(log_probs, by_fc, g, converter, len_window=lw)
                if lx != g:
                    n_override += 1
            else:
                lx = g                                            # lexicon n/a

            for m, pred in (("greedy", g), ("lexicon", lx)):
                err = editdistance.eval(gt, pred); L = len(gt)
                if err > L: err = L
                agg[m][subset]["e"] += err; agg[m][subset]["l"] += L
                agg[m][subset]["n"] += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == n:
                lg = agg["greedy"]["lex"]; ll = agg["lexicon"]["lex"]
                print(f"  [{idx+1}/{n}] lex CER greedy="
                      f"{lg['e']/max(lg['l'],1):.4f}  lexicon="
                      f"{ll['e']/max(ll['l'],1):.4f}", flush=True)

    def cer(d, sub): return d[sub]["e"] / max(d[sub]["l"], 1)
    def overall(d):
        e = d["lex"]["e"] + d["nonlex"]["e"]; l = d["lex"]["l"] + d["nonlex"]["l"]
        return e / max(l, 1)

    n_lex = agg["greedy"]["lex"]["n"]
    result = {
        "eval_split": split,
        "lexicon_size": len(lex_words),
        "lex_coverage": 1.0 - miss_cov / max(n_lex, 1),
        "n_lex_words_not_in_lexicon": miss_cov,
        "n_lex_overridden_by_lexicon": n_override,
        "greedy": {
            "lex_cer": cer(agg["greedy"], "lex"),
            "nonlex_cer": cer(agg["greedy"], "nonlex"),
            "overall_cer": overall(agg["greedy"]),
        },
        "lexicon": {
            "lex_cer": cer(agg["lexicon"], "lex"),
            "nonlex_cer": cer(agg["lexicon"], "nonlex"),
            "overall_cer": overall(agg["lexicon"]),
        },
        "paper_baseline": {"lex": 0.281, "nonlex": 0.365, "overall": 0.2924},
        "n_clips_per_subset": {sub: agg["greedy"][sub]["n"] for sub in ("lex", "nonlex")},
    }
    return result


if __name__ == "__main__":
    import argparse as _ap
    # Parse base AirTypingOptions args, collecting our extra flags as
    # "unrecognized", then parse those with a separate parser.  This avoids
    # any fragility in extending the base parser after construction.
    o = AirTypingOptions()
    opts, _extra = o.parser.parse_known_args()
    _ep = _ap.ArgumentParser()
    _ep.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    _ep.add_argument("--train_root", type=str, required=True)
    _ep.add_argument("--len_window", type=int, default=4)
    _ep.add_argument("--wordfreq_topk", type=int, default=0,
                     help="add the top-K frequent English words to the lexicon "
                          "(needs `pip install wordfreq`; 0 = train words only)")
    _mine = _ep.parse_args(_extra)
    opts.eval_split = _mine.eval_split
    opts.train_root = _mine.train_root
    opts.len_window = _mine.len_window
    split = opts.eval_split

    marker = os.path.join(opts.load_dir, f".stage13b_lexicon_{split}_evaluated")
    if split == "test" and os.path.exists(marker):
        print(f"ERROR: lexicon test already evaluated. Marker: {marker}", file=sys.stderr)
        sys.exit(2)

    converter = utils.StrLabelConverter(utils.ALPHABET)
    by_fc, lex_words = build_lexicon(opts.train_root, converter,
                                     wordfreq_topk=_mine.wordfreq_topk)
    res = run(opts, split, by_fc, lex_words)

    out = os.path.join(opts.load_dir, f"lexicon_{split}_{opts.model_name}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    if split == "test":
        open(marker, "w").write(out)

    print("\n" + "=" * 64)
    print(f"  LEXICON-CONSTRAINED DECODE — {split.upper()}  -> {out}")
    print("=" * 64)
    print(f"  lexicon size       : {res['lexicon_size']}   "
          f"coverage of split lex words: {100*res['lex_coverage']:.1f}%")
    print(f"  LEX  CER   greedy={res['greedy']['lex_cer']:.4f}   "
          f"lexicon={res['lexicon']['lex_cer']:.4f}   (paper 0.281)")
    print(f"  NONLEX CER greedy={res['greedy']['nonlex_cer']:.4f}   "
          f"(lexicon n/a)   (paper 0.365)")
    print(f"  OVERALL CER greedy={res['greedy']['overall_cer']:.4f}   "
          f"lexicon={res['lexicon']['overall_cer']:.4f}   (paper 0.2924)")
    print("=" * 64)
    d = res['greedy']['lex_cer'] - res['lexicon']['lex_cer']
    print(f"  lex CER improvement from lexicon decoding: {d:+.4f}")

"""
stage17/seq_lexicon_decode.py — lexicon-constrained CTC decode on Stage 11's
landmark Conformer+CTC (the model that ACTUALLY reads the trajectory: 0.43).

Why: the lex subset is a (near-)closed vocabulary of frequent English words.
Greedy CTC makes character slips ("removable" -> "removble").  Scoring every
lexicon word under the CTC model (exact CTC forward) and picking the most
probable valid word fixes those slips.  No retraining; CPU is fine (the
Conformer over cached [32,190] features is tiny).

Reuses the verified machinery from paper_replication/lexicon_decode.py
(ctc_logp CTC-forward, soft-fallback lexicon decode) but:
  * loads ConformerCTC (models/conformer_ctc.py) from the Stage 11 checkpoint
  * reads the per-clip landmark cache (stage17/common.iter_clips)
  * builds the lexicon from TRAIN lex labels only (no val/test leakage) and
    reports coverage of the eval split's lex words (audits closed-vocab).

Vocab: paper 28-token scheme — blank=0, a-z=1..26, '-'=27 (repeat separator).
nonlex (random strings) always uses greedy; a word lexicon doesn't apply.

Discipline: val is a free diagnostic; test is one-shot (marker-gated by caller).
"""

from __future__ import annotations

import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import editdistance
except ImportError:
    editdistance = None


# ---------------------------------------------------------------------------
# Converter (paper 28-token scheme; '-' separates consecutive identical chars)
# ---------------------------------------------------------------------------

class CTCConverter:
    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz"):
        self.alphabet = alphabet + "-"                 # 27 chars -> idx 1..27
        self.dict = {c: i + 1 for i, c in enumerate(self.alphabet)}

    def encode(self, text: str) -> list[int]:
        """Insert '-' between consecutive identical chars (matches Stage 11
        StrLabelConverter), e.g. 'letter' -> [l,e,t,-,t,e,r]."""
        ids, prev = [], ""
        for ch in text.lower():
            if ch not in self.dict:
                continue
            if ch == prev:
                ids.append(self.dict["-"])
            ids.append(self.dict[ch]); prev = ch
        return ids


# ---------------------------------------------------------------------------
# CTC forward (log domain) — verified against brute force in paper_replication
# ---------------------------------------------------------------------------

def _logsumexp2(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_logp(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """log P(target | log_probs) via the CTC forward algorithm.
    log_probs: [T, V] log-softmax.  target: label ids incl. '-' separators."""
    T = log_probs.shape[0]
    if len(target) == 0:
        return float(log_probs[:, blank].sum())
    ext = [blank]
    for c in target:
        ext += [int(c), blank]
    S = len(ext)
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
# Decoders
# ---------------------------------------------------------------------------

def greedy_decode(log_probs, converter, blank=0):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != blank and i != prev and 0 < i <= len(converter.alphabet):
            out.append(converter.alphabet[i - 1])
        prev = i
    return "".join(out).replace("-", "")


def lexicon_decode(log_probs, by_fc, greedy_str, converter, blank=0, len_window=4):
    """Soft-fallback lexicon decode: greedy is always a candidate, so the result
    is never less probable than greedy (monotone -> can only help)."""
    gl = len(greedy_str)
    best_w = greedy_str
    best_s = ctc_logp(log_probs, converter.encode(greedy_str), blank)
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
# Lexicon from TRAIN lex labels (no leakage)
# ---------------------------------------------------------------------------

def _label_only(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        v = d["label"]
        return str(v.item() if getattr(v, "ndim", 0) == 0 else v).strip().lower()


def build_lexicon_from_cache(cache_root, converter, wordfreq_topk=0):
    from stage17.common import iter_clips
    words = set()
    for c in iter_clips(cache_root, "train", ("lex",)):
        w = _label_only(c["npz"])
        if w:
            words.add(w)
    n_train = len(words)
    if wordfreq_topk > 0:
        try:
            from wordfreq import top_n_list
            for w in top_n_list("en", wordfreq_topk):
                w = w.strip().lower()
                if w.isascii() and w.isalpha() and 1 <= len(w) <= 20:
                    words.add(w)
        except ImportError:
            print("[lexicon] wordfreq not installed; using train words only", flush=True)
    alpha = set(converter.alphabet)
    by_fc, kept = {}, set()
    for w in sorted(words):
        if not w or any(ch not in alpha for ch in w):
            continue
        ids = converter.encode(w)
        by_fc.setdefault(w[0], []).append((w, ids, len(w)))
        kept.add(w)
    print(f"[lexicon] train_lex unique words={n_train}  encodable retained={len(kept)}",
          flush=True)
    return by_fc, kept


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_stage11(ckpt_path, device, **arch):
    import torch
    from models.conformer_ctc import ConformerCTC
    a = dict(input_dim=190, vocab_size=28, d_model=256, n_layers=4, n_heads=4,
             conv_kernel=15, dropout=0.2, upsample=2, input_layernorm=False)
    a.update(arch)
    model = ConformerCTC(**a).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("encoder_state_dict") if isinstance(state, dict) else None
    if sd is None:
        sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_stage11] WARNING missing={len(missing)} unexpected={len(unexpected)} "
              f"-- arch may not match the checkpoint (check d_model/n_layers).", flush=True)
        if missing:    print("  missing[:5]   :", list(missing)[:5], flush=True)
        if unexpected: print("  unexpected[:5]:", list(unexpected)[:5], flush=True)
    else:
        print("[load_stage11] all encoder keys matched", flush=True)
    if isinstance(state, dict) and "best_payload" in state:
        print("[load_stage11] checkpoint best_payload:", state["best_payload"], flush=True)
    return model


# ---------------------------------------------------------------------------
# Evaluate: greedy vs lexicon, per subset
# ---------------------------------------------------------------------------

def evaluate_split(model, cache_root, split, converter, by_fc, lex_words, device,
                   len_window=4):
    import torch
    from stage17.common import iter_clips, load_clip_npz
    if editdistance is None:
        raise ImportError("pip install editdistance")
    agg = {m: {sub: {"e": 0, "l": 0, "n": 0} for sub in ("lex", "nonlex")}
           for m in ("greedy", "lexicon")}
    miss_cov = n_override = 0
    clips = iter_clips(cache_root, split, ("lex", "nonlex"))
    for i, c in enumerate(clips):
        d = load_clip_npz(c["npz"])
        feats = torch.from_numpy(d["feature"]).float().unsqueeze(0).to(device)
        in_lens = torch.LongTensor([feats.shape[1]]).to(device)
        with torch.no_grad():
            log_probs, enc_lens = model(feats, in_lens)
        lp = log_probs[0, : int(enc_lens[0])].float().cpu().numpy()       # [T_out, 28]
        gt = d["label"]; subset = d["subset"] or c["subset"]
        g = greedy_decode(lp, converter)
        if subset == "lex":
            if gt not in lex_words:
                miss_cov += 1
            lx = lexicon_decode(lp, by_fc, g, converter, len_window=len_window)
            if lx != g:
                n_override += 1
        else:
            lx = g
        for m, pred in (("greedy", g), ("lexicon", lx)):
            err = min(editdistance.eval(gt, pred), len(gt))
            agg[m][subset]["e"] += err; agg[m][subset]["l"] += len(gt); agg[m][subset]["n"] += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(clips):
            lg = agg["greedy"]["lex"]; ll = agg["lexicon"]["lex"]
            print(f"  [{i+1}/{len(clips)}] lex greedy={lg['e']/max(lg['l'],1):.4f} "
                  f"lexicon={ll['e']/max(ll['l'],1):.4f}", flush=True)

    def cer(d, sub): return d[sub]["e"] / max(d[sub]["l"], 1)
    def overall(d):
        e = d["lex"]["e"] + d["nonlex"]["e"]; l = d["lex"]["l"] + d["nonlex"]["l"]
        return e / max(l, 1)

    n_lex = agg["greedy"]["lex"]["n"]
    return {
        "eval_split": split,
        "len_window": len_window,
        "lexicon_size": len(lex_words),
        "lex_coverage": 1.0 - miss_cov / max(n_lex, 1),
        "n_lex_words_not_in_lexicon": miss_cov,
        "n_lex_overridden_by_lexicon": n_override,
        "greedy": {"lex": cer(agg["greedy"], "lex"), "nonlex": cer(agg["greedy"], "nonlex"),
                   "overall": overall(agg["greedy"])},
        "lexicon": {"lex": cer(agg["lexicon"], "lex"), "nonlex": cer(agg["lexicon"], "nonlex"),
                    "overall": overall(agg["lexicon"])},
        "paper_baseline": {"lex": 0.281, "nonlex": 0.365, "overall": 0.2924},
        "n_clips": {sub: agg["greedy"][sub]["n"] for sub in ("lex", "nonlex")},
    }


def print_result(res):
    print("=" * 64)
    print(f"  STAGE 11 + LEXICON DECODE — {res['eval_split'].upper()}")
    print("=" * 64)
    print(f"  lexicon size {res['lexicon_size']} | coverage of split lex words "
          f"{100*res['lex_coverage']:.1f}% | overrides {res['n_lex_overridden_by_lexicon']}")
    g, x = res["greedy"], res["lexicon"]
    print(f"  LEX     greedy={g['lex']:.4f}  ->  lexicon={x['lex']:.4f}   (paper 0.281)")
    print(f"  NONLEX  greedy={g['nonlex']:.4f}  (lexicon n/a)              (paper 0.365)")
    print(f"  OVERALL greedy={g['overall']:.4f}  ->  lexicon={x['overall']:.4f}  (paper 0.2924)")
    print(f"  lex CER improvement: {g['lex'] - x['lex']:+.4f}")
    print("=" * 64)

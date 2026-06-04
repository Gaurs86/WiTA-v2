"""
stage16/lm_decode.py — CTC decoding with a KenLM n-gram LM via pyctcdecode,
with a PER-SUBSET decoder choice.  This is the main CER lever (CPU, no GPU).

Decoders:
  * lexical (lex)    : WORD-level KenLM beam search, optionally lexicon-
                       constrained to the train word vocab (unigrams).  alpha
                       (LM weight) + beta (word bonus) tuned on val.
  * nonlexical(nonlex): a word LM HURTS random strings -> use greedy, OR a
                       CHARACTER-level KenLM with low weight.  Choose per subset.

Corpora are built from TRAIN gt.txt only (no val/test leakage):
  - word corpus : one frequent word per line (for the word LM + lexicon).
  - char corpus : space-separated characters per line (for the char LM).

KenLM build (run once, documented):
  lmplz -o 4 < word_corpus.txt  > word.arpa ; build_binary word.arpa word.bin
  lmplz -o 6 < char_corpus.txt  > char.arpa ; build_binary char.arpa char.bin
(or pip install https://github.com/kpu/kenlm/archive/master.zip and pass .arpa)

pyctcdecode: labels = ['', a..z] (index 0 = blank), matching CharConverter.

Eval: per-subset CER (lex vs nonlex), greedy vs LM, marker-gated one-shot test.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import (CharConverter, walk_clips, clip_cache_name,        # noqa: E402
                            cer_pair, gt_string, find_data_root)
from stage16.temporal_ctc import TemporalCTC                                   # noqa: E402


# ---------------------------------------------------------------------------
# Corpora from train labels
# ---------------------------------------------------------------------------

def build_corpora(data_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    words = []
    for sub in ("lex", "nonlex"):
        for e in walk_clips(data_root, "train", subsets=(sub,)):
            w = e["label"].strip().lower()
            if w:
                words.append((w, sub))
    lex_words = sorted({w for w, s in words if s == "lex"})
    all_words = [w for w, _ in words]
    word_corpus = os.path.join(out_dir, "word_corpus.txt")
    with open(word_corpus, "w") as f:
        for w in all_words:
            f.write(w + "\n")
    char_corpus = os.path.join(out_dir, "char_corpus.txt")
    with open(char_corpus, "w") as f:
        for w in all_words:
            f.write(" ".join(list(w)) + "\n")
    vocab_path = os.path.join(out_dir, "lex_vocab.txt")
    with open(vocab_path, "w") as f:
        for w in lex_words:
            f.write(w + "\n")
    print(f"[lm] corpora -> {out_dir}  words={len(all_words)} lex_vocab={len(lex_words)}",
          flush=True)
    return {"word_corpus": word_corpus, "char_corpus": char_corpus,
            "lex_vocab": vocab_path, "lex_words": lex_words}


# ---------------------------------------------------------------------------
# Decoders (pyctcdecode)
# ---------------------------------------------------------------------------

def make_decoders(converter, word_lm=None, char_lm=None, lex_unigrams=None,
                  alpha_word=0.5, beta_word=1.5, alpha_char=0.3, beta_char=0.0):
    """Return (word_decoder, char_decoder).  Either may be None if its LM path
    isn't provided.  word_decoder is optionally lexicon-constrained via
    unigrams (the train lex vocab)."""
    from pyctcdecode import build_ctcdecoder
    labels = converter.pyctc_labels                       # ['', a..z]
    word_dec = build_ctcdecoder(
        labels, kenlm_model_path=word_lm,
        unigrams=lex_unigrams, alpha=alpha_word, beta=beta_word,
    ) if word_lm else None
    char_dec = build_ctcdecoder(
        labels, kenlm_model_path=char_lm,
        alpha=alpha_char, beta=beta_char,
    ) if char_lm else None
    return word_dec, char_dec


def greedy_decode(log_probs, converter):
    return converter.decode_ctc(log_probs.argmax(-1).tolist())


# ---------------------------------------------------------------------------
# Evaluation: per-subset, greedy vs LM
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_root, cache_root, split, converter, device,
             word_dec=None, char_dec=None, nonlex_mode="greedy",
             beam_width=64, use_amp=True):
    """nonlex_mode: 'greedy' or 'char' (char-LM).  lex always uses word_dec if
    provided else greedy."""
    model.eval()
    agg = {m: {s: {"e": 0, "l": 0, "n": 0} for s in ("lex", "nonlex")}
           for m in ("greedy", "lm")}
    for sub in ("lex", "nonlex"):
        for e in walk_clips(data_root, split, subsets=(sub,)):
            f = Path(cache_root) / split / sub / f"{clip_cache_name(e['signer'], e['clip_id'])}.npy"
            if not f.exists():
                continue
            feats = torch.from_numpy(np.load(f).astype(np.float32)).unsqueeze(0).to(device)
            in_lens = torch.LongTensor([feats.shape[1]]).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                lp = model(feats, in_lens)[0]              # [T, V]
            lp = lp.float().cpu().numpy()
            gt = gt_string(e["label"])

            g = greedy_decode(torch.from_numpy(lp), converter)
            if sub == "lex" and word_dec is not None:
                lm = word_dec.decode(lp, beam_width=beam_width).strip().replace(" ", "")
            elif sub == "nonlex" and nonlex_mode == "char" and char_dec is not None:
                lm = char_dec.decode(lp, beam_width=beam_width).strip().replace(" ", "")
            else:
                lm = g                                     # nonlex greedy (word LM would hurt)

            for m, pred in (("greedy", g), ("lm", lm)):
                ce, cl = cer_pair(gt, pred)
                agg[m][sub]["e"] += ce; agg[m][sub]["l"] += cl; agg[m][sub]["n"] += 1

    def cer(d, s): return d[s]["e"] / max(d[s]["l"], 1)
    def overall(d):
        e = d["lex"]["e"] + d["nonlex"]["e"]; l = d["lex"]["l"] + d["nonlex"]["l"]
        return e / max(l, 1)
    return {
        "split": split,
        "greedy": {"lex": cer(agg["greedy"], "lex"), "nonlex": cer(agg["greedy"], "nonlex"),
                   "overall": overall(agg["greedy"])},
        "lm":     {"lex": cer(agg["lm"], "lex"), "nonlex": cer(agg["lm"], "nonlex"),
                   "overall": overall(agg["lm"])},
        "n_clips": {s: agg["greedy"][s]["n"] for s in ("lex", "nonlex")},
        "paper_baseline": {"lex": 0.281, "nonlex": 0.365, "overall": 0.2924},
    }


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    model = TemporalCTC(backbone=state.get("backbone", "bilstm"),
                        d_model=state.get("d_model", 256),
                        n_layers=state.get("n_layers", 3)).to(device).eval()
    model.load_state_dict(state["model"])
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--lm_dir", type=str, required=True, help="dir with word.bin / char.bin / corpora")
    ap.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--alpha_word", type=float, default=0.5)
    ap.add_argument("--beta_word", type=float, default=1.5)
    ap.add_argument("--nonlex_mode", type=str, default="greedy", choices=["greedy", "char"])
    ap.add_argument("--lexicon_constrained", action="store_true")
    ap.add_argument("--beam_width", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = args.data_root or find_data_root()
    converter = CharConverter()

    split = args.eval_split
    marker = os.path.join(os.path.dirname(args.ckpt), f".stage16_lm_{split}_evaluated")
    if split == "test" and os.path.exists(marker):
        print(f"ERROR: test already evaluated ({marker}).", file=sys.stderr); sys.exit(2)

    # corpora (for lexicon unigrams)
    corp = build_corpora(data_root, args.lm_dir)
    word_bin = os.path.join(args.lm_dir, "word.bin")
    char_bin = os.path.join(args.lm_dir, "char.bin")
    word_lm = word_bin if os.path.isfile(word_bin) else None
    char_lm = char_bin if os.path.isfile(char_bin) else None
    if word_lm is None:
        print("WARNING: no word.bin found; lex will fall back to greedy. "
              "Build KenLM first (see module docstring).", flush=True)
    unigrams = corp["lex_words"] if args.lexicon_constrained else None
    word_dec, char_dec = make_decoders(converter, word_lm=word_lm, char_lm=char_lm,
                                       lex_unigrams=unigrams,
                                       alpha_word=args.alpha_word, beta_word=args.beta_word)

    model = load_model(args.ckpt, device)
    res = evaluate(model, data_root, args.cache_root, split, converter, device,
                   word_dec=word_dec, char_dec=char_dec, nonlex_mode=args.nonlex_mode,
                   beam_width=args.beam_width)
    res["config"] = {"alpha_word": args.alpha_word, "beta_word": args.beta_word,
                     "nonlex_mode": args.nonlex_mode,
                     "lexicon_constrained": args.lexicon_constrained}
    out = os.path.join(os.path.dirname(args.ckpt), f"lm_{split}_eval.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    if split == "test":
        open(marker, "w").write(out)

    print("\n" + "=" * 64)
    print(f"  STAGE 16 LM DECODE — {split.upper()}  -> {out}")
    print("=" * 64)
    print(f"  GREEDY  lex={res['greedy']['lex']:.4f}  nonlex={res['greedy']['nonlex']:.4f}"
          f"  overall={res['greedy']['overall']:.4f}")
    print(f"  +LM     lex={res['lm']['lex']:.4f}  nonlex={res['lm']['nonlex']:.4f}"
          f"  overall={res['lm']['overall']:.4f}")
    print(f"  paper   lex=0.281   nonlex=0.365   overall=0.2924")
    print("=" * 64)

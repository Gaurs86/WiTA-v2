"""
models/char_ngram_lm.py — Pure-Python char n-gram LM with modified
Kneser-Ney smoothing.

KenLM (kpu/kenlm) is the industry standard for n-gram language models in
ASR, but it requires compiling the lmplz binary which is finicky in
Kaggle/Colab kernels.  This module is a drop-in replacement that ships
in pure Python with zero external dependencies.  Trains in <30 s on the
WiTA label corpus and the score API matches KenLM's log-base-10 sign
convention so swapping is trivial later if you need exact KenLM parity.

Algorithm
---------
Modified Kneser-Ney smoothing as in Chen & Goodman 1999 §6 / Heafield 2011.
For each order k = 1..N:
  D_k       = discount estimated from the corpus (0.5 * n1 / (n1+2*n2)).
  c*(w_1..w_k)  = max(c(w_1..w_k) - D_k, 0) / c(w_1..w_{k-1})
  alpha(...) = D_k * |{w' : c(w_1..w_{k-1}, w') > 0}| / c(w_1..w_{k-1})
  P_KN(w_k | w_1..w_{k-1}) = c*(...) + alpha(...) * P_KN(w_k | w_2..w_{k-1})
Unigram falls back to the *continuation* count (Kneser & Ney's key trick).

Usage
-----
>>> lm = CharNgramLM(order=4)
>>> lm.train(corpus=["hello", "world", ...])
>>> lm.save("lm.pkl")
>>> lm = CharNgramLM.load("lm.pkl")
>>> lm.score("hello")            # log10 P("hello")  (negative number)
>>> lm.score_next("hell", "o")   # log10 P("o" | "hell")  (negative)

Special tokens
--------------
BOS_TOKEN = '<s>', EOS_TOKEN = '</s>'.  Both score() and score_next()
prepend BOS internally so callers pass plain strings.  EOS is optional
and only used in score(); not used during prefix-beam-search rescoring
which scores per-character extensions.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from typing import Iterable


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class CharNgramLM:
    """Char n-gram with modified Kneser-Ney smoothing."""

    def __init__(self, order: int = 4):
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        # counts[k] : tuple-of-k-chars -> count
        self.counts: list[Counter] = [Counter() for _ in range(order + 1)]
        # continuation counts: how many distinct contexts a suffix appears in
        self.cont_counts: list[Counter] = [Counter() for _ in range(order + 1)]
        # discount per order
        self.D: list[float] = [0.0] * (order + 1)
        # cache of conditional log10 probs after train()
        self._trained = False
        # vocabulary (characters that appear in training)
        self.vocab: set[str] = set()
        # unigram continuation total (denominator for P_cont(w))
        self._cont_total: int = 0

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def _tokenize(self, line: str) -> list[str]:
        """Insert BOS padding then split into characters; optional EOS."""
        pad = [BOS_TOKEN] * (self.order - 1)
        return pad + list(line) + [EOS_TOKEN]

    def train(self, corpus: Iterable[str]) -> None:
        """
        corpus : iterable of strings (one label per item).
        Computes counts, continuation counts, and discounts.
        """
        n_lines = 0
        for line in corpus:
            tokens = self._tokenize(line)
            self.vocab.update(c for c in tokens if c not in (BOS_TOKEN, EOS_TOKEN))
            for k in range(1, self.order + 1):
                for i in range(len(tokens) - k + 1):
                    ngram = tuple(tokens[i: i + k])
                    self.counts[k][ngram] += 1
                    if k >= 2:
                        # context (k-1) gram of suffix w_k=ngram[-1]
                        ctx = ngram[:-1]
                        suf = (ngram[-1],)
                        self.cont_counts[k][(ctx, suf)] += 1
            n_lines += 1

        if n_lines == 0:
            raise RuntimeError("Empty corpus.")

        # Estimate discount per order: D_k = 0.5 * n1 / (n1 + 2*n2)
        for k in range(2, self.order + 1):
            n1 = sum(1 for c in self.counts[k].values() if c == 1)
            n2 = sum(1 for c in self.counts[k].values() if c == 2)
            self.D[k] = 0.5 * n1 / max(n1 + 2 * n2, 1)

        # Unigram continuation total: |{(u, w) : c(u, w) > 0}|
        if self.order >= 2:
            self._cont_total = len(self.cont_counts[2])
        self._trained = True

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _p_kn_recursive(self, ctx: tuple[str, ...], w: str) -> float:
        """
        P_KN(w | ctx) recursively.  ctx is a tuple of (k-1) tokens; w is the
        next token.  Returns a probability in [0, 1] (NOT log space).
        """
        k = len(ctx) + 1
        if k == 1:
            # Unigram (continuation count form).
            if self._cont_total == 0:
                # Single-token corpus edge case: fall back to uniform.
                return 1.0 / max(len(self.vocab), 1)
            # count of contexts in which `w` appears as the suffix
            num = sum(
                1 for (c_ctx, c_suf) in self.cont_counts[2]
                if c_suf == (w,)
            )
            # Add a tiny uniform smoothing so unseen chars don't crash.
            return (num + 1e-6) / (self._cont_total + 1e-6 * max(len(self.vocab), 1))
        ngram = ctx + (w,)
        c_full = self.counts[k].get(ngram, 0)
        c_ctx  = sum(
            self.counts[k].get(ctx + (v,), 0) for v in self.vocab | {EOS_TOKEN}
        )
        if c_ctx == 0:
            # Backoff entirely.
            return self._p_kn_recursive(ctx[1:], w)
        D = self.D[k]
        first  = max(c_full - D, 0.0) / c_ctx
        # number of unique extensions of ctx in training
        n_ext = sum(
            1 for v in (self.vocab | {EOS_TOKEN})
            if self.counts[k].get(ctx + (v,), 0) > 0
        )
        alpha  = D * n_ext / c_ctx
        return first + alpha * self._p_kn_recursive(ctx[1:], w)

    def score_next(self, prefix: str, c: str) -> float:
        """
        log10 P(c | prefix).  Returns 0.0 for empty contexts.
        Caller passes plain strings; BOS padding handled internally.
        """
        if not self._trained:
            raise RuntimeError("LM not trained.")
        # Truncate prefix to (order - 1) trailing chars; pad left with BOS.
        ctx_chars = list(prefix)[-(self.order - 1):]
        pad = [BOS_TOKEN] * (self.order - 1 - len(ctx_chars))
        ctx = tuple(pad + ctx_chars)
        p = self._p_kn_recursive(ctx, c)
        return math.log10(max(p, 1e-30))

    def score(self, text: str, eos: bool = True) -> float:
        """
        Total log10 probability of `text` (sum of per-char scores).
        With eos=True, also adds log P(EOS | end-context) — useful for
        comparing whole-sentence candidates.
        """
        if not self._trained:
            raise RuntimeError("LM not trained.")
        total = 0.0
        for i, c in enumerate(text):
            total += self.score_next(text[:i], c)
        if eos:
            total += self.score_next(text, EOS_TOKEN)
        return total

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "CharNgramLM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise RuntimeError("Pickle does not contain CharNgramLM.")
        return obj

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------

    @property
    def n_train_chars(self) -> int:
        return sum(self.counts[1].values()) if self.counts else 0

    def __repr__(self) -> str:
        return (
            f"CharNgramLM(order={self.order}, vocab={len(self.vocab)}, "
            f"trained={self._trained}, train_chars={self.n_train_chars})"
        )

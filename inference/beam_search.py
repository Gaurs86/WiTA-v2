"""
inference/beam_search.py — CTC prefix beam search with LM shallow fusion.

Implements the Graves & Jaitly (2014) CTC prefix beam search with an
optional language-model weighting term:

    score(prefix) = log p_ctc(prefix | x) + alpha * log p_lm(prefix)
                                          + beta  * |prefix|

  * alpha : LM weight (typical 0.3–1.5 for char LMs).
  * beta  : word-insertion / length bonus (typical 0.0–1.0).  Counteracts
            the LM's tendency to prefer short outputs.

Algorithm per timestep maintains two prefix-conditioned probabilities:
    p_b(prefix)  = ending in blank
    p_nb(prefix) = ending in a non-blank that we just emitted
At every timestep we extend each surviving prefix by every symbol that has
non-trivial probability at that timestep.  CTC's collapse rule (consecutive
identical non-blanks merge unless separated by a blank) is handled inside
the update equations.

This is a numerically careful implementation in log-space and with a beam
pruner that keeps only the top-`beam` prefixes by joint score after each
timestep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NEG_INF = -1e30


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------

@dataclass
class _Beam:
    prefix: tuple[int, ...] = field(default_factory=tuple)   # token ids
    log_pb: float = NEG_INF        # log P(prefix, ending in blank)
    log_pnb: float = NEG_INF       # log P(prefix, ending in non-blank)
    log_lm: float = 0.0            # cached log10 LM score

    @property
    def log_p_ctc(self) -> float:
        return _logaddexp(self.log_pb, self.log_pnb)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def ctc_prefix_beam_search(
    log_probs: np.ndarray,         # [T, V] log softmax probabilities
    *,
    blank: int = 0,
    sep:   Optional[int] = None,   # CTC repeat separator; skipped in id_to_char output
    beam:  int = 8,                # smaller default — char-level rarely needs > 8
    id_to_char: Optional[dict[int, str]] = None,
    lm = None,                     # CharNgramLM or None
    alpha: float = 0.5,            # LM weight (ln-space conversion handled internally)
    beta:  float = 0.0,            # length bonus
    blank_threshold: float = 0.0,  # skip extending past timesteps where blank > thr
    symbol_top_k: Optional[int] = 10,  # only expand top-K likely symbols per step
) -> tuple[str, list[int]]:
    """
    Returns (decoded_string, decoded_id_list).  When `id_to_char` is None
    the decoded_string is empty and the caller is responsible for mapping
    ids to characters.

    The LM scores log10; we convert to natural log internally so `alpha`
    is in nats-per-character.
    """
    T, V = log_probs.shape
    assert 0 <= blank < V, "blank index out of range"
    # Defensive int casts — JSON-deserialised sweep configs hand us floats.
    beam = int(beam)
    if symbol_top_k is not None:
        symbol_top_k = int(symbol_top_k)
    LN10 = math.log(10.0)

    init = _Beam(prefix=tuple(), log_pb=0.0, log_pnb=NEG_INF, log_lm=0.0)
    beams: dict[tuple[int, ...], _Beam] = {tuple(): init}

    for t in range(T):
        lp_t = log_probs[t]
        # Skip extension at strongly-blank timesteps (cheap acceleration).
        if blank_threshold > 0.0 and math.exp(lp_t[blank]) >= blank_threshold:
            symbols_to_consider = [blank]
        elif symbol_top_k is not None and symbol_top_k < V:
            # Restrict to top-K symbols by log-prob this timestep, but
            # always include the blank token (CTC depends on it for
            # repeat segregation and blank-emission).
            top_idx = np.argpartition(lp_t, -symbol_top_k)[-symbol_top_k:]
            symbols_to_consider = set(int(i) for i in top_idx)
            symbols_to_consider.add(blank)
            symbols_to_consider = list(symbols_to_consider)
        else:
            symbols_to_consider = list(range(V))

        new_beams: dict[tuple[int, ...], _Beam] = {}

        for prefix, beam_obj in beams.items():
            for s in symbols_to_consider:
                lps = lp_t[s]
                if lps <= -25:    # skip absurdly low symbol log-probs
                    continue
                if s == blank:
                    # Extend with blank: prefix unchanged.
                    new_log_pb = _logaddexp(
                        new_beams.get(prefix, _Beam(prefix=prefix)).log_pb,
                        beam_obj.log_p_ctc + lps,
                    )
                    nb = new_beams.get(prefix)
                    if nb is None:
                        nb = _Beam(prefix=prefix, log_lm=beam_obj.log_lm)
                        new_beams[prefix] = nb
                    nb.log_pb = new_log_pb
                else:
                    # Extend with non-blank symbol s.
                    last = prefix[-1] if prefix else None
                    if s == last:
                        # Repeat: must have had a blank in between to count.
                        new_prefix = prefix       # collapses
                        nb = new_beams.get(new_prefix)
                        if nb is None:
                            nb = _Beam(prefix=new_prefix, log_lm=beam_obj.log_lm)
                            new_beams[new_prefix] = nb
                        nb.log_pnb = _logaddexp(nb.log_pnb,
                                                beam_obj.log_pb + lps)
                        # ALSO non-collapsing extension: append duplicate via blank-emit
                        new_prefix2 = prefix + (s,)
                        nb2 = new_beams.get(new_prefix2)
                        new_lm = beam_obj.log_lm
                        if lm is not None and id_to_char is not None and s in id_to_char:
                            c_str = id_to_char[s]
                            prev_str = "".join(id_to_char[i] for i in prefix
                                               if i in id_to_char)
                            new_lm = beam_obj.log_lm + lm.score_next(prev_str, c_str)
                        if nb2 is None:
                            nb2 = _Beam(prefix=new_prefix2, log_lm=new_lm)
                            new_beams[new_prefix2] = nb2
                        nb2.log_pnb = _logaddexp(nb2.log_pnb,
                                                 beam_obj.log_pb + lps)
                    else:
                        new_prefix = prefix + (s,)
                        nb = new_beams.get(new_prefix)
                        new_lm = beam_obj.log_lm
                        if lm is not None and id_to_char is not None and s in id_to_char:
                            c_str = id_to_char[s]
                            prev_str = "".join(id_to_char[i] for i in prefix
                                               if i in id_to_char)
                            new_lm = beam_obj.log_lm + lm.score_next(prev_str, c_str)
                        if nb is None:
                            nb = _Beam(prefix=new_prefix, log_lm=new_lm)
                            new_beams[new_prefix] = nb
                        else:
                            # Multiple paths can land on the same prefix; keep
                            # the LM score (it's a function of prefix, not path).
                            nb.log_lm = new_lm
                        nb.log_pnb = _logaddexp(nb.log_pnb,
                                                beam_obj.log_p_ctc + lps)

        # Prune to top `beam` by joint score.
        def joint_score(b: _Beam) -> float:
            ctc = b.log_p_ctc
            if lm is None:
                lm_score = 0.0
            else:
                lm_score = alpha * LN10 * b.log_lm
            return ctc + lm_score + beta * len(b.prefix)

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda kv: joint_score(kv[1]),
                   reverse=True)[:beam]
        )

    # Final: pick beam with highest joint score (with EOS bonus from LM).
    def final_joint(b: _Beam) -> float:
        ctc = b.log_p_ctc
        lm_score = 0.0
        if lm is not None and id_to_char is not None:
            prev_str = "".join(id_to_char[i] for i in b.prefix
                               if i in id_to_char)
            lm_score = alpha * LN10 * (b.log_lm + lm.score_next(prev_str, "</s>"))
        return ctc + lm_score + beta * len(b.prefix)

    best = max(beams.values(), key=final_joint)
    ids = list(best.prefix)
    if id_to_char is None:
        return "", ids
    # If `sep` is provided, drop separator tokens from the output (CTC
    # repeat separator is internal to encoding, not part of the label).
    decoded = "".join(
        id_to_char[i] for i in ids
        if i in id_to_char and (sep is None or i != sep)
    )
    return decoded, ids

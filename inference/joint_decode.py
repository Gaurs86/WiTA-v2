"""
inference/joint_decode.py — decoder-side fixes for Stage 11.

Three building blocks, none requiring retraining:

  1. attention_beam_search() — top-K beam search through the existing
     attention decoder with length normalisation (B1).
  2. ctc_sequence_log_prob() — CTC forward algorithm: log P(target | x)
     summed over alignments.  Used to score attention-search candidates
     against the CTC posterior (B2 / B3).
  3. joint_decode() — runs attention beam search, then rescores the top-K
     candidates with the combined CTC + attention + LM score (B3).

All functions operate on a single batch element at a time so the caller
loops over the val/test loader.  Vectorising across the batch is doable
but the per-clip cost is already <100 ms on T4 for beam=8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


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
# CTC forward algorithm — log P(target | log_probs) summed over alignments
# ---------------------------------------------------------------------------

def ctc_sequence_log_prob(
    log_probs: np.ndarray,        # [T, V] log-softmax CTC posterior
    target:    list[int],          # token ids of the target sequence
    blank:     int,
) -> float:
    """
    Compute log P(target | log_probs) under CTC, summed over all valid
    alignments.  Implements the standard Graves CTC forward recursion in
    log-space.  O(T * 2L) time, single-thread Python.

    Returns -inf if target cannot be produced in T steps.
    """
    T, V = log_probs.shape
    L = len(target)
    if L == 0:
        # Empty target: only the all-blank alignment is valid.
        return float(np.sum(log_probs[:, blank]))
    # Extended target: [b, t_1, b, t_2, b, ..., b, t_L, b], length 2L+1.
    ext: list[int] = [blank]
    for t in target:
        ext.append(int(t)); ext.append(blank)
    Lext = len(ext)
    if Lext > T + 1:
        # Not enough time steps to emit all the symbols + their blanks.
        # Roughly: need T >= 2L_eff - 1 where L_eff is L minus duplicate
        # collapses; here we require T >= L+1 as the cheap check.
        # (CTC forward will return -inf in that case anyway.)
        pass

    alpha = np.full((T, Lext), NEG_INF, dtype=np.float64)
    alpha[0, 0] = float(log_probs[0, ext[0]])
    if Lext > 1:
        alpha[0, 1] = float(log_probs[0, ext[1]])
    for t in range(1, T):
        for s in range(Lext):
            sym = ext[s]
            prev = alpha[t-1, s]
            if s >= 1:
                prev = _logaddexp(prev, alpha[t-1, s-1])
            if s >= 2 and sym != blank and sym != ext[s-2]:
                prev = _logaddexp(prev, alpha[t-1, s-2])
            alpha[t, s] = prev + float(log_probs[t, sym])
    return _logaddexp(alpha[-1, -1], alpha[-1, -2]) if Lext >= 2 else float(alpha[-1, -1])


# ---------------------------------------------------------------------------
# Attention beam search (B1)
# ---------------------------------------------------------------------------

@dataclass
class _AttnBeam:
    ids:        list[int] = field(default_factory=list)   # excludes BOS
    log_p:      float = 0.0
    finished:   bool = False

    def length_normed(self, alpha: float) -> float:
        L = max(len(self.ids), 1)
        return self.log_p / (L ** alpha)


@torch.no_grad()
def attention_beam_search(
    decoder,
    memory:      torch.Tensor,            # [1, T, d] encoder memory
    memory_pad:  Optional[torch.Tensor],  # [1, T] bool
    *,
    beam_width:  int = 8,
    alpha:       float = 0.7,            # length-norm exponent
    max_len:     Optional[int] = None,
) -> list[_AttnBeam]:
    """
    Autoregressive beam search through `decoder`.  Returns the surviving
    `beam_width` candidate sequences sorted by length-normalised score.

    `decoder` must expose:
        .bos, .eos, .att_vocab_size, .forward(memory, memory_pad, tgt_in)
    matching `models/attention_decoder.AttentionDecoder`.
    """
    device   = memory.device
    bos, eos = decoder.bos, decoder.eos
    Vsize    = decoder.att_vocab_size
    beam_width = int(beam_width)
    max_len  = int(max_len or decoder.max_decode_len)

    # Start with one beam containing only BOS.
    beams = [_AttnBeam(ids=[], log_p=0.0)]
    for step in range(max_len):
        if all(b.finished for b in beams):
            break
        # Build a [B', L] tensor of decoder inputs across the live beams.
        live = [b for b in beams if not b.finished]
        done = [b for b in beams if     b.finished]
        # Pad each live beam's input to the same length.
        max_len_b = max(len(b.ids) for b in live) + 1   # +1 for BOS
        dec_in = torch.full((len(live), max_len_b), bos,
                            dtype=torch.long, device=device)
        for i, b in enumerate(live):
            for j, tok in enumerate(b.ids):
                dec_in[i, j + 1] = tok
        # Repeat memory across live beams.
        mem = memory.expand(len(live), -1, -1)
        pad = memory_pad.expand(len(live), -1) if memory_pad is not None else None
        logits = decoder(mem, pad, dec_in)                  # [B', max_len_b, V]
        # Use the LAST RELEVANT step per beam, but the dec_in has same length
        # across beams (post-pad).  The next-token logits live at the position
        # of the most recently EMITTED token, i.e. len(b.ids) for beam b
        # (after the BOS at position 0).  This is the same for all beams in
        # this step thanks to our padding -- max_len_b - 1.
        next_step = max_len_b - 1
        step_logits = logits[:, next_step, :]               # [B', V]
        log_probs   = F.log_softmax(step_logits, dim=-1)    # [B', V]

        # Expand each live beam by top-(beam_width) continuations.
        new_beams: list[_AttnBeam] = list(done)
        topv, topi = torch.topk(log_probs, k=min(beam_width, Vsize), dim=-1)
        topv = topv.cpu().numpy()
        topi = topi.cpu().numpy()
        for i, b in enumerate(live):
            for k in range(topv.shape[1]):
                tok = int(topi[i, k])
                lp  = float(topv[i, k])
                if tok == eos:
                    nb = _AttnBeam(ids=list(b.ids), log_p=b.log_p + lp,
                                   finished=True)
                else:
                    nb = _AttnBeam(ids=b.ids + [tok], log_p=b.log_p + lp,
                                   finished=False)
                new_beams.append(nb)

        # Prune to top `beam_width` by LENGTH-NORMALISED score.
        new_beams.sort(key=lambda b: b.length_normed(alpha), reverse=True)
        beams = new_beams[:beam_width]

    # If nothing finished, return the top of the live beams.
    beams.sort(key=lambda b: b.length_normed(alpha), reverse=True)
    return beams


# ---------------------------------------------------------------------------
# Joint rescoring (B2 / B3)
# ---------------------------------------------------------------------------

@dataclass
class JointHypothesis:
    ids:           list[int]
    attn_log_p:    float
    ctc_log_p:     float = NEG_INF
    lm_log_p:      float = 0.0
    combined:      float = 0.0
    decoded:       str   = ""


def joint_rescore(
    candidates:       list[_AttnBeam],
    ctc_log_probs:    np.ndarray,         # [T, V] CTC log-softmax
    *,
    blank:            int,
    alpha_ctc:        float = 0.5,
    alpha_attn:       float = 0.5,
    alpha_lm:         float = 0.0,
    gamma_len:        float = 0.0,
    lm = None,                            # CharNgramLM or None
    id_to_char:       Optional[dict[int, str]] = None,
    sep_idx:          Optional[int] = None,
) -> list[JointHypothesis]:
    """
    For each candidate produced by attention_beam_search, compute:
      - attn_log_p (already given by the candidate)
      - ctc_log_p  (CTC forward algorithm on ctc_log_probs)
      - lm_log_p   (LM total log10 prob of the decoded string)
    Return the same list, sorted by `combined` descending.

    `combined = alpha_attn * attn + alpha_ctc * ctc + alpha_lm * LN10 * lm
              + gamma_len * len(ids)`
    """
    LN10 = math.log(10.0)
    ctc_V = ctc_log_probs.shape[1]            # CTC vocab size (28 for English)
    out: list[JointHypothesis] = []
    for cand in candidates:
        ids = list(cand.ids)
        attn_lp = float(cand.log_p)
        # The attention beam may occasionally emit BOS/EOS/PAD tokens
        # (indices 28, 29, 30 for English) which don't exist in the CTC
        # vocab.  Filter to char-only (1 <= i < ctc_V) before scoring.
        # We score the displayed string under CTC; PAD/BOS at the
        # boundaries are noise either way.
        ids_ctc = [i for i in ids if 1 <= i < ctc_V]
        ctc_lp  = ctc_sequence_log_prob(ctc_log_probs, ids_ctc, blank=blank) \
                  if alpha_ctc != 0.0 else NEG_INF
        decoded = ""
        if id_to_char is not None:
            decoded = "".join(
                id_to_char[i] for i in ids
                if i in id_to_char and (sep_idx is None or i != sep_idx)
            )
        lm_lp = 0.0
        if lm is not None and id_to_char is not None and decoded:
            lm_lp = float(lm.score(decoded, eos=True))  # log10
        combined = (alpha_attn * attn_lp
                    + (alpha_ctc * ctc_lp if ctc_lp != NEG_INF else 0.0)
                    + alpha_lm * LN10 * lm_lp
                    + gamma_len * len(ids))
        out.append(JointHypothesis(
            ids=ids, attn_log_p=attn_lp, ctc_log_p=ctc_lp,
            lm_log_p=lm_lp, combined=combined, decoded=decoded,
        ))
    out.sort(key=lambda h: h.combined, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Convenience: decode one clip with a configurable strategy
# ---------------------------------------------------------------------------

def decode_one_clip(
    encoder,
    decoder,
    feats:           torch.Tensor,           # [1, T_in, D]
    in_lens:         torch.Tensor,           # [1] long
    *,
    cfg,
    mode:            str = "joint",          # 'ctc_greedy'|'attn_greedy'|'ctc_beam'|'ctc_lm_beam'|'attn_beam'|'joint'
    beam_width:      int = 8,
    length_alpha:    float = 0.7,
    alpha_ctc:       float = 0.5,
    alpha_attn:      float = 0.5,
    alpha_lm:        float = 0.5,
    gamma_len:       float = 0.0,
    lm = None,
    ctc_lm_alpha:    float = 0.5,
    ctc_lm_beta:     float = 0.0,
    ctc_lm_beam:     int = 8,                  # was 32 — too slow for char-level
    ctc_lm_symbol_top_k: int = 10,             # only expand top-K likely symbols per step
) -> str:
    """
    Decode a single clip via the requested strategy.  Returns the decoded
    string (characters only — separator and special tokens stripped).
    """
    device = feats.device
    chars  = cfg.vocab.chars
    blank  = cfg.vocab.blank_idx
    sep    = cfg.vocab.sep_idx
    id_to_char = {i + 1: c for i, c in enumerate(chars)}
    # Defensive int casts — sweep configs sometimes hand us 10.0 instead of 10.
    beam_width          = int(beam_width)
    ctc_lm_beam         = int(ctc_lm_beam)
    ctc_lm_symbol_top_k = int(ctc_lm_symbol_top_k)

    with torch.no_grad():
        h, pad_mask = encoder.encode(feats, in_lens)
        log_probs, enc_lens = encoder.decode_ctc(h, in_lens)
    lp_np = log_probs[0].float().cpu().numpy()

    def _ids_to_str(ids):
        return "".join(id_to_char[i] for i in ids
                       if i in id_to_char and i != sep)

    if mode == "ctc_greedy":
        argmax = lp_np.argmax(axis=-1)
        merged, prev = [], None
        for t in argmax:
            t = int(t)
            if t != prev and t != blank:
                merged.append(t)
            prev = t
        return _ids_to_str(merged)

    if mode == "attn_greedy":
        ids = decoder.greedy_decode(h, pad_mask)[0].tolist()
        return _ids_to_str(ids)

    if mode == "ctc_beam":
        # CTC beam without LM — diagnostic baseline.  If this doesn't beat
        # ctc_greedy, the issue is the algorithm itself, not the LM.
        from .beam_search import ctc_prefix_beam_search
        decoded, _ = ctc_prefix_beam_search(
            lp_np, blank=blank, sep=sep, beam=ctc_lm_beam,
            id_to_char=id_to_char, lm=None,
            alpha=0.0, beta=ctc_lm_beta,
            symbol_top_k=ctc_lm_symbol_top_k,
        )
        return decoded

    if mode == "ctc_lm_beam":
        from .beam_search import ctc_prefix_beam_search
        decoded, _ = ctc_prefix_beam_search(
            lp_np, blank=blank, sep=sep, beam=ctc_lm_beam,
            id_to_char=id_to_char, lm=lm,
            alpha=ctc_lm_alpha, beta=ctc_lm_beta,
            symbol_top_k=ctc_lm_symbol_top_k,
        )
        return decoded

    if mode == "attn_beam":
        beams = attention_beam_search(
            decoder, h, pad_mask,
            beam_width=beam_width, alpha=length_alpha,
        )
        return _ids_to_str(beams[0].ids) if beams else ""

    if mode == "joint":
        beams = attention_beam_search(
            decoder, h, pad_mask,
            beam_width=beam_width, alpha=length_alpha,
        )
        if not beams:
            return ""
        hyps = joint_rescore(
            beams, lp_np, blank=blank,
            alpha_ctc=alpha_ctc, alpha_attn=alpha_attn, alpha_lm=alpha_lm,
            gamma_len=gamma_len, lm=lm,
            id_to_char=id_to_char, sep_idx=sep,
        )
        return hyps[0].decoded if hyps else ""

    raise ValueError(f"Unknown decode mode: {mode}")

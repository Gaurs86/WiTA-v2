"""
stage18/audit.py — CPU-only data/lexicon/tracking audit (the "do first" step).

Answers, before any retraining or GPU:
  1. DATA: how many clips / signer-dirs / unique signers / views exist per
     split-subset, in the RAW dataset vs the landmark cache.  Detects multi-view
     (a signer-id with >1 session dir) and whether the cache uses every clip.
  2. LEXICON OOV: are the lexical labels in a fixed external vocabulary
     (wordfreq top-K as a Google-1B proxy)?  Reports per-split OOV + the actual
     lexical-vocab size + train/test word overlap -> tells us if a 6000-word
     trie-beam can help (and that it's leakage-free).
  3. TRACKING: per-clip fingertip jitter / max-jump / %low-confidence from the
     cached features, aggregated per signer -> later correlate with per-signer
     CER to see if tracking (not style) is the ceiling.
"""

from __future__ import annotations

import os
import re
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import walk_clips, find_data_root                 # noqa: E402
from stage17.common import iter_clips, load_clip_npz, fingertip_xy     # noqa: E402

_SIGNER_RE = re.compile(r"^(?P<signer>.+?)_(?:Male|Female)_\d+", re.IGNORECASE)


def signer_id(dirname: str) -> str:
    m = _SIGNER_RE.match(dirname)
    return m.group("signer").upper() if m else dirname


# ---------------------------------------------------------------------------
# 1. DATA structure / multi-view detection
# ---------------------------------------------------------------------------

def audit_structure(raw_root=None):
    raw_root = raw_root or find_data_root()
    print(f"[audit] raw_root = {raw_root}\n")
    print(f"{'split':6} {'subset':7} {'clips':>7} {'dirs':>6} {'signers':>8} {'maxdirs/signer':>15}")
    grand = {}
    for split in ("train", "val", "test"):
        for sub in ("lex", "nonlex"):
            clips = walk_clips(raw_root, split, subsets=(sub,))
            dirs = sorted({c["signer"] for c in clips})
            per = defaultdict(set)
            for d in dirs:
                per[signer_id(d)].add(d)
            maxd = max((len(v) for v in per.values()), default=0)
            print(f"{split:6} {sub:7} {len(clips):7d} {len(dirs):6d} {len(per):8d} {maxd:15d}")
            grand[(split, sub)] = {"clips": len(clips), "dirs": len(dirs),
                                   "signers": len(per), "max_dirs_per_signer": maxd}
    # view/session suffixes: the part of the dir name AFTER <signer>_<gender>_<age>
    suff = defaultdict(int)
    for c in walk_clips(raw_root, "train", subsets=("lex", "nonlex")):
        m = _SIGNER_RE.match(c["signer"])
        suff[c["signer"][m.end():].lstrip("_") if m else "?"] += 1
    print("\n[audit] distinct dir-name suffixes after <signer>_<gender>_<age> (train):")
    for s, n in sorted(suff.items(), key=lambda x: -x[1])[:12]:
        print(f"    {s!r:30} {n}")
    anymulti = any(v["max_dirs_per_signer"] > 1 for v in grand.values())
    print(f"\n[audit] MULTI-VIEW/SESSION present (a signer with >1 dir): {anymulti}")
    print("    -> if False, item 1 is moot: there are no extra views to add.")
    return grand


def audit_cache_vs_raw(raw_root=None, cache_root=None):
    raw_root = raw_root or find_data_root()
    from stage17.common import find_landmark_cache
    cache_root = cache_root or find_landmark_cache()
    print(f"\n[audit] cache_root = {cache_root}")
    print(f"{'split':6} {'subset':7} {'raw_clips':>9} {'cached_npz':>11} {'coverage':>9}")
    for split in ("train", "val", "test"):
        for sub in ("lex", "nonlex"):
            raw_n = len(walk_clips(raw_root, split, subsets=(sub,)))
            cac_n = len(iter_clips(cache_root, split, (sub,)))
            cov = cac_n / max(raw_n, 1)
            print(f"{split:6} {sub:7} {raw_n:9d} {cac_n:11d} {cov:8.1%}")


# ---------------------------------------------------------------------------
# 2. LEXICON OOV
# ---------------------------------------------------------------------------

def audit_lexicon(raw_root=None, top_k=6000):
    raw_root = raw_root or find_data_root()
    lex = {}
    for split in ("train", "val", "test"):
        words = sorted({c["label"].strip().lower()
                        for c in walk_clips(raw_root, split, subsets=("lex",))
                        if c["label"].strip()})
        lex[split] = set(words)
    union = lex["train"] | lex["val"] | lex["test"]
    print(f"\n[audit] lexical vocabulary: train={len(lex['train'])} val={len(lex['val'])} "
          f"test={len(lex['test'])} | UNION={len(union)} unique words")
    # train/test word overlap (why Stage 17b's train-lexicon was only ~49%)
    ov = len(lex["test"] & lex["train"]) / max(len(lex["test"]), 1)
    print(f"[audit] test-lex words also in TRAIN lex: {ov:.1%}  (Stage 17b coverage)")
    # external fixed vocab (Google-1B proxy = wordfreq top-K)
    try:
        from wordfreq import top_n_list
        ext = {w for w in top_n_list("en", top_k)
               if w.isascii() and w.isalpha()}
        for split in ("train", "val", "test"):
            oov = sum(1 for w in lex[split] if w not in ext) / max(len(lex[split]), 1)
            print(f"[audit] {split} lex OOV vs wordfreq top-{top_k}: {oov:.1%} "
                  f"(coverage {1-oov:.1%})")
        print("    -> low test OOV => a fixed-vocab trie-beam (item 2) is viable & leakage-free.")
    except ImportError:
        print("[audit] wordfreq not installed (pip install wordfreq) — skipping external OOV.")
    return lex, union


# ---------------------------------------------------------------------------
# 3. TRACKING quality (from cached features)
# ---------------------------------------------------------------------------

def audit_tracking(cache_root=None, out_json=None):
    from stage17.common import find_landmark_cache
    cache_root = cache_root or find_landmark_cache()
    per_signer = defaultdict(lambda: {"n": 0, "speed": [], "jerk": [], "lowconf": [], "det": []})
    rows = []
    for split in ("train", "val", "test"):
        for c in iter_clips(cache_root, split, ("lex", "nonlex")):
            d = load_clip_npz(c["npz"]); f = d["feature"]
            xy = fingertip_xy(f)                                  # [T,2]
            vel = np.diff(xy, axis=0)
            speed = np.hypot(vel[:, 0], vel[:, 1]) if len(vel) else np.array([0.0])
            jerk = np.abs(np.diff(speed)) if len(speed) > 1 else np.array([0.0])
            vis = f[:, 189] if f.shape[1] > 189 else np.ones(len(f))
            sid = signer_id(d["signer"])
            s = per_signer[sid]
            s["n"] += 1
            s["speed"].append(float(speed.max())); s["jerk"].append(float(jerk.mean()))
            s["lowconf"].append(float((vis < 0.5).mean())); s["det"].append(float(d["detected"]))
            rows.append({"signer": sid, "clip": c["stem"], "split": split,
                         "subset": d["subset"], "label": d["label"],
                         "max_speed": float(speed.max()), "mean_jerk": float(jerk.mean()),
                         "lowconf_frac": float((vis < 0.5).mean()), "detected": float(d["detected"])})
    print(f"\n[audit] tracking stats over {len(rows)} clips, {len(per_signer)} signers")
    print(f"{'signer':10} {'n':>4} {'maxspeed':>9} {'jerk':>8} {'lowconf%':>9} {'det':>6}")
    agg = []
    for sid, s in per_signer.items():
        agg.append((sid, s["n"], np.mean(s["speed"]), np.mean(s["jerk"]),
                    np.mean(s["lowconf"]) * 100, np.mean(s["det"])))
    for sid, n, sp, jk, lc, dt in sorted(agg, key=lambda x: -x[4])[:15]:   # worst lowconf
        print(f"{sid:10} {n:4d} {sp:9.4f} {jk:8.5f} {lc:9.1f} {dt:6.2f}")
    if out_json:
        json.dump(rows, open(out_json, "w"), default=float)
        print(f"[audit] per-clip tracking stats -> {out_json} "
              f"(join with per-clip CER to correlate)")
    return rows

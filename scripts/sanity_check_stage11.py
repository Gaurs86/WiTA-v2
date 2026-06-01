"""
scripts/sanity_check_stage11.py — eight pre-launch checks for Stage 11.

Per §4 of the Stage 11 prompt: catch silent bugs in <2 min so we don't
waste 4 GPU-hours on a contaminated headline run.

Usage:
    python scripts/sanity_check_stage11.py \
        --cache-root /kaggle/input/wita-full-english-landmark-cache/landmark_cache_122 \
        --old-cache  /kaggle/input/wita-38-skeleton-cache/skeleton_features_t32.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


def check_1_feature_distribution(cache_root: Path, old_cache_pt: str | None):
    """Check 1: new 122-signer cache statistics match the 38-signer cache."""
    new_pat = list((cache_root / "train" / "lex").glob("*.npz"))[:200]
    if not new_pat:
        return False, "No .npz files under train/lex"
    new_arr = np.stack([
        np.load(p, allow_pickle=False)["feature"].astype(np.float32)
        for p in random.sample(new_pat, min(100, len(new_pat)))
    ])
    new_mean, new_std = float(new_arr.mean()), float(new_arr.std())
    msg = f"new mean={new_mean:.4f}  std={new_std:.4f}"
    if old_cache_pt and os.path.exists(old_cache_pt):
        import torch
        old = torch.load(old_cache_pt, map_location="cpu", weights_only=False)
        feats_list = old["feats"]
        idx = random.sample(range(len(feats_list)), min(100, len(feats_list)))
        old_arr = np.stack([feats_list[i].float().numpy() for i in idx])
        old_mean, old_std = float(old_arr.mean()), float(old_arr.std())
        msg += f"  ||  old mean={old_mean:.4f}  std={old_std:.4f}"
        ok = (abs(new_mean - old_mean) <= 0.10 * abs(old_mean) + 1e-3
              and abs(new_std  - old_std)  <= 0.10 * abs(old_std)  + 1e-3)
        if not ok:
            msg += "  (DRIFT >10%; pin MediaPipe version)"
        return ok, msg
    return True, msg + "  (no old cache to compare; skipping drift check)"


def check_2_sample_counts(cache_root: Path):
    """Check 2: train ~ 0.8, val ~ 0.1, test ~ 0.1."""
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for subset in ("lex", "nonlex"):
            d = cache_root / split / subset
            counts[split][subset] = len(list(d.glob("*.npz"))) if d.exists() else 0
    total = sum(sum(v.values()) for v in counts.values())
    if total == 0:
        return False, "Cache is empty"
    fractions = {s: sum(v.values()) / total for s, v in counts.items()}
    msg = (f"counts={counts}  total={total}  "
           f"fractions={ {s: round(f,3) for s,f in fractions.items()} }")
    ok = (0.70 <= fractions["train"] <= 0.90
          and 0.05 <= fractions["val"]   <= 0.15
          and 0.05 <= fractions["test"]  <= 0.15)
    if not ok:
        msg += "  (fractions outside 0.8/0.1/0.1 ±0.05)"
    return ok, msg


def check_3_signer_disjoint(cache_root: Path):
    """Check 3: train, val, test signer sets are disjoint."""
    from wita_v2.datasets.landmark_paper_split import collect_signers
    sets = {s: collect_signers(cache_root, s) for s in ("train", "val", "test")}
    tr_va = sets["train"] & sets["val"]
    tr_te = sets["train"] & sets["test"]
    va_te = sets["val"]   & sets["test"]
    overlaps = {"train∩val": sorted(tr_va), "train∩test": sorted(tr_te),
                "val∩test":  sorted(va_te)}
    ok = not (tr_va or tr_te or va_te)
    return ok, (f"train={len(sets['train'])}  val={len(sets['val'])}  "
                f"test={len(sets['test'])}  overlaps={overlaps}")


def check_4_no_clip_leakage(cache_root: Path):
    """Check 4: no clip ID appears in both train and test."""
    from wita_v2.datasets.landmark_paper_split import collect_clip_ids
    tr = collect_clip_ids(cache_root, "train")
    te = collect_clip_ids(cache_root, "test")
    overlap = sorted(tr & te)
    ok = not overlap
    return ok, f"|train|={len(tr)}  |test|={len(te)}  overlap={len(overlap)}"


def check_5_vocab_coverage(cache_root: Path):
    """Check 5: every char in val+test labels is in the training vocabulary."""
    from wita_v2.datasets.vocab import make_converter
    conv = make_converter("english")
    train_chars: set[str] = set()
    eval_chars:  set[str] = set()
    for npz in (cache_root / "train").rglob("*.npz"):
        with np.load(npz, allow_pickle=False) as d:
            train_chars.update(str(d["label"].item()).lower())
    for s in ("val", "test"):
        for npz in (cache_root / s).rglob("*.npz"):
            with np.load(npz, allow_pickle=False) as d:
                eval_chars.update(str(d["label"].item()).lower())
    out_of_train = eval_chars - train_chars
    out_of_vocab = {c for c in eval_chars if c not in conv.dict}
    ok = not out_of_vocab
    return ok, (f"|train_chars|={len(train_chars)}  |eval_chars|={len(eval_chars)}  "
                f"out_of_train={sorted(out_of_train)}  "
                f"out_of_vocab={sorted(out_of_vocab)}")


def check_6_t_out_feasibility(cache_root: Path, T_native: int = 32,
                              upsample: int = 2):
    """Check 6: T_out = upsample * T_native >= 2 * max_label_len + 1."""
    from wita_v2.datasets.vocab import make_converter
    conv = make_converter("english")
    max_L = 0
    longest = ""
    for s in ("train", "val", "test"):
        for npz in (cache_root / s).rglob("*.npz"):
            with np.load(npz, allow_pickle=False) as d:
                lab = str(d["label"].item())
            enc, _ = conv.encode(lab)
            if enc.shape[0] > max_L:
                max_L = enc.shape[0]; longest = lab
    T_out = T_native * upsample
    bound = 2 * max_L + 1
    ok = T_out >= bound
    return ok, (f"max encoded label length = {max_L}  ('{longest}'),  "
                f"T_out = {T_out},  2L+1 = {bound}")


def check_7_feature_shape_and_dtype(cache_root: Path):
    """Check 7: every loaded feature has shape [32, 190] and is finite."""
    bad = 0; total = 0
    for npz in cache_root.rglob("*.npz"):
        total += 1
        with np.load(npz, allow_pickle=False) as d:
            f = d["feature"]
            if f.shape != (32, 190) or not np.all(np.isfinite(f.astype(np.float32))):
                bad += 1
                if bad <= 3:
                    print(f"    bad: {npz}  shape={f.shape}  "
                          f"any_nan={np.any(~np.isfinite(f))}")
        if total >= 2000 and bad == 0:
            # Spot-check 2000 then bail to avoid full-disk walk.
            break
    ok = bad == 0
    return ok, f"checked {total} files, {bad} bad"


def check_8_one_batch_dry_run(cache_root: Path):
    """
    Check 8: build a single mini-batch (BS=4, 16 train samples) and run a
    handful of training steps; assert both CTC and attention losses move.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from wita_v2.datasets.landmark_paper_split import WiTAPaperSplitDataset
    from wita_v2.configs.default     import Config, DataConfig, EncoderConfig, TrainConfig
    from wita_v2.models.conformer_ctc import ConformerCTC
    from wita_v2.models.attention_decoder import AttentionDecoder, build_attention_targets
    from wita_v2.training.stage11_train import _collate
    import torch.nn as nn

    cfg = Config(
        data=DataConfig(hf_repo_id="yewon816/WiTA", lang="english",
                        max_zips=None, max_frames=64, seed=42),
        encoder=EncoderConfig(arch="siglip"),
        train=TrainConfig(num_epochs=1, batch_size=4, seed=42,
                          checkpoint_dir="/tmp/_unused"),
    ).build()
    device = cfg.device

    ds = WiTAPaperSplitDataset(cache_root, "train", subsets=("lex", "nonlex"))
    ds = Subset(ds, list(range(min(16, len(ds)))))
    coll = lambda b: _collate(b, pad_idx=cfg.vocab.pad_idx)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=coll)

    enc = ConformerCTC(input_dim=190, vocab_size=cfg.vocab.ctc_vocab_size,
                       d_model=256, n_layers=4, n_heads=4, conv_kernel=15,
                       dropout=0.2, upsample=2, input_layernorm=False).to(device)
    dec = AttentionDecoder(att_vocab_size=cfg.vocab.attn_vocab_size,
                           bos_idx=cfg.vocab.sos_idx, eos_idx=cfg.vocab.eos_idx,
                           d_model=256, n_layers=2, n_heads=4,
                           ff_mult=4, dropout=0.2).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=5e-4)
    ctc_loss = nn.CTCLoss(blank=cfg.vocab.blank_idx, zero_infinity=True)
    ce_loss  = nn.CrossEntropyLoss(ignore_index=cfg.vocab.pad_idx,
                                   label_smoothing=0.1)

    history_ctc, history_attn = [], []
    enc.train(); dec.train()
    steps = 0
    while steps < 12:
        for feats, labels, in_lens, lab_lens, _, _, _ in loader:
            feats   = feats.to(device);   labels   = labels.to(device)
            in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)
            h, pad_mask = enc.encode(feats, in_lens)
            log_probs, enc_lens = enc.decode_ctc(h, in_lens)
            cl = ctc_loss(log_probs.transpose(0, 1).float(),
                          labels, enc_lens, lab_lens)
            dec_in, dec_tg = build_attention_targets(
                labels, lab_lens, bos=cfg.vocab.sos_idx, eos=cfg.vocab.eos_idx,
                pad=cfg.vocab.pad_idx,
            )
            dec_logits = dec(h, pad_mask, dec_in)
            al = ce_loss(dec_logits.reshape(-1, dec.att_vocab_size), dec_tg.reshape(-1))
            total = 0.5 * cl + 0.5 * al
            opt.zero_grad(); total.backward(); opt.step()
            history_ctc.append(float(cl.item()))
            history_attn.append(float(al.item()))
            steps += 1
            if steps >= 12: break

    ok_ctc  = history_ctc[-1]  < history_ctc[0]
    ok_attn = history_attn[-1] < history_attn[0]
    msg = (f"ctc:  {history_ctc[0]:.3f} -> {history_ctc[-1]:.3f}  "
           f"({'↓' if ok_ctc  else 'NOT ↓'})    "
           f"attn: {history_attn[0]:.3f} -> {history_attn[-1]:.3f}  "
           f"({'↓' if ok_attn else 'NOT ↓'})")
    return (ok_ctc and ok_attn), msg


# ---------------------------------------------------------------------------

CHECKS = [
    ("1. feature distribution",      check_1_feature_distribution),
    ("2. sample counts",             check_2_sample_counts),
    ("3. signer-disjoint splits",    check_3_signer_disjoint),
    ("4. no train/test clip leakage", check_4_no_clip_leakage),
    ("5. vocabulary coverage",       check_5_vocab_coverage),
    ("6. T_out feasibility",         check_6_t_out_feasibility),
    ("7. feature shape + finite",    check_7_feature_shape_and_dtype),
    ("8. one-batch dry run",         check_8_one_batch_dry_run),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--old-cache",  default=None,
        help="Path to the 38-signer .pt skeleton cache (for check 1).")
    args = p.parse_args(argv)
    cache_root = Path(args.cache_root)
    random.seed(42)

    all_ok = True
    for label, fn in CHECKS:
        try:
            if "1" in label[:2]:
                ok, msg = fn(cache_root, args.old_cache)
            else:
                ok, msg = fn(cache_root)
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}]  {label:<35}  {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All 8 checks passed.  Safe to launch Stage 11 training.")
        return 0
    print("One or more checks FAILED.  Do NOT launch the headline run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# coding: utf-8
"""
wita-final.py

Standalone script version of wita-final.ipynb.
All Jupyter magics (%%capture, !pip, !git, !rm) have been converted to
plain Python (subprocess calls) so this runs with `python wita-final.py`
instead of requiring a notebook/IPython kernel.
"""

# ## Cell 1 — clone + deps
# ---------------------------------------------------------------------------
import subprocess
import sys
import os
import glob
import torch
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\50128313\gaurang-personal\WiTA-v2")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")


# ## Cell 2 — toggleable normalisation + both caches
# ---------------------------------------------------------------------------
import datasets.landmark_paper_split as lps
from normalize_cache import normalize_feature

CACHE_T32 = str(REPO_ROOT / "landmark_cache_122")
CACHE_T64 = str(REPO_ROOT / "landmark_cache_122_t64")
assert os.path.isdir(os.path.join(CACHE_T32, "train")), f"fix T32 cache: {CACHE_T32}"
assert os.path.isdir(os.path.join(CACHE_T64, "train")), f"fix T64 cache: {CACHE_T64}"

# Normalisation is applied on-the-fly ONLY when the flag is True -> lets us toggle factor N.
if not getattr(lps.WiTAPaperSplitDataset, "_fac_patched", False):
    _orig = lps.WiTAPaperSplitDataset.__getitem__

    def _getitem(self, i):
        out = _orig(self, i)
        x = out[0]
        if getattr(lps.WiTAPaperSplitDataset, "_normalize_on", False):
            x = torch.from_numpy(normalize_feature(x.detach().cpu().numpy())).to(x.dtype)
            out = (x,) + tuple(out[1:])
        if not getattr(lps.WiTAPaperSplitDataset, "_printed_shape", True):  # verify T once per cell
            print(
                f"   [verify] feature shape = {tuple(x.shape)} | normalize="
                f"{getattr(lps.WiTAPaperSplitDataset, '_normalize_on', False)}"
            )
            lps.WiTAPaperSplitDataset._printed_shape = True
        return out

    lps.WiTAPaperSplitDataset.__getitem__ = _getitem
    lps.WiTAPaperSplitDataset._fac_patched = True

print("factorial harness ready: N toggled by _normalize_on, T toggled by cache_root")


# ## Cell 3 — config (identical to Stage 11; only LOG/CKPT dirs differ)
# ---------------------------------------------------------------------------
import logging
import random
import numpy as np
from default import Config, DataConfig, EncoderConfig, TrainConfig

VARIANT_NAME = "stage18_norm"
LOG_DIR = str(REPO_ROOT / f"logs_{VARIANT_NAME}")
CKPT_DIR = str(REPO_ROOT / f"checkpoints_{VARIANT_NAME}")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "log.log"))],
)

SEED = 42
NUM_EPOCHS = 80
BATCH_SIZE = 32
LAMBDA_CTC = 0.5
DEC_N_LAYERS = 2
DEC_N_HEADS = 4
LR_PEAK = 5e-4
WEIGHT_DECAY = 5e-2
GRAD_CLIP = 1.0
DROPOUT = 0.2
WARMUP_PCT = 0.05
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
CONV_KERNEL = 15
UPSAMPLE = 2

cfg = Config(
    data=DataConfig(hf_repo_id="yewon816/WiTA", lang="english", max_zips=None, max_frames=64, seed=SEED),
    encoder=EncoderConfig(arch="siglip"),
    train=TrainConfig(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR_PEAK,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        num_workers=2,
        warmup_pct=WARMUP_PCT,
        seed=SEED,
        checkpoint_dir=CKPT_DIR,
    ),
).build()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

print("variant", VARIANT_NAME, "| device", cfg.device)


# ## Cell 4 — train (normalized features; baseline augmentation, P_AFFINE=0)
# ---------------------------------------------------------------------------
import json
from stage11_train import train_stage11
from skeleton_augment import LandmarkAugment

LOG_DIR = str(REPO_ROOT / "logs_factorial")
CKPT_DIR = str(REPO_ROOT / "ckpt_factorial")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

train_aug = LandmarkAugment()
EPOCHS, SEED = 60, 42  # one fixed protocol for all four cells

FACTORIAL = [
    #{"name": "C1_raw_T32", "cache": CACHE_T32, "normalize": False, "T": 32},
    #{"name": "C2_norm_T32", "cache": CACHE_T32, "normalize": True, "T": 32},
    {"name": "C3_raw_T64", "cache": CACHE_T64, "normalize": False, "T": 64},
    #{"name": "C4_norm_T64", "cache": CACHE_T64, "normalize": True, "T": 64},
]  # to save ~40 min you MAY drop C1/C4 and reuse existing numbers, but all-4 is the clean version

res_all = []
for f in FACTORIAL:
    print("\n" + "=" * 72 + f"\nFACTORIAL {f['name']}  (normalize={f['normalize']}, T={f['T']})\n" + "=" * 72)
    lps.WiTAPaperSplitDataset._normalize_on = f["normalize"]
    lps.WiTAPaperSplitDataset._printed_shape = False  # re-arm the per-cell shape check
    r = train_stage11(
        cache_root=f["cache"],
        cfg=cfg,
        num_epochs=EPOCHS,
        batch_size=32,
        lr_peak=5e-4,
        weight_decay=5e-2,
        grad_clip=1.0,
        dropout=0.2,
        d_model=256,
        n_layers=4,
        n_heads=4,
        conv_kernel=15,
        upsample=2,
        warmup_pct=0.05,
        dec_n_layers=2,
        dec_n_heads=4,
        lambda_ctc=0.5,
        label_smoothing=0.1,
        transform=train_aug,
        seed=SEED,
        checkpoint_dir=CKPT_DIR,
        log_dir=LOG_DIR,
        variant=f["name"],
    )
    bp = r.get("best_payload", {}) or {}
    res_all.append(
        {
            "name": f["name"],
            "T": f["T"],
            "normalize": f["normalize"],
            "val_overall": r["best_val_overall"],
            "best_epoch": r["best_epoch"],
            "val_lex": bp.get("val_lex_cer"),
            "val_nonlex": bp.get("val_nonlex_cer"),
        }
    )

d = {x["name"]: x["val_overall"] for x in res_all}
C1, C2, C3, C4 = d["C1_raw_T32"], d["C2_norm_T32"], d["C3_raw_T64"], d["C4_norm_T64"]

print("\n" + "=" * 48 + "\nFACTORIAL  (validation overall CER)\n" + "=" * 48)
print(f"{'':10}{'T=32':>10}{'T=64':>10}")
print(f"{'raw':10}{C1:10.4f}{C3:10.4f}")
print(f"{'norm':10}{C2:10.4f}{C4:10.4f}")
print(f"\nmain effect  T (T64-T32) : {((C3 + C4) - (C1 + C2)) / 2:+.4f}")
print(f"main effect  N (norm-raw): {((C2 + C4) - (C1 + C3)) / 2:+.4f}")
print(f"interaction  N x T       : {(C4 - C3) - (C2 - C1):+.4f}")

json.dump(res_all, open(os.path.join(LOG_DIR, "factorial.json"), "w"), indent=2)
print("\nsaved", os.path.join(LOG_DIR, "factorial.json"))
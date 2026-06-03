"""
training/stage12_train.py — Stage 12 paper-comparable run with VideoMAE+LoRA.

Architecture (frozen by Stage 9a + the Stage 12 prompt):
    VideoMAE-base + LoRA (r=16, q/k/v)               -> tube features
    Spatial mean over patches -> temporal upsample    -> [B, T_native=32, 768]
    Landmark stream (190d) -> proj 128                 -> [B, 32, 128]
    Concat -> fusion proj 256                          -> [B, 32, 256]
    Conformer (4 layers, d=256, h=4, k=15, drop 0.2)   -> [B, 32, 256]
    ConvTranspose1d upsample x2                        -> [B, 64, 256]
    CTC head + Attention decoder (2 layers, h=4, d=256, smoothing 0.1)
    lambda_ctc = 0.5

Test-set discipline: this trainer NEVER touches test_loader during
training.  `final_test_eval()` is called exactly once by the
orchestrator notebook, gated by a marker file in /kaggle/working/logs/.
"""

from __future__ import annotations

import os
import json
import time
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import editdistance

from ..models.stage12_model     import Stage12Model
from ..models.attention_decoder import build_attention_targets
from ..datasets.vocab           import make_converter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dual-cache dataset: pairs handcrop video with the existing landmark cache
# ---------------------------------------------------------------------------

_VALID_SPLITS  = {"train", "val", "test"}
_VALID_SUBSETS = {"lex", "nonlex"}


class WiTAPaperSplitDualDataset(Dataset):
    """
    Pairs handcrop video [T=16, 224, 224, 3] uint8 with landmark features
    [T_native, 190] fp16 -> fp32, joined on '<SIGNER>__<clip_id>'.

    Parameters
    ----------
    handcrop_root  : root produced by datasets.handcrop_cache
    landmark_root  : root produced by datasets.landmark_cache_122
    split          : 'train' | 'val' | 'test'
    subsets        : ('lex',), ('nonlex',), or both
    T_native       : pad/truncate landmark T axis to this many frames (default 32)

    Returns: (video_uint8[T=16,224,224,3], landmark[T_native,190], label_ids,
              signer, subset, label_str)
    """

    def __init__(
        self,
        handcrop_root:  str | Path,
        landmark_root:  str | Path,
        split:          str,
        subsets:        Sequence[str] = ("lex", "nonlex"),
        *,
        converter=None,
        T_native:       int = 32,
        lang:           str = "english",
    ):
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS}, got {split!r}")
        for s in subsets:
            if s not in _VALID_SUBSETS:
                raise ValueError(f"subset must be in {_VALID_SUBSETS}, got {s!r}")
        self.handcrop_root = Path(handcrop_root)
        self.landmark_root = Path(landmark_root)
        self.split    = split
        self.subsets  = tuple(subsets)
        self.T_native = int(T_native)

        if converter is None:
            from ..datasets.vocab import make_converter
            converter = make_converter(lang)
        self.converter = converter

        # Inner-join the two caches by stem '<SIGNER>__<clip_id>'.
        # Skip clips present in only one cache, report counts + first-N
        # examples per subset, continue training on the join.  Use print()
        # rather than logger.warning so Kaggle stdout always shows it.
        self.entries: list[dict] = []
        miss_per_subset: dict[str, dict] = {}
        SHOW_FIRST_N = 10
        for subset in self.subsets:
            v_dir = self.handcrop_root / split / subset
            l_dir = self.landmark_root / split / subset
            if not v_dir.exists():
                raise FileNotFoundError(f"missing handcrop dir: {v_dir}")
            if not l_dir.exists():
                raise FileNotFoundError(f"missing landmark dir: {l_dir}")

            v_stems = {p.stem: p for p in sorted(v_dir.glob("*.npz"))}
            l_stems = {p.stem: p for p in sorted(l_dir.glob("*.npz"))}
            miss_v: list[str] = []
            miss_l: list[str] = []

            for stem in sorted(set(v_stems) | set(l_stems)):
                if stem not in v_stems:
                    miss_v.append(stem); continue
                if stem not in l_stems:
                    miss_l.append(stem); continue
                if "__" not in stem:
                    print(f"  [Stage12 dual {split}/{subset}] unexpected stem "
                          f"{stem!r}; skipping.", flush=True)
                    continue
                signer_id, clip_id = stem.split("__", 1)
                self.entries.append({
                    "video_path":    str(v_stems[stem]),
                    "landmark_path": str(l_stems[stem]),
                    "signer":        signer_id,
                    "clip_id":       clip_id,
                    "subset":        subset,
                })
            miss_per_subset[subset] = {"video": miss_v, "landmark": miss_l}

        if not self.entries:
            raise RuntimeError(
                f"No paired clips found under {handcrop_root} ∩ {landmark_root}"
                f" for split={split} subsets={self.subsets}"
            )

        # Per-subset cache-mismatch report.  Distinguishes "whole signer
        # dropped" (= regex / extractor bug) from "scattered per-clip
        # failures" (= benign / MediaPipe).
        total_miss_v = sum(len(d["video"])    for d in miss_per_subset.values())
        total_miss_l = sum(len(d["landmark"]) for d in miss_per_subset.values())
        if total_miss_v or total_miss_l:
            print(f"\n[Stage12 dual {split}] cache-mismatch report:", flush=True)
            print(f"  missing in handcrop cache (landmark-only): {total_miss_v}",
                  flush=True)
            print(f"  missing in landmark cache (handcrop-only): {total_miss_l}",
                  flush=True)
            for subset, d in miss_per_subset.items():
                mv, ml = d["video"], d["landmark"]
                if not (mv or ml): continue
                print(f"  --- {split}/{subset} ---", flush=True)
                if mv:
                    print(f"    {len(mv)} missing in handcrop cache; first "
                          f"{min(SHOW_FIRST_N, len(mv))}:", flush=True)
                    for s in mv[:SHOW_FIRST_N]:
                        print(f"      {s}", flush=True)
                if ml:
                    print(f"    {len(ml)} missing in landmark cache; first "
                          f"{min(SHOW_FIRST_N, len(ml))}:", flush=True)
                    for s in ml[:SHOW_FIRST_N]:
                        print(f"      {s}", flush=True)
            # Heuristic: if all missing stems share a signer prefix, it's a
            # whole-signer drop -- almost certainly a bug worth fixing
            # before training rather than tolerating.
            for subset, d in miss_per_subset.items():
                for side, stems in d.items():
                    if not stems: continue
                    signers = {s.split("__", 1)[0] for s in stems if "__" in s}
                    if len(signers) <= 3 and len(stems) >= 20:
                        print(f"\n  HEURISTIC WARNING: all {len(stems)} "
                              f"{side}-side misses in {split}/{subset} come "
                              f"from {len(signers)} signer(s): "
                              f"{sorted(signers)}.  This looks like a "
                              f"whole-signer drop -- verify the corresponding "
                              f"cache wasn't extracted with the old regex.",
                              flush=True)
        logger.info(
            "[Stage12 dual %s/%s] %d clips across %d signers",
            split, "+".join(self.subsets),
            len(self.entries), len({e["signer"] for e in self.entries}),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_landmark(self, path: str) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as d:
            feat = d["feature"].astype(np.float32)   # [T_in, 190]
        x = torch.from_numpy(feat)                   # [T_in, 190]
        T_in, D = x.shape
        if T_in == self.T_native:
            return x
        # Linear-interpolate landmark T to T_native so the fusion concat lines up.
        x = x.transpose(0, 1).unsqueeze(0)           # [1, 190, T_in]
        x = F.interpolate(x, size=self.T_native, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1)          # [T_native, 190]

    def __getitem__(self, i):
        e = self.entries[i]
        with np.load(e["video_path"], allow_pickle=False) as d:
            video = d["video"]                       # [T=16, 224, 224, 3] uint8
            label = str(d["label"].item())
        # [T, H, W, 3] -> [T, 3, H, W] uint8 (normalisation lives in the backbone).
        v = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
        l = self._load_landmark(e["landmark_path"])
        enc, _ = self.converter.encode(label)
        return v, l, enc, e["signer"], e["subset"], label

    @property
    def signers(self) -> list[str]:
        return sorted({e["signer"] for e in self.entries})

    def per_subset_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e["subset"]] = out.get(e["subset"], 0) + 1
        return out


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_dual(batch, pad_idx: int):
    videos, landmarks, labels, signers, subsets, label_strs = zip(*batch)
    videos    = torch.stack(videos, dim=0)                              # [B, 16, 3, 224, 224] uint8
    landmarks = torch.stack(landmarks, dim=0)                           # [B, T_native, 190]
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=pad_idx)
    input_lens = torch.full((videos.size(0),), landmarks.size(1), dtype=torch.long)
    label_lens = torch.LongTensor([l.shape[0] for l in labels])
    return (videos, landmarks, labels_pad, input_lens, label_lens,
            list(signers), list(subsets), list(label_strs))


# ---------------------------------------------------------------------------
# Decode helpers + eval aggregation (same shape as Stage 11)
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 999)]


def _length_bucket(L: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= L <= hi:
            return f"{lo}-{hi if hi < 999 else 'inf'}"
    return "unknown"


def _ctc_greedy(log_probs: torch.Tensor, enc_lens: torch.Tensor, blank: int):
    out = []
    argmax = log_probs.argmax(dim=-1)
    for b in range(argmax.shape[0]):
        seq = argmax[b, : int(enc_lens[b].item())].tolist()
        merged = []; prev = None
        for t in seq:
            if t != prev and t != blank:
                merged.append(t)
            prev = t
        out.append(merged)
    return out


def _ids_to_str(ids, chars):
    return "".join(chars[t-1] if 1 <= t <= len(chars) else "?" for t in ids)


@torch.no_grad()
def evaluate_loader(
    loader,
    *,
    model:  Stage12Model,
    cfg,
    blank:  int,
    sos:    int,
    eos:    int,
    pad:    int,
    device: str,
    autocast_dtype: Optional[torch.dtype] = None,
):
    model.eval()
    pairs_all: list[dict] = []
    sum_val_ctc = 0.0; n_val_batches = 0
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    use_amp = autocast_dtype is not None and torch.cuda.is_available()
    for videos, landmarks, labels, in_lens, lab_lens, signers, subsets, label_strs in loader:
        videos    = videos.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        labels    = labels.to(device);    in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

        ctx = torch.cuda.amp.autocast(dtype=autocast_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            h, pad_mask = model.encode(videos, landmarks, in_lens)
            log_probs, enc_lens = model.encoder.decode_ctc(h, in_lens)
        v_ctc = ctc_loss_fn(log_probs.transpose(0, 1).float(),
                            labels, enc_lens, lab_lens)
        sum_val_ctc += float(v_ctc.item()); n_val_batches += 1

        ctc_preds  = _ctc_greedy(log_probs, enc_lens, blank)
        attn_preds = model.decoder.greedy_decode(h, pad_mask)
        for b in range(len(ctc_preds)):
            gt     = label_strs[b]
            p_ctc  = _ids_to_str(ctc_preds[b],           cfg.vocab.chars)
            p_attn = _ids_to_str(attn_preds[b].tolist(), cfg.vocab.chars)
            e_ctc  = editdistance.eval(gt, p_ctc)
            e_attn = editdistance.eval(gt, p_attn)
            e_best = min(e_ctc, e_attn)
            pred_best = p_ctc if e_ctc <= e_attn else p_attn
            pairs_all.append({
                "gt":     gt,
                "pred":   pred_best,
                "edit":   int(e_best),
                "L":      int(len(gt)),
                "signer": signers[b],
                "subset": subsets[b],
                "ctc":    p_ctc,
                "attn":   p_attn,
                "e_ctc":  int(e_ctc),
                "e_attn": int(e_attn),
            })

    val_ctc = sum_val_ctc / max(n_val_batches, 1)

    def _cer(rows):
        n = sum(r["edit"] for r in rows)
        d = sum(r["L"]    for r in rows)
        return n / max(d, 1)

    overall_cer = _cer(pairs_all)
    per_subset: dict[str, float] = {}
    per_signer: dict[str, float] = {}
    per_bucket: dict[str, float] = {}
    by_subset: dict[str, list]  = defaultdict(list)
    by_signer: dict[str, list]  = defaultdict(list)
    by_bucket: dict[str, list]  = defaultdict(list)
    for r in pairs_all:
        by_subset[r["subset"]].append(r)
        by_signer[r["signer"]].append(r)
        by_bucket[_length_bucket(r["L"])].append(r)
    for k, v in by_subset.items(): per_subset[k] = _cer(v)
    for k, v in by_signer.items(): per_signer[k] = _cer(v)
    for k, v in by_bucket.items(): per_bucket[k] = _cer(v)

    return {
        "val_ctc_loss":   val_ctc,
        "overall_cer":    overall_cer,
        "per_subset_cer": per_subset,
        "per_signer_cer": per_signer,
        "per_length_cer": per_bucket,
        "n_clips":        len(pairs_all),
        "pairs":          pairs_all,
    }


# ---------------------------------------------------------------------------
# Param grouping (two LR groups: backbone vs everything else)
# ---------------------------------------------------------------------------

def _split_param_groups(model: Stage12Model, lr_backbone: float, lr_head: float, weight_decay: float):
    backbone_params, head_params = [], []
    backbone_names: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith("backbone."):
            backbone_params.append(p); backbone_names.append(name)
        else:
            head_params.append(p)
    print(f"[stage12] backbone trainable params: {sum(p.numel() for p in backbone_params):,}",
          flush=True)
    print(f"[stage12] head     trainable params: {sum(p.numel() for p in head_params):,}",
          flush=True)
    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay,
         "name": "backbone"},
        {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay,
         "name": "head"},
    ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_stage12(
    handcrop_root:  str,
    landmark_root:  str,
    *,
    cfg,
    num_epochs:     int = 50,
    batch_size:     int = 16,
    lr_backbone:    float = 5e-5,
    lr_head:        float = 5e-4,
    weight_decay:   float = 5e-2,
    grad_clip:      float = 1.0,
    warmup_pct:     float = 0.05,
    lambda_ctc:     float = 0.5,
    label_smoothing: float = 0.1,
    # model knobs
    videomae_model_name: str = "MCG-NJU/videomae-base",
    lora_r:         int = 16,
    lora_alpha:     int = 32,
    lora_dropout:   float = 0.1,
    d_model:        int = 256,
    n_layers:       int = 4,
    n_heads:        int = 4,
    conv_kernel:    int = 15,
    upsample:       int = 2,
    dropout:        float = 0.2,
    dec_n_layers:   int = 2,
    dec_n_heads:    int = 4,
    T_native:       int = 32,
    # runtime
    amp_dtype:      Optional[str] = "fp16",     # "fp16", "bf16", or None
    gradient_checkpointing: bool = True,
    num_workers:    int = 2,
    seed:           int = 42,
    checkpoint_dir: str = "/kaggle/working/checkpoints",
    log_dir:        str = "/kaggle/working/logs",
    variant:        str = "stage12",
) -> dict:
    """Stage 12 training.  Saves best-val checkpoint; never touches test."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device     = cfg.device
    converter  = make_converter(cfg.data.lang)
    pad_idx    = cfg.vocab.pad_idx
    blank      = cfg.vocab.blank_idx
    att_V      = cfg.vocab.attn_vocab_size
    sos        = cfg.vocab.sos_idx
    eos        = cfg.vocab.eos_idx
    ctc_V      = cfg.vocab.ctc_vocab_size

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # AMP setup.
    autocast_dtype = None
    if amp_dtype == "fp16": autocast_dtype = torch.float16
    elif amp_dtype == "bf16": autocast_dtype = torch.bfloat16
    use_amp = autocast_dtype is not None and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and autocast_dtype == torch.float16))
    print(f"[stage12] AMP enabled={use_amp}  dtype={autocast_dtype}", flush=True)

    # ---- data ----
    train_ds = WiTAPaperSplitDualDataset(
        handcrop_root, landmark_root, "train",
        subsets=("lex", "nonlex"), converter=converter, T_native=T_native,
    )
    val_ds = WiTAPaperSplitDualDataset(
        handcrop_root, landmark_root, "val",
        subsets=("lex", "nonlex"), converter=converter, T_native=T_native,
    )
    coll = lambda b: _collate_dual(b, pad_idx=pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=coll, drop_last=True,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=coll,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    print(f"[stage12] train={len(train_ds)}  val={len(val_ds)}  "
          f"signers train/val = {len(train_ds.signers)}/{len(val_ds.signers)}",
          flush=True)
    print(f"[stage12] train subset counts: {train_ds.per_subset_counts()}",
          flush=True)
    print(f"[stage12] val   subset counts: {val_ds.per_subset_counts()}",
          flush=True)

    # ---- model ----
    model = Stage12Model(
        ctc_vocab_size  = ctc_V,
        attn_vocab_size = att_V,
        sos_idx         = sos,
        eos_idx         = eos,
        videomae_model_name = videomae_model_name,
        lora_r          = lora_r,
        lora_alpha      = lora_alpha,
        lora_dropout    = lora_dropout,
        d_model         = d_model,
        n_layers        = n_layers,
        n_heads         = n_heads,
        conv_kernel     = conv_kernel,
        dropout         = dropout,
        upsample        = upsample,
        T_native        = T_native,
        dec_n_layers    = dec_n_layers,
        dec_n_heads     = dec_n_heads,
        gradient_checkpointing = gradient_checkpointing,
    ).to(device)
    print(f"[stage12] total trainable params: {model.num_trainable:,}",
          flush=True)

    param_groups = _split_param_groups(model, lr_backbone, lr_head, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    total_steps = num_epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr_backbone, lr_head],
        total_steps=total_steps, pct_start=warmup_pct, anneal_strategy="cos",
    )
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    ce  = nn.CrossEntropyLoss(ignore_index=pad_idx,
                              label_smoothing=label_smoothing)

    history: list[dict] = []
    best_overall = float("inf")
    best_epoch   = -1
    best_payload: dict = {}
    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pt")

    print(
        f"\n=== Stage 12 training ===  variant={variant}  "
        f"lambda_ctc={lambda_ctc}  lr=[{lr_backbone:.0e}/{lr_head:.0e}]  "
        f"epochs={num_epochs}  batch={batch_size}  seed={seed}",
        flush=True,
    )

    for epoch in range(num_epochs):
        model.train()
        sum_ctc = sum_attn = sum_total = 0.0
        n_batches = 0
        t0 = time.time()

        for videos, landmarks, labels, in_lens, lab_lens, _, _, _ in train_loader:
            videos    = videos.to(device, non_blocking=True)
            landmarks = landmarks.to(device, non_blocking=True)
            labels    = labels.to(device);    in_lens = in_lens.to(device); lab_lens = lab_lens.to(device)

            dec_in, dec_tg = build_attention_targets(
                labels, lab_lens, bos=sos, eos=eos, pad=pad_idx,
            )

            optimizer.zero_grad(set_to_none=True)
            ctx = (torch.cuda.amp.autocast(dtype=autocast_dtype)
                   if use_amp else torch.cuda.amp.autocast(enabled=False))
            with ctx:
                log_probs, enc_lens, dec_logits, _, _ = model(
                    videos, landmarks, in_lens, dec_in,
                )
                ctc_loss = ctc(log_probs.transpose(0, 1).float(),
                               labels, enc_lens, lab_lens)
                attn_loss = ce(dec_logits.reshape(-1, model.decoder.att_vocab_size),
                               dec_tg.reshape(-1))
                total = lambda_ctc * ctc_loss + (1 - lambda_ctc) * attn_loss

            # Only step the LR scheduler if the optimizer actually stepped.
            # AMP can skip an optimizer step on the very first iteration
            # if it detects inf/nan during loss scaling; stepping the
            # scheduler anyway emits a "scheduler.step() before
            # optimizer.step()" warning and silently skips an LR value.
            optimizer_stepped = True
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]],
                                         grad_clip)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                # If GradScaler skipped the step (e.g. due to inf in grads),
                # the scale factor changes; compare before/after to detect.
                optimizer_stepped = scaler.get_scale() >= old_scale
            else:
                total.backward()
                nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]],
                                         grad_clip)
                optimizer.step()
            if optimizer_stepped:
                scheduler.step()

            sum_ctc   += float(ctc_loss.item())
            sum_attn  += float(attn_loss.item())
            sum_total += float(total.item())
            n_batches += 1

        train_ctc   = sum_ctc   / max(n_batches, 1)
        train_attn  = sum_attn  / max(n_batches, 1)
        train_total = sum_total / max(n_batches, 1)

        # ---- val evaluation ----
        val_out = evaluate_loader(
            val_loader, model=model, cfg=cfg,
            blank=blank, sos=sos, eos=eos, pad=pad_idx,
            device=device, autocast_dtype=autocast_dtype,
        )
        val_overall = val_out["overall_cer"]
        val_lex     = val_out["per_subset_cer"].get("lex",    float("nan"))
        val_non     = val_out["per_subset_cer"].get("nonlex", float("nan"))

        dt = time.time() - t0
        print(
            f"[E{epoch+1:3d}/{num_epochs}] "
            f"ctc={train_ctc:.4f}  attn={train_attn:.4f}  total={train_total:.4f}  "
            f"val_ctc={val_out['val_ctc_loss']:.4f}  "
            f"val CER overall={val_overall:.4f}  lex={val_lex:.4f}  "
            f"nonlex={val_non:.4f}  {dt:.0f}s",
            flush=True,
        )
        snap = {k: v for k, v in val_out.items() if k != "pairs"}
        history.append({
            "epoch":           epoch + 1,
            "train_ctc":       train_ctc,
            "train_attn":      train_attn,
            "train_total":     train_total,
            **snap,
        })

        if val_overall < best_overall:
            best_overall = val_overall
            best_epoch   = epoch + 1
            best_payload = {
                "val_overall_cer":  val_overall,
                "val_lex_cer":      val_lex,
                "val_nonlex_cer":   val_non,
                "val_per_signer":   dict(val_out["per_signer_cer"]),
                "val_per_length":   dict(val_out["per_length_cer"]),
                "epoch":            epoch + 1,
            }
            torch.save({
                "variant":          variant,
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "best_payload":     best_payload,
                "lambda_ctc":       lambda_ctc,
                "dec_n_layers":     dec_n_layers,
                "lora_r":           lora_r,
                "lora_alpha":       lora_alpha,
                "lora_dropout":     lora_dropout,
                "T_native":         T_native,
                "seed":             seed,
                "videomae_model_name": videomae_model_name,
            }, ckpt_path)
            print(f"    * new best val_overall_cer={val_overall:.4f}", flush=True)

    result = {
        "variant":          variant,
        "num_epochs":       num_epochs,
        "n_train_clips":    len(train_ds),
        "n_val_clips":      len(val_ds),
        "seed":             seed,
        "lambda_ctc":       lambda_ctc,
        "dec_n_layers":     dec_n_layers,
        "best_val_overall": best_overall,
        "best_epoch":       best_epoch,
        "best_payload":     best_payload,
        "checkpoint_path":  ckpt_path,
        "history":          history,
    }
    json_path = os.path.join(log_dir, f"{variant}_training.json")
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, indent=2, sort_keys=True)
    with open(json_path.replace(".json", "_full.json"), "w") as f:
        json.dump(result["history"], f, indent=2, sort_keys=True)
    return result


# ---------------------------------------------------------------------------
# Test-set evaluation — called EXACTLY ONCE by the orchestrator
# ---------------------------------------------------------------------------

def final_test_eval(
    handcrop_root: str,
    landmark_root: str,
    checkpoint:    str,
    *,
    cfg,
    batch_size:    int = 16,
    T_native:      int = 32,
    amp_dtype:     Optional[str] = "fp16",
    num_workers:   int = 2,
    log_dir:       str = "/kaggle/working/logs",
    variant:       str = "stage12",
    # passthrough model knobs (must match training run)
    videomae_model_name: str = "MCG-NJU/videomae-base",
    lora_r:        int = 16,
    lora_alpha:    int = 32,
    lora_dropout:  float = 0.1,
    d_model:       int = 256,
    n_layers:      int = 4,
    n_heads:       int = 4,
    conv_kernel:   int = 15,
    dropout:       float = 0.2,
    upsample:      int = 2,
    dec_n_layers:  int = 2,
    dec_n_heads:   int = 4,
) -> dict:
    """Load best-val checkpoint, evaluate ONCE on test, write headline JSON."""
    marker = os.path.join(log_dir, f".{variant}_test_evaluated")
    if os.path.exists(marker):
        raise RuntimeError(
            f"Test evaluation already completed.  Marker: {marker}.  "
            "Per the Stage 12 prompt the test set must be evaluated exactly "
            "once.  Delete the marker manually if you intentionally re-run."
        )

    device     = cfg.device
    converter  = make_converter(cfg.data.lang)
    pad_idx    = cfg.vocab.pad_idx
    blank      = cfg.vocab.blank_idx
    att_V      = cfg.vocab.attn_vocab_size
    sos        = cfg.vocab.sos_idx
    eos        = cfg.vocab.eos_idx
    ctc_V      = cfg.vocab.ctc_vocab_size

    autocast_dtype = None
    if amp_dtype == "fp16": autocast_dtype = torch.float16
    elif amp_dtype == "bf16": autocast_dtype = torch.bfloat16

    test_ds = WiTAPaperSplitDualDataset(
        handcrop_root, landmark_root, "test",
        subsets=("lex", "nonlex"), converter=converter, T_native=T_native,
    )
    coll = lambda b: _collate_dual(b, pad_idx=pad_idx)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=coll,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    model = Stage12Model(
        ctc_vocab_size  = ctc_V, attn_vocab_size = att_V,
        sos_idx         = sos, eos_idx = eos,
        videomae_model_name = videomae_model_name,
        lora_r          = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout,
        d_model         = d_model, n_layers = n_layers, n_heads = n_heads,
        conv_kernel     = conv_kernel, dropout = dropout,
        upsample        = upsample, T_native = T_native,
        dec_n_layers    = dec_n_layers, dec_n_heads = dec_n_heads,
        gradient_checkpointing = False,
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[stage12] load_state_dict missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    print(f"[stage12] loaded {checkpoint}  best val_overall="
          f"{state.get('best_payload', {}).get('val_overall_cer', '?')}",
          flush=True)

    test_out = evaluate_loader(
        test_loader, model=model, cfg=cfg,
        blank=blank, sos=sos, eos=eos, pad=pad_idx,
        device=device, autocast_dtype=autocast_dtype,
    )

    test_overall = test_out["overall_cer"]
    test_lex     = test_out["per_subset_cer"].get("lex",    float("nan"))
    test_non     = test_out["per_subset_cer"].get("nonlex", float("nan"))

    # WRITE THE HEADLINE FIRST.  Marker file + JSON committed before
    # any prose / diagnostic analysis happens.
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, f"{variant}_test_headline.json")
    headline = {
        "variant":          variant,
        "test_overall_cer": float(test_overall),
        "test_lex_cer":     float(test_lex),
        "test_nonlex_cer":  float(test_non),
        "n_test_clips":     test_out["n_clips"],
        "paper_baseline": {
            "overall": 0.2924,
            "lex":     0.281,
            "nonlex":  0.365,
        },
        "stage11_baseline": {
            "overall": 0.4498,
            "lex":     0.4348,
        },
        "checkpoint":       checkpoint,
    }
    with open(summary_path, "w") as f:
        json.dump(headline, f, indent=2, sort_keys=True)
    with open(marker, "w") as f:
        f.write(summary_path)

    print("\n" + "=" * 64)
    print(f"  STAGE 12 TEST-SET HEADLINE (written to {summary_path})")
    print("=" * 64)
    print(f"  test_overall_cer : {test_overall:.4f}   (paper: 0.2924, S11: 0.4498)")
    print(f"  test_lex_cer     : {test_lex:.4f}       (paper: 0.281,  S11: 0.4348)")
    print(f"  test_nonlex_cer  : {test_non:.4f}       (paper: 0.365)")
    print("=" * 64 + "\n")

    diag_path = os.path.join(log_dir, f"{variant}_test_full.json")
    with open(diag_path, "w") as f:
        json.dump({
            "headline":        headline,
            "per_signer_cer":  test_out["per_signer_cer"],
            "per_length_cer":  test_out["per_length_cer"],
            "n_clips_per_subset": {
                k: sum(1 for r in test_out["pairs"] if r["subset"] == k)
                for k in ("lex", "nonlex")
            },
        }, f, indent=2, sort_keys=True)
    return {
        "headline":       headline,
        "per_signer_cer": test_out["per_signer_cer"],
        "per_length_cer": test_out["per_length_cer"],
        "pairs":          test_out["pairs"],
    }

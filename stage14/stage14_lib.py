"""
stage14_lib.py — Stage 14: VideoMAE-base + gradual unfreezing + CTC,
on FULL FRAMES (no hand crop, no MediaPipe), to compare the encoder
against the paper's R3D (Stage 13B) on the paper's input paradigm.

Design vs Stage 12 (which used LoRA + MediaPipe hand crops + landmark
fusion and plateaued at test 0.59):
  * Full frames resized to 224x224 (no crop)  -> matches the paper's
    "full frame" input; in-distribution for VideoMAE's Kinetics
    pretraining; no MediaPipe dependency / detection-quality issues.
  * Gradual unfreezing (frozen -> last4 -> last8 -> full)  -> full
    adaptation capacity, vs LoRA r=16's ~1.5M params.
  * CTC only (no attention decoder, no landmarks)  -> matches the
    paper's decoder so the ENCODER is the only variable vs R3D.

Reads the Kaggle-mounted WiTA JPG frames directly (on-the-fly decode);
VideoMAE-base's heavy forward on a T4 hides the decode cost, so it
stays GPU-bound without a pre-decode cache.

Vocab (CTC): blank=0, a..z = 1..26, '-' (repeat separator) = 27 -> 28.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import editdistance
except ImportError:
    editdistance = None


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOCAB = 28          # blank=0, a..z=1..26, '-'=27
T_IN = 16           # VideoMAE-base num_frames
IMG_SIZE = 224
T_OUT = 24          # 8 tubes upsampled x3


# ---------------------------------------------------------------------------
# Char <-> id
# ---------------------------------------------------------------------------

class CharConverter:
    """CTC char converter matching the paper's StrLabelConverter:
    inserts the '-' separator between repeated adjacent characters on
    encode; collapses CTC blanks + repeats and drops '-' on decode."""

    def __init__(self, alphabet: str = ALPHABET):
        self.alphabet = alphabet + "-"                 # 27 chars, indices 1..27
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.alphabet)}

    def encode(self, text: str) -> torch.LongTensor:
        text = text.lower()
        ids = []
        prev = None
        for c in text:
            if c == prev:
                ids.append(self.char_to_idx["-"])      # CTC repeat separator
            if c in self.char_to_idx:
                ids.append(self.char_to_idx[c])
                prev = c
            else:
                prev = None                            # unknown char breaks the run
        return torch.LongTensor(ids)

    def decode_ctc(self, ids) -> str:
        out = []
        prev = -1
        for i in ids:
            i = int(i)
            if i != 0 and i != prev and 0 < i <= len(self.alphabet):
                out.append(self.alphabet[i - 1])
            prev = i
        return "".join(out).replace("-", "")

    def gt_string(self, label: str) -> str:
        """The ground-truth string for CER scoring (lowercased, as-is)."""
        return label.lower()


# ---------------------------------------------------------------------------
# Dataset — full frames, read JPGs on the fly
# ---------------------------------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class WiTAFullFrameDataset(Dataset):
    """
    Walks the Kaggle WiTA layout:
        <root>/eng_<split>_<subset>/<subset>/<signer_dir>/<clip_idx>/<frames>.jpg
        <root>/eng_<split>_<subset>/<subset>/<signer_dir>/gt.txt   (label per clip_idx)

    Returns (frames[T=16,3,224,224] float normalized, target_ids).
    Full frame resized to 224x224 (no crop).
    """

    def __init__(self, root, split, converter: CharConverter,
                 subsets=("lex", "nonlex"), augment=False, t_in=T_IN,
                 img_size=IMG_SIZE):
        self.converter = converter
        self.augment = augment
        self.t_in = t_in
        self.img_size = img_size
        self.entries = []
        root = Path(root)
        for subset in subsets:
            base = root / f"eng_{split}_{subset}" / subset
            if not base.exists():
                # Some mounts omit the inner <subset>/ layer.
                alt = root / f"eng_{split}_{subset}"
                base = alt if alt.exists() else base
            if not base.exists():
                continue
            for signer_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                gt = signer_dir / "gt.txt"
                if not gt.exists():
                    continue
                try:
                    lines = open(gt, "r", encoding="utf-8", errors="replace").read().splitlines()
                except Exception:
                    continue
                for clip_dir in sorted(p for p in signer_dir.iterdir() if p.is_dir()):
                    try:
                        idx = int(clip_dir.name)
                    except ValueError:
                        continue
                    if idx >= len(lines):
                        continue
                    label = lines[idx].strip()
                    if not label:
                        continue
                    self.entries.append({
                        "dir": str(clip_dir),
                        "label": label,
                        "subset": subset,
                        "signer": signer_dir.name,
                    })
        print(f"[Stage14 {split}] {len(self.entries)} clips "
              f"({len({e['signer'] for e in self.entries})} signers)", flush=True)

    def __len__(self):
        return len(self.entries)

    def _load_frames(self, clip_dir):
        files = sorted(f for f in os.listdir(clip_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            raise RuntimeError(f"no frames in {clip_dir}")
        n = len(files)
        sel = np.linspace(0, n - 1, self.t_in).round().astype(int)
        sel = np.clip(sel, 0, n - 1)
        imgs = []
        for i in sel:
            img = Image.open(os.path.join(clip_dir, files[int(i)])).convert("RGB")
            # FULL FRAME, NO CROP: resize the whole frame to square 224x224.
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            imgs.append(np.asarray(img, dtype=np.uint8))
        return np.stack(imgs)                                  # [T, 224, 224, 3] uint8

    def __getitem__(self, idx):
        e = self.entries[idx]
        frames = self._load_frames(e["dir"])
        if self.augment and random.random() < 0.5:
            # Mild brightness jitter, same factor across the clip (cheap).
            bright = random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(np.float32) * bright, 0, 255).astype(np.uint8)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0   # [T,3,H,W]
        x = (x - _MEAN) / _STD
        target = self.converter.encode(e["label"])
        return x, target


def collate_fn(batch):
    videos = torch.stack([b[0] for b in batch])                 # [B,T,3,H,W]
    targets = [b[1] for b in batch]
    target_lens = torch.LongTensor([len(t) for t in targets])
    maxlen = int(max(target_lens.max().item(), 1))
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, t in enumerate(targets):
        if len(t) > 0:
            padded[i, :len(t)] = t
    return videos, padded, target_lens


# ---------------------------------------------------------------------------
# Model — VideoMAE + ConvTranspose1d upsample + CTC head
# ---------------------------------------------------------------------------

class VideoMAE_CTC(nn.Module):
    def __init__(self, vocab_size=VOCAB, t_upsample=3, gradient_checkpointing=True):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        if gradient_checkpointing:
            # use_reentrant=False so grads flow to unfrozen blocks even though
            # pixel_values has requires_grad=False (the Stage 12 lesson).
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
        self.d = self.backbone.config.hidden_size               # 768
        self.spatial = (self.backbone.config.image_size //
                        self.backbone.config.patch_size) ** 2    # 196
        self.upsample = nn.ConvTranspose1d(self.d, self.d,
                                           kernel_size=t_upsample, stride=t_upsample)
        self.norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, vocab_size)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values).last_hidden_state   # [B, N, 768]
        B, N, D = out.shape
        t_tubes = N // self.spatial
        out = out.view(B, t_tubes, self.spatial, D).mean(dim=2)            # [B, t_tubes, 768]
        out = out.transpose(1, 2)                                          # [B, 768, t_tubes]
        out = self.upsample(out)                                           # [B, 768, T_out]
        out = out.transpose(1, 2)                                          # [B, T_out, 768]
        out = self.norm(out)
        return self.head(out)                                              # [B, T_out, V]


# ---------------------------------------------------------------------------
# Gradual unfreezing
# ---------------------------------------------------------------------------

def set_phase(model: VideoMAE_CTC, phase: int):
    for p in model.backbone.parameters():
        p.requires_grad = False
    blocks = model.backbone.encoder.layer                       # 12 blocks (ViT-base)
    if phase == 1:
        pass
    elif phase == 2:
        for p in blocks[8:].parameters(): p.requires_grad = True
    elif phase == 3:
        for p in blocks[4:].parameters(): p.requires_grad = True
    elif phase == 4:
        for p in model.backbone.parameters(): p.requires_grad = True
    else:
        raise ValueError(phase)
    # Post-encoder LN (if present) trainable from phase 2 on.
    if phase >= 2 and hasattr(model.backbone, "layernorm"):
        for p in model.backbone.layernorm.parameters():
            p.requires_grad = True
    for m in (model.upsample, model.norm, model.head):
        for p in m.parameters():
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"phase": phase, "trainable": trainable, "total": total,
            "ratio": trainable / total}


def build_optimizer_for_phase(model: VideoMAE_CTC, phase: int):
    enc = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    phase_lr = {
        1: {"head": 5e-4, "enc": None},
        2: {"head": 5e-4, "enc": 5e-5},
        3: {"head": 2e-4, "enc": 1e-5},
        4: {"head": 5e-5, "enc": 5e-6},
    }[phase]
    groups = [{"params": head, "lr": phase_lr["head"], "weight_decay": 0.0}]
    if enc and phase_lr["enc"] is not None:
        groups.append({"params": enc, "lr": phase_lr["enc"], "weight_decay": 1e-2})
    return torch.optim.AdamW(groups)


# ---------------------------------------------------------------------------
# CER + train/val
# ---------------------------------------------------------------------------

def cer_score(ref, hyp):
    if editdistance is None:
        raise ImportError("pip install editdistance")
    err = editdistance.eval(ref, hyp)
    return min(err, len(ref)), len(ref)


@torch.no_grad()
def evaluate_cer(model, loader, converter, device, use_amp=True):
    model.eval()
    tot_e = tot_l = 0
    for videos, targets, target_lens in loader:
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(videos)
        preds = logits.argmax(-1)
        for b in range(videos.size(0)):
            pred = converter.decode_ctc(preds[b].cpu().tolist())
            gt_ids = targets[b, :int(target_lens[b])].tolist()
            gt = converter.decode_ctc(gt_ids)
            e, l = cer_score(gt, pred)
            tot_e += e; tot_l += l
    return tot_e / max(tot_l, 1)


def train_phase(model, train_loader, val_loader, converter, device,
                phase, epoch_start, epoch_end, best_val_cer,
                ckpt_path, scaler, log_every=50, use_amp=True):
    info = set_phase(model, phase)
    print(f"\n=== Phase {phase} (epochs {epoch_start}..{epoch_end-1}) ===  "
          f"trainable {info['trainable']/1e6:.2f}M / {info['total']/1e6:.2f}M "
          f"({100*info['ratio']:.1f}%)", flush=True)
    optimizer = build_optimizer_for_phase(model, phase)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

    for epoch in range(epoch_start, epoch_end):
        model.train()
        t0 = time.time()
        losses = []
        for bidx, (videos, targets, target_lens) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(videos)                              # [B,T_out,V]
            log_probs = logits.float().permute(1, 0, 2).log_softmax(-1)
            in_lens = torch.full((videos.size(0),), logits.size(1),
                                 dtype=torch.long, device=device)
            loss = ctc(log_probs, targets, in_lens, target_lens)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.item()))
            if (bidx + 1) % log_every == 0:
                print(f"  E{epoch:3d} [{bidx+1:4d}] loss={np.mean(losses[-log_every:]):.4f}",
                      flush=True)

        val_cer = evaluate_cer(model, val_loader, converter, device, use_amp=use_amp)
        dt = time.time() - t0
        print(f"E{epoch:3d}  train_loss={np.mean(losses):.4f}  val_cer={val_cer:.4f}  "
              f"({dt:.0f}s)", flush=True)

        is_best = val_cer < best_val_cer
        if is_best:
            best_val_cer = val_cer
        _save_ckpt(ckpt_path, model, optimizer, scaler, epoch + 1, phase,
                   val_cer, best_val_cer, is_best)
        if is_best:
            print(f"    * new best val_cer={val_cer:.4f}", flush=True)
    return best_val_cer


def _save_ckpt(ckpt_path, model, optimizer, scaler, next_epoch, phase,
               val_cer, best_val_cer, is_best):
    payload = {
        "model": model.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": next_epoch, "phase": phase,
        "val_cer": val_cer, "best_val_cer": best_val_cer,
    }
    torch.save(payload, ckpt_path)
    if is_best:
        torch.save(payload, ckpt_path.replace(".pt", "_best.pt"))


# ---------------------------------------------------------------------------
# One-shot test evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test(model, test_ds, converter, device, batch_size=4,
                  num_workers=2, use_amp=True):
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=collate_fn)
    model.eval()
    per_subset = {"lex": {"e": 0, "l": 0, "n": 0}, "nonlex": {"e": 0, "l": 0, "n": 0}}
    per_length = {b: {"e": 0, "l": 0} for b in ("1-4", "5-8", "9-12", "13+")}
    per_signer = {}
    base = 0
    for videos, targets, target_lens in loader:
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(videos)
        preds = logits.argmax(-1)
        for b in range(videos.size(0)):
            e = test_ds.entries[base + b]
            subset = e["subset"]; signer = e["signer"]
            pred = converter.decode_ctc(preds[b].cpu().tolist())
            gt = converter.gt_string(e["label"])
            err = editdistance.eval(gt, pred); L = len(gt)
            if err > L: err = L
            per_subset[subset]["e"] += err; per_subset[subset]["l"] += L
            per_subset[subset]["n"] += 1
            bucket = ("1-4" if L <= 4 else "5-8" if L <= 8 else "9-12" if L <= 12 else "13+")
            per_length[bucket]["e"] += err; per_length[bucket]["l"] += L
            sd = per_signer.setdefault(signer, {"e": 0, "l": 0})
            sd["e"] += err; sd["l"] += L
        base += videos.size(0)

    lex = per_subset["lex"]["e"] / max(per_subset["lex"]["l"], 1)
    non = per_subset["nonlex"]["e"] / max(per_subset["nonlex"]["l"], 1)
    tot_e = per_subset["lex"]["e"] + per_subset["nonlex"]["e"]
    tot_l = per_subset["lex"]["l"] + per_subset["nonlex"]["l"]
    return {
        "test_lex_cer": lex,
        "test_nonlex_cer": non,
        "test_overall_cer": tot_e / max(tot_l, 1),
        "per_length_cer": {k: v["e"] / max(v["l"], 1) for k, v in per_length.items()},
        "per_signer_cer": {k: v["e"] / max(v["l"], 1) for k, v in per_signer.items()},
        "n_clips_per_subset": {k: v["n"] for k, v in per_subset.items()},
        "paper_baseline": {"lex": 0.281, "nonlex": 0.365, "overall": 0.2924},
        "stage12_videomae_lora": {"overall": 0.5913, "note": "MediaPipe crop + LoRA + landmark fusion"},
    }

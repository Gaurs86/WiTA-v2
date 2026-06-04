"""
stage16/temporal_ctc.py — compact temporal model + CTC over cached CLIP features.

The frozen CLIP ViT already ran (features cached as [T, 512] per clip).  Here we
train a small BiLSTM (default) or Transformer encoder + CTC head (V=28) on those
features.  T4-friendly: tiny model, large batches, ViT never runs.

Includes:
  TemporalAugment       frame-drop / speed-perturb / temporal-subsample on [T,D]
  CachedFeatureDataset  loads <SIGNER>__<clip_id>.npy, joins labels, train aug
  collate_fn            pads features + targets, returns CTC input lengths
  TemporalCTC           input-proj -> BiLSTM/Transformer -> Linear(.,28)
  train_ctc             AMP CTC training with per-subset greedy-CER validation
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage16.common import (CharConverter, walk_clips, clip_cache_name, VOCAB,  # noqa: E402
                            cer_pair, gt_string)


# ---------------------------------------------------------------------------
# Temporal augmentation (on cached [T, D] features; cheap, no ViT)
# ---------------------------------------------------------------------------

class TemporalAugment:
    """Label-preserving temporal aug.  Keeps the FULL temporal span (never crops
    away the start/end of the writing, which would change the CTC label) -- only
    drops/duplicates interior frames and mildly resamples speed."""

    def __init__(self, p_drop=0.1, p_speed=0.5, speed_min=0.8, speed_max=1.2,
                 min_len=8):
        self.p_drop = p_drop
        self.p_speed = p_speed
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.min_len = min_len

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        T = feats.shape[0]
        # Speed perturbation: resample to T' = T / speed (keeps first+last frame).
        if random.random() < self.p_speed and T > self.min_len:
            speed = random.uniform(self.speed_min, self.speed_max)
            new_t = max(self.min_len, int(round(T / speed)))
            idx = np.linspace(0, T - 1, new_t).round().astype(int)
            feats = feats[idx]
            T = feats.shape[0]
        # Interior frame dropping (never drop first/last).
        if self.p_drop > 0 and T > self.min_len + 2:
            keep = [0]
            for i in range(1, T - 1):
                if random.random() > self.p_drop:
                    keep.append(i)
            keep.append(T - 1)
            if len(keep) >= self.min_len:
                feats = feats[keep]
        return feats


# ---------------------------------------------------------------------------
# Dataset over cached features
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    def __init__(self, data_root, cache_root, split, converter: CharConverter,
                 subsets=("lex", "nonlex"), augment=False):
        self.converter = converter
        self.aug = TemporalAugment() if augment else None
        cache_root = Path(cache_root)
        self.entries = []
        miss = 0
        for e in walk_clips(data_root, split, subsets=subsets):
            f = cache_root / split / e["subset"] / f"{clip_cache_name(e['signer'], e['clip_id'])}.npy"
            if not f.exists():
                miss += 1; continue
            e = dict(e); e["feat"] = str(f)
            self.entries.append(e)
        print(f"[Stage16 {split}] {len(self.entries)} clips "
              f"({len({e['signer'] for e in self.entries})} signers); missing feats={miss}",
              flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        feats = np.load(e["feat"]).astype(np.float32)      # [T, 512]
        if self.aug is not None:
            feats = self.aug(feats)
        x = torch.from_numpy(feats)                         # [T, 512]
        y = torch.LongTensor(self.converter.encode(e["label"]))
        return x, y, e["subset"], e["signer"]


def collate_fn(batch):
    xs, ys, subs, signers = zip(*batch)
    in_lens = torch.LongTensor([x.shape[0] for x in xs])
    Tmax = int(in_lens.max())
    D = xs[0].shape[1]
    feats = torch.zeros(len(xs), Tmax, D)
    for i, x in enumerate(xs):
        feats[i, :x.shape[0]] = x
    tgt_lens = torch.LongTensor([len(y) for y in ys])
    targets = torch.cat(ys) if len(ys) else torch.LongTensor([])
    return feats, targets, in_lens, tgt_lens, list(subs), list(signers)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TemporalCTC(nn.Module):
    def __init__(self, in_dim=512, d_model=256, vocab=VOCAB, backbone="bilstm",
                 n_layers=3, n_heads=4, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model),
                                  nn.GELU(), nn.Dropout(dropout))
        self.backbone = backbone
        if backbone == "bilstm":
            self.rnn = nn.LSTM(d_model, d_model, num_layers=n_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
            head_in = 2 * d_model
        elif backbone == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=4 * d_model,
                                               dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            head_in = d_model
        else:
            raise ValueError(backbone)
        self.head = nn.Linear(head_in, vocab)

    def forward(self, x, in_lens=None):
        # x: [B, T, in_dim] -> log_probs [B, T, V]
        h = self.proj(x)
        if self.backbone == "bilstm":
            if in_lens is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    h, in_lens.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.rnn(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            else:
                h, _ = self.rnn(h)
        else:
            pad_mask = None
            if in_lens is not None:
                T = h.size(1)
                pad_mask = torch.arange(T, device=h.device)[None, :] >= in_lens[:, None].to(h.device)
            h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.head(h).log_softmax(-1)                # [B, T, V]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _greedy(log_probs_bt_v, in_lens, converter):
    out = []
    am = log_probs_bt_v.argmax(-1)
    for b in range(am.shape[0]):
        out.append(converter.decode_ctc(am[b, : int(in_lens[b])].tolist()))
    return out


@torch.no_grad()
def evaluate(model, loader, converter, device, use_amp=True):
    model.eval()
    agg = {s: {"e": 0, "l": 0} for s in ("lex", "nonlex")}
    for feats, targets, in_lens, tgt_lens, subs, _ in loader:
        feats = feats.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            lp = model(feats, in_lens.to(device))
        preds = _greedy(lp.float().cpu(), in_lens, converter)
        # rebuild gt strings from targets
        off = 0
        for b in range(len(subs)):
            L = int(tgt_lens[b])
            gt = converter.ids_to_text(targets[off:off + L].tolist()); off += L
            e, l = cer_pair(gt, preds[b])
            agg[subs[b]]["e"] += e; agg[subs[b]]["l"] += l
    cer = {s: agg[s]["e"] / max(agg[s]["l"], 1) for s in agg}
    tot_e = sum(agg[s]["e"] for s in agg); tot_l = sum(agg[s]["l"] for s in agg)
    cer["overall"] = tot_e / max(tot_l, 1)
    return cer


def train_ctc(data_root, cache_root, *, out_dir, backbone="bilstm", d_model=256,
              n_layers=3, dropout=0.3, epochs=40, batch=64, lr=3e-4, wd=1e-2,
              num_workers=2, use_amp=True, seed=42, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    converter = CharConverter()

    tr = CachedFeatureDataset(data_root, cache_root, "train", converter, augment=True)
    va = CachedFeatureDataset(data_root, cache_root, "val", converter, augment=False)
    trl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=num_workers,
                     collate_fn=collate_fn, drop_last=True, pin_memory=True)
    val = DataLoader(va, batch_size=batch, shuffle=False, num_workers=num_workers,
                     collate_fn=collate_fn, pin_memory=True)

    model = TemporalCTC(d_model=d_model, backbone=backbone, n_layers=n_layers,
                        dropout=dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage16] {backbone} params={n_par/1e6:.2f}M  epochs={epochs} batch={batch}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * max(len(trl), 1),
                                                pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = float("inf"); best_payload = {}
    for ep in range(epochs):
        model.train(); t0 = time.time(); losses = []
        tr_e = tr_l = 0
        seen = 0; samp_gt = samp_pred = ""          # one reservoir-sampled example
        for feats, targets, in_lens, tgt_lens, _, _ in trl:
            feats = feats.to(device); targets = targets.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                lp = model(feats, in_lens.to(device))       # [B,T,V]
            loss = ctc(lp.permute(1, 0, 2).float(), targets,
                       in_lens.to(device), tgt_lens.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.item()))
            # TRAIN-CER diagnostic (greedy).  Decisive: if train_cer -> ~0 while
            # val stays high, the model fits train but CLIP features don't
            # generalize across signers; if train_cer stays high, the features
            # lack the trajectory signal (feature-limited) and more epochs cannot
            # help.  Only the small argmax is moved to CPU, so overhead is tiny.
            with torch.no_grad():
                am = lp.argmax(-1).cpu()
                tgt_cpu = targets.detach().cpu().tolist()
                off = 0
                for b in range(am.shape[0]):
                    pred = converter.decode_ctc(am[b, :int(in_lens[b])].tolist())
                    L = int(tgt_lens[b])
                    gt = converter.ids_to_text(tgt_cpu[off:off + L]); off += L
                    e, l = cer_pair(gt, pred); tr_e += e; tr_l += l
                    seen += 1                       # reservoir-sample 1 example/epoch
                    if random.random() < 1.0 / seen:
                        samp_gt, samp_pred = gt, pred
        tr_cer = tr_e / max(tr_l, 1)
        lr_now = opt.param_groups[0]["lr"]
        cer = evaluate(model, val, converter, device, use_amp=use_amp)
        print(f"E{ep:3d}/{epochs} loss={np.mean(losses):.4f} train_cer={tr_cer:.4f} "
              f"lr={lr_now:.2e} val overall={cer['overall']:.4f} lex={cer['lex']:.4f} "
              f"nonlex={cer['nonlex']:.4f}  {time.time()-t0:.0f}s", flush=True)
        print(f"      ex  gt='{samp_gt}'  ctc='{samp_pred}'", flush=True)
        if cer["overall"] < best:
            best = cer["overall"]
            best_payload = {"epoch": ep, **cer}
            torch.save({"model": model.state_dict(), "cer": cer, "epoch": ep,
                        "backbone": backbone, "d_model": d_model,
                        "n_layers": n_layers}, os.path.join(out_dir, "best.pt"))
            print(f"    * new best overall={best:.4f}", flush=True)
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"best": best_payload}, f, indent=2)
    return best_payload


def _selftest():
    """Build model + 50-step overfit on random data; loss must fall."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conv = CharConverter()
    m = TemporalCTC().to(dev)
    x = torch.randn(2, 32, 512, device=dev)
    in_lens = torch.LongTensor([32, 30])
    lp = m(x, in_lens)
    print("forward:", tuple(lp.shape), "(expect [2,32,28])")
    tgt = torch.cat([torch.LongTensor(conv.encode("cat")),
                     torch.LongTensor(conv.encode("dog"))]).to(dev)
    tl = torch.LongTensor([len(conv.encode("cat")), len(conv.encode("dog"))])
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for s in range(50):
        opt.zero_grad()
        lp = m(x, in_lens)
        loss = ctc(lp.permute(1, 0, 2), tgt, in_lens.to(dev), tl.to(dev))
        loss.backward(); opt.step()
        if s % 10 == 0:
            print(f"  step {s}: loss={loss.item():.4f}")
    print(f"final loss={loss.item():.4f} (should be small if wired correctly)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--cache_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="stage16_ctc")
    ap.add_argument("--backbone", type=str, default="bilstm", choices=["bilstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        from stage16.common import find_data_root
        dr = args.data_root or find_data_root()
        train_ctc(dr, args.cache_root, out_dir=args.out_dir, backbone=args.backbone,
                  epochs=args.epochs, batch=args.batch, lr=args.lr)

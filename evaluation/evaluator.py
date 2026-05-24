"""
evaluation/evaluator.py — CER evaluation + qualitative sample printer.
Self-contained; uses only internal modules.
"""
from __future__ import annotations
import random, logging
from typing import Literal

import torch
from torch.utils.data import DataLoader

from ..configs.default import Config

def _unwrap(m):
    import torch.nn as nn
    return m.module if isinstance(m, nn.DataParallel) else m
from ..datasets.vocab import StrLabelConverter, cer as compute_cer
from .metrics import decode_ctc_indices, decode_attn_indices
from ..training.losses import prepare_attn_targets

logger = logging.getLogger(__name__)


def evaluate_cer(
    model,
    dataloader:   DataLoader,
    converter:    StrLabelConverter,
    cfg:          Config,
    decode_mode:  Literal["ctc", "attn"] = "ctc",
    max_batches:  int | None = None,
) -> tuple[float, list[tuple[str, str]]]:
    """
    Compute mean CER over the dataloader.

    Parameters
    ----------
    max_batches : if set, stop early (fast validation pass).
                  Defaults to cfg.train.val_limit.

    Returns (mean_cer, [(gt_str, pred_str), …])
    """
    if max_batches is None:
        max_batches = cfg.train.val_limit

    device = cfg.device
    vc     = cfg.vocab
    model.eval()

    total_err = total_len = 0
    pairs: list[tuple[str, str]] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            clips, labels, input_lens, label_lens = batch
            clips      = clips.to(device)
            labels     = labels.to(device)
            input_lens = input_lens.to(device)
            label_lens = label_lens.to(device)
            B          = clips.shape[0]

            if decode_mode == "ctc":
                tgt_in, tgt_out, tgt_pad = prepare_attn_targets(labels, label_lens, vc)
                ctc_lp, _, _ = model(clips, input_lens, tgt_in, tgt_pad)
                pred_seqs  = _unwrap(model).decode_ctc_greedy(clips, input_lens)
            else:
                pred_t    = _unwrap(model).decode_attention(clips, input_lens)
                pred_seqs = [pred_t[b].tolist() for b in range(B)]

            for b in range(B):
                gt_str = converter.decode(
                    labels[b, :label_lens[b].item()].int(),
                    torch.IntTensor([label_lens[b].item()]),
                )
                if decode_mode == "ctc":
                    pred_str = decode_ctc_indices(pred_seqs[b], converter)
                else:
                    pred_str = decode_attn_indices(
                        pred_seqs[b], converter, vc.sos_idx, vc.eos_idx, vc.pad_idx)

                err, length = compute_cer(gt_str, pred_str)
                err = min(err, length)
                total_err += err
                total_len += length
                pairs.append((gt_str, pred_str))

    mean_cer = total_err / max(total_len, 1)
    return mean_cer, pairs


def print_sample_table(
    model,
    dataloader:  DataLoader,
    converter:   StrLabelConverter,
    cfg:         Config,
    epoch:       int | None = None,
    max_batches: int | None = None,
) -> None:
    """Print GT vs CTC vs Attn predictions for random validation samples."""
    if max_batches is None:
        max_batches = cfg.train.val_limit

    device = cfg.device
    vc     = cfg.vocab
    n_samples = cfg.train.qual_n
    model.eval()

    all_gt, all_ctc, all_attn = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            clips, labels, input_lens, label_lens = batch
            clips      = clips.to(device)
            labels     = labels.to(device)
            input_lens = input_lens.to(device)
            label_lens = label_lens.to(device)
            B = clips.shape[0]

            ctc_seqs  = _unwrap(model).decode_ctc_greedy(clips, input_lens)
            attn_t    = _unwrap(model).decode_attention(clips, input_lens)
            attn_seqs = [attn_t[b].tolist() for b in range(B)]

            for b in range(B):
                gt = converter.decode(
                    labels[b, :label_lens[b].item()].int(),
                    torch.IntTensor([label_lens[b].item()]),
                )
                all_gt.append(gt)
                all_ctc.append(decode_ctc_indices(ctc_seqs[b], converter))
                all_attn.append(decode_attn_indices(
                    attn_seqs[b], converter, vc.sos_idx, vc.eos_idx, vc.pad_idx))

    n    = min(n_samples, len(all_gt))
    idxs = random.sample(range(len(all_gt)), n) if len(all_gt) >= n else list(range(len(all_gt)))

    W   = 76
    hdr = f"\n{'═'*W}\n  Qualitative samples"
    if epoch: hdr += f" — epoch {epoch}"
    hdr += f"\n{'═'*W}"
    fmt = "{:<18} {:<18} {:<18} {:>8} {:>9}"
    rows = [hdr, fmt.format("Ground Truth", "CTC Pred", "Attn Pred", "CTC CER", "Attn CER"), "─"*W]

    for i in idxs:
        gt, ctc, attn = all_gt[i], all_ctc[i], all_attn[i]
        c_err, c_len  = compute_cer(gt, ctc)
        a_err, a_len  = compute_cer(gt, attn)
        rows.append(fmt.format(
            gt[:18], ctc[:18], attn[:18],
            f"{c_err/max(c_len,1):.3f}",
            f"{a_err/max(a_len,1):.3f}",
        ))

    rows.append("═" * W)
    out = "\n".join(rows)
    print(out)
    logger.info(out)

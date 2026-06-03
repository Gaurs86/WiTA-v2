"""
sanity_check.py — pre-flight checks before launching Stage 13B training.

Run from inside paper_replication/:
    python sanity_check.py --data_path_train=$HOME/wita-data/english/train

Checks:
  S1  Data path & glob count.
  S2  Model construction + forward (CTC + attn branches).
  S3  Single-batch overfit attempt (50 iters; loss should fall < 0.5).
  S4  calc_seq_len ratios.
  S5  Best-val checkpoint reload (skipped here -- happens after training).
"""

import os
import sys
import glob
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from options import AirTypingOptions
from model   import GestureTranslator
from data    import AirTypingDataset
from train   import pad_collate, build_attn_io
from utils   import calc_seq_len


def s1_check_data(opts):
    print("\n=== S1: data path + clip count ===")
    for tag, path in (('train', opts.data_path_train),
                      ('val',   opts.data_path_val),
                      ('test',  opts.data_path_test)):
        clips = glob.glob(os.path.join(path, '*/*/*'))
        clips = [c for c in clips if not c.endswith('gt.txt') and not c.endswith('gt2.txt')]
        gts   = glob.glob(os.path.join(path, '*/*/gt.txt'))
        print(f"  {tag:5s}  {path}\n        clip_dirs={len(clips)}  gt_files={len(gts)}")
        assert len(clips) > 0, f"no clips found under {path}"
    print("S1 OK")


def s2_check_model(opts):
    """
    Build model + smoke forward.

    IMPORTANT: the paper's resnet3d.py switches between two squeeze
    patterns based on `opts.batch_size / cuda_device_count <= 1`.  If
    we use a B=2 input tensor while opts.batch_size=1 (the argparse
    default), we hit the squeeze-unsqueeze branch with a tensor it
    can't reshape -- RuntimeError on permute.

    Fix: use a smoke-test batch size that lines up with opts.batch_size,
    so we exercise the same code path training will use.
    """
    print("\n=== S2: model construction + forward ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GestureTranslator(opts).to(device)
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  total params:    {total/1e6:.2f} M")
    print(f"  trainable:       {trainable/1e6:.2f} M")

    # Pick a smoke-test B that matches opts so the paper's branch logic
    # in resnet3d.py exercises the same code path as training.
    B = max(int(opts.batch_size), 1)
    # Cap at 4 to keep the smoke test cheap on memory.
    B = min(B, 4)
    T = 64
    x = torch.rand(B, T, 3, 112, 112).to(device)
    x_lens = torch.LongTensor([calc_seq_len(T)] * B).to(device)
    sample_label = [m.SOS_TOKEN, 3, 8, 15, 12, m.PAD_TOKEN, m.PAD_TOKEN]
    attn_in = torch.tensor([sample_label] * B).to(device)
    ctc, attn = m(x, x_lens, attn_input=attn_in)
    print(f"  smoke-test B     : {B}")
    print(f"  CTC logits shape : {tuple(ctc.shape)}   (expect [{B}, ~16, {m.vocab_ctc}])")
    if attn is not None:
        print(f"  Attn logits shape: {tuple(attn.shape)}   (expect [{B}, {len(sample_label)}, {m.vocab_attn}])")
    print("S2 OK")
    return m, device


def s3_overfit(opts, model, device, n_iters=50):
    print(f"\n=== S3: single-batch overfit ({n_iters} iters; expect loss < 0.5) ===")
    data_train = AirTypingDataset(opts, opts.data_path_train)
    # Use a small batch that still lands on the paper's else-branch in
    # resnet3d.py (which needs B > opts.batch_size/device_count threshold).
    # If user passed --batch_size>=2 this is consistent; if they passed
    # batch_size=1 we bump the smoke batch to 1 to stay on the same branch.
    smoke_B = max(2, min(int(opts.batch_size), 4))
    if int(opts.batch_size) <= 1:
        smoke_B = 1   # match paper's unsqueeze branch
    idxs = list(range(min(max(smoke_B * 2, 8), len(data_train))))
    subset = Subset(data_train, idxs)
    loader = DataLoader(subset, batch_size=smoke_B, shuffle=False,
                        num_workers=0, collate_fn=pad_collate, drop_last=False)
    print(f"  smoke-test loader batch = {smoke_B}")
    batch = next(iter(loader))
    xx_pad, yy_pad, x_lens, y_lens = batch
    xx_pad, yy_pad = xx_pad.to(device), yy_pad.to(device)
    x_lens, y_lens = x_lens.to(device), y_lens.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    ctc = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    use_joint = bool(opts.use_joint_decoder) and opts.loss_type == 'joint'

    for it in range(n_iters):
        optimizer.zero_grad()
        if use_joint:
            attn_in, attn_tg = build_attn_io(yy_pad, y_lens)
            ctc_logits, attn_logits = model(xx_pad, x_lens, attn_input=attn_in)
            cl = ctc(ctc_logits.permute(1, 0, 2).log_softmax(-1), yy_pad, x_lens, y_lens)
            al = F.cross_entropy(
                attn_logits.reshape(-1, attn_logits.size(-1)),
                attn_tg.reshape(-1),
                ignore_index=GestureTranslator.PAD_TOKEN,
                label_smoothing=opts.label_smoothing,
            )
            loss = opts.lambda_ctc * cl + (1 - opts.lambda_ctc) * al
            if it % 10 == 0 or it == n_iters - 1:
                print(f"  iter {it:3d}  total={loss.item():.4f}  ctc={cl.item():.4f}  attn={al.item():.4f}")
        else:
            ctc_logits, _ = model(xx_pad, x_lens)
            loss = ctc(ctc_logits.permute(1, 0, 2).log_softmax(-1), yy_pad, x_lens, y_lens)
            if it % 10 == 0 or it == n_iters - 1:
                print(f"  iter {it:3d}  ctc={loss.item():.4f}")
        loss.backward()
        optimizer.step()

    final = float(loss.item())
    if final >= 0.5:
        print(f"  WARNING: loss={final:.4f} >= 0.5 after {n_iters} iters; pipeline may be wrong.")
    else:
        print(f"  loss={final:.4f} < 0.5 -- S3 OK")


def s4_seqlen():
    print("\n=== S4: calc_seq_len ratios ===")
    for T_in in [40, 60, 80, 100]:
        T_out = calc_seq_len(T_in)
        print(f"  T_in={T_in:3d}  T_out={T_out:3d}  ratio={T_out/T_in:.2f}")
        if T_out < 10:
            print(f"    WARNING: T_out very small -- CTC feasibility may be tight for long words.")
    print("S4 OK")


def main():
    opts = AirTypingOptions().parse()
    s1_check_data(opts)
    model, device = s2_check_model(opts)
    s4_seqlen()
    s3_overfit(opts, model, device, n_iters=50)
    print("\nAll sanity checks complete.")


if __name__ == "__main__":
    main()

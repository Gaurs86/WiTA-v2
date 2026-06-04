"""
eval_test.py — Stage 13B one-shot test-set evaluation.

Loads a best-val checkpoint, runs over the test split EXACTLY ONCE, writes
test_eval_<model_name>.json next to the checkpoint, and creates a marker
file `.stage13b_test_evaluated` so accidental re-runs raise rather than
overwrite a published number.

Usage:
  python eval_test.py \
      --model_name=stage13b_test \
      --model_type=r3d --num_res_layer=1 --pooling_type=average --img_size=112 \
      --data_type=english --batch_size=1 --num_workers=4 \
      --loss_type=joint --use_joint_decoder=True \
      --lambda_ctc=0.5 --attn_decoder_layers=2 --attn_decoder_heads=4 \
      --load_dir=$HOME/wita-stage13/WiTA/stage13b_joint/models/best \
      --data_path_test=$HOME/wita-data/english/test
"""

import os
import sys
import json
import torch
import editdistance
from torch.utils.data import DataLoader

from data    import AirTypingDataset
from model   import GestureTranslator
from options import AirTypingOptions
from train   import pad_collate


def _decode_attn(token_ids, converter):
    out = []
    for t in token_ids:
        if t == GestureTranslator.EOS_TOKEN or t == GestureTranslator.PAD_TOKEN:
            break
        if t == GestureTranslator.SOS_TOKEN:
            continue
        if 1 <= t <= len(converter.alphabet):
            ch = converter.alphabet[t - 1]
            if ch != '-':
                out.append(ch)
    return ''.join(out)


def _length_bucket(L: int) -> str:
    if L <= 4:  return '1-4'
    if L <= 8:  return '5-8'
    if L <= 12: return '9-12'
    return '13+'


def evaluate(opts):
    device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")

    data_test = AirTypingDataset(opts, opts.data_path_test)
    loader = DataLoader(
        data_test, batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.num_workers, collate_fn=pad_collate, drop_last=False,
    )

    model = GestureTranslator(opts).to(device).eval()
    ckpt = os.path.join(opts.load_dir, "model.pth")
    assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"
    state = torch.load(ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval_test] loaded {ckpt}")
    if missing:    print(f"  missing keys    : {len(missing)}")
    if unexpected: print(f"  unexpected keys : {len(unexpected)}")

    per_subset = {'lex':    {'err': 0, 'len': 0, 'n': 0},
                  'nonlex': {'err': 0, 'len': 0, 'n': 0}}
    per_length = {b: {'err': 0, 'len': 0, 'n': 0}
                  for b in ('1-4', '5-8', '9-12', '13+')}
    per_signer: dict[str, dict[str, int]] = {}
    pairs: list[dict] = []

    use_joint = bool(opts.use_joint_decoder) and opts.loss_type == 'joint'
    print(f"[eval_test] use_joint={use_joint}  loss_type={opts.loss_type}")
    print(f"[eval_test] n_clips={len(data_test)}")

    with torch.no_grad():
        for idx in range(len(data_test)):
            video_path = data_test.video_list[idx]
            subset = 'lex' if '/lex/' in video_path else 'nonlex'
            # signer subdir = the second-to-last path component (the word index is last).
            signer = os.path.basename(os.path.dirname(video_path))
            label_str = data_test.labels[idx]

            video, _ = data_test[idx]
            video = video.unsqueeze(0).to(device)
            # CTC length downsample (paper convention r3d w/ num_res_layer=1).
            from utils import calc_seq_len
            x_lens = torch.LongTensor([calc_seq_len(video.size(1))]).to(device)

            if use_joint:
                pred_tokens = model.decode_attention(video, x_lens, max_len=opts.attn_max_len)
                pred = _decode_attn(pred_tokens[0].tolist(), data_test.converter)
            else:
                ctc_logits, _ = model(video, x_lens)
                ctc_log_probs = ctc_logits.log_softmax(-1)
                y_pred = ctc_log_probs[0].argmax(-1)
                T = int(x_lens[0].item())
                pred = data_test.converter.decode(y_pred[:T], torch.IntTensor([T]))

            err = editdistance.eval(label_str, pred)
            L = len(label_str)
            if err > L:
                err = L

            per_subset[subset]['err'] += err
            per_subset[subset]['len'] += L
            per_subset[subset]['n']   += 1
            bucket = _length_bucket(L)
            per_length[bucket]['err'] += err
            per_length[bucket]['len'] += L
            per_length[bucket]['n']   += 1
            sd = per_signer.setdefault(signer, {'err': 0, 'len': 0, 'n': 0})
            sd['err'] += err; sd['len'] += L; sd['n'] += 1

            pairs.append({'gt': label_str, 'pred': pred, 'edit': int(err),
                          'L': L, 'subset': subset, 'signer': signer})

            if (idx + 1) % 100 == 0 or (idx + 1) == len(data_test):
                cur_err = per_subset['lex']['err'] + per_subset['nonlex']['err']
                cur_len = per_subset['lex']['len'] + per_subset['nonlex']['len']
                cur_cer = cur_err / max(cur_len, 1)
                print(f"  [{idx+1}/{len(data_test)}]  running overall CER = {cur_cer:.4f}",
                      flush=True)

    lex_cer    = per_subset['lex']['err']    / max(per_subset['lex']['len'], 1)
    nonlex_cer = per_subset['nonlex']['err'] / max(per_subset['nonlex']['len'], 1)
    total_err  = per_subset['lex']['err'] + per_subset['nonlex']['err']
    total_len  = per_subset['lex']['len'] + per_subset['nonlex']['len']
    overall    = total_err / max(total_len, 1)
    per_length_cer = {k: v['err'] / max(v['len'], 1) for k, v in per_length.items()}
    per_signer_cer = {k: v['err'] / max(v['len'], 1) for k, v in per_signer.items()}

    return {
        'phase':            '13B' if use_joint else '13A',
        'model_name':       opts.model_name,
        'checkpoint':       ckpt,
        'n_test_clips':     len(data_test),
        'n_clips_per_subset': {k: v['n'] for k, v in per_subset.items()},
        'test_lex_cer':     lex_cer,
        'test_nonlex_cer':  nonlex_cer,
        'test_overall_cer': overall,
        'per_length_cer':   per_length_cer,
        'per_signer_cer':   per_signer_cer,
        'paper_baseline':   {'lex': 0.281, 'nonlex': 0.365, 'overall': 0.2924},
        'config': {
            'model_type':          opts.model_type,
            'num_res_layer':       opts.num_res_layer,
            'img_size':            opts.img_size,
            'use_joint_decoder':   bool(opts.use_joint_decoder),
            'loss_type':           opts.loss_type,
            'lambda_ctc':          opts.lambda_ctc,
            'attn_decoder_layers': opts.attn_decoder_layers,
            'attn_decoder_heads':  opts.attn_decoder_heads,
            'label_smoothing':     opts.label_smoothing,
        },
        'pairs_sample': pairs[:50],   # first 50 for sanity; full list in __full.json
        'pairs_all':    pairs,
    }


if __name__ == "__main__":
    _opts_obj = AirTypingOptions()
    # --eval_split controls the one-shot discipline:
    #   val  -> diagnostic, NO marker, can run freely (the playground).
    #   test -> the one-shot headline, marker-gated.
    # Point --data_path_test at whichever split you pass here.
    _opts_obj.parser.add_argument("--eval_split", type=str, default="test",
                                  choices=["val", "test"],
                                  help="'val' = free diagnostic (no marker); "
                                       "'test' = one-shot headline (marker-gated)")
    opts = _opts_obj.parse()
    split = opts.eval_split

    out_dir = opts.load_dir
    marker = os.path.join(out_dir, f".stage13b_{split}_evaluated")
    if split == "test" and os.path.exists(marker):
        print(f"ERROR: test already evaluated.  Marker: {marker}", file=sys.stderr)
        print("Per the §7 contract the test set is evaluated exactly once.", file=sys.stderr)
        print("If you intentionally need to re-run, delete the marker manually.", file=sys.stderr)
        sys.exit(2)

    decode = "ctc" if opts.loss_type == "ctc" or not opts.use_joint_decoder else "attn"
    result = evaluate(opts)
    result["eval_split"] = split
    result["decode"] = decode

    headline = {k: v for k, v in result.items() if k != 'pairs_all'}
    tag = f"{split}_{decode}_{opts.model_name}"
    head_path = os.path.join(out_dir, f"eval_{tag}.json")
    full_path = os.path.join(out_dir, f"eval_{tag}_full.json")
    with open(head_path, 'w') as f:
        json.dump(headline, f, indent=2, default=float)
    with open(full_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    if split == "test":
        with open(marker, 'w') as f:
            f.write(head_path)

    h = headline
    print("\n" + "=" * 64)
    print(f"  STAGE 13B {split.upper()} ({decode} decode)  -> {head_path}")
    print("=" * 64)
    print(f"  overall_cer : {h['test_overall_cer']:.4f}   (paper: 0.2924)")
    print(f"  lex_cer     : {h['test_lex_cer']:.4f}       (paper: 0.281)")
    print(f"  nonlex_cer  : {h['test_nonlex_cer']:.4f}     (paper: 0.365)")
    print("=" * 64)
    print(f"  per-length CER : {h['per_length_cer']}")
    print(f"  n_test_clips   : {h['n_test_clips']}  ({h['n_clips_per_subset']})")
    print()

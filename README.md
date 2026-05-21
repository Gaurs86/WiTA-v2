# WiTA v2 — Self-Contained Air-Writing Recognition

WiTA v2 is a clean-room reimplementation of the WiTA (Writing in the Air) air-writing recognition system. It is fully self-contained — no dependency on the original WiTA repository — and designed to run end-to-end on a single Kaggle T4 GPU.

The model combines a **3-D ResNet video encoder** with a **hybrid CTC + Transformer attention decoder**, trained jointly on the `yewon816/WiTA` HuggingFace dataset.

---

## Repository layout

```
wita_v2/
├── configs/
│   ├── __init__.py             # Re-exports Config, sub-configs, vocab constants
│   └── default.py              # Dataclass Config tree (VocabConfig … TrainConfig)
│
├── datasets/
│   ├── vocab.py                # StrLabelConverter, ALPHABET, cer(), wer()
│   ├── transforms.py           # ClipColorJitter, ClipRandomRotation, ClipToTensor …
│   ├── augmentations.py        # WiTAClipAugmentation (PIL + tensor pipeline)
│   ├── collate.py              # calc_seq_len_* helpers + make_pad_collate()
│   └── dataset.py              # WiTADataset + stream_and_index() + make_dataloaders()
│
├── models/
│   ├── encoders/
│   │   ├── resnet3d.py         # VideoResNet family (r3d / mc3 / rmc3 / r2plus1d / r2d)
│   │   └── registry.py         # Encoder registry (Phase 2 extension point)
│   ├── modules/
│   │   └── recurrent.py        # BiRNNHead, TransformerEncoderHead, CTCProjection
│   ├── decoders/
│   │   └── attention.py        # AttentionDecoder (Transformer seq2seq)
│   └── hybrid_model.py         # WiTAHybridModel + build_model(cfg)
│
├── training/
│   ├── losses.py               # hybrid_loss(), prepare_attn_targets(), get_lambda_ctc()
│   ├── schedulers.py           # WarmupMultiStepLR, build_scheduler(), build_optimizer()
│   └── trainer.py              # train() — AMP, grad accumulation, checkpointing
│
├── evaluation/
│   ├── metrics.py              # decode_ctc_indices(), decode_attn_indices()
│   └── evaluator.py            # evaluate_cer(), print_sample_table()
│
├── utils/
│   ├── checkpoint.py           # save() / load() with all 5 state-dicts
│   └── logging_utils.py        # setup()
│
├── scripts/
│   ├── train.py                # Thin wrapper → main.py
│   └── evaluate.py             # Standalone evaluation script
│
├── notebooks/
│   └── run_kaggle.ipynb        # 10-cell Kaggle notebook
│
├── main.py                     # CLI entry point (argparse)
├── requirements.txt
└── README.md
```

---

## Architecture

```
clips  [B, T, C, H, W]
       │
       ▼
VideoResNet encoder          (r3d / mc3 / r2plus1d …)
       │  features [B, T', enc_dim]
       ├─────────────────────────────────────┐
       ▼                                     ▼
BiRNN / Transformer head             AttentionDecoder
       │  rnn_out [B, T', rnn_dim]       attn_logits [B, L, attn_vocab]
       ▼
CTCProjection
       │  ctc_logits [T', B, ctc_vocab]
```

The two heads share the encoder but are otherwise independent. During training the total loss is:

```
loss = λ_ctc · CTC_loss + (1 − λ_ctc) · CrossEntropy_loss
```

`λ_ctc` is linearly annealed from `0.50` → `0.20` over the course of training so the attention decoder is gradually emphasised.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Korean support (optional):
pip install hgtk
```

### 2. Train with defaults (English, R3D encoder, BiLSTM head)

```bash
python main.py
```

This downloads the `yewon816/WiTA` dataset automatically via `huggingface_hub`, trains for 40 epochs with a micro-batch size of 4 and 4 gradient-accumulation steps (effective batch = 16), and saves checkpoints to `/kaggle/working/checkpoints/`.

### 3. Common overrides

```bash
# Smoke-test: 2 ZIPs, 2 epochs
python main.py --max_zips 2 --epochs 2

# Korean data
python main.py --lang korean

# Heavier encoder + GRU head
python main.py --arch r2plus1d --recurrent gru --enc_dim 512

# Resume from latest
python main.py --resume /kaggle/working/checkpoints/latest.pt

# Evaluate only (no training)
python main.py --eval_only --resume /kaggle/working/checkpoints/best.pt
```

### 4. Standalone evaluation

```bash
# Evaluate best.pt (default):
python scripts/evaluate.py

# Specify a checkpoint and write a JSON report:
python scripts/evaluate.py \
    --ckpt /kaggle/working/checkpoints/epoch_040.pt \
    --decode both \
    --show_n 40 \
    --out_json results.json
```

### 5. Kaggle notebook

Open `notebooks/run_kaggle.ipynb` and run all cells. The notebook handles dataset download, training (40 epochs), and a final evaluation table in 10 cells.

---

## Configuration

All configuration lives in `configs/default.py` as a tree of Python dataclasses. There are no global mutable singletons — every module receives a `Config` (or sub-config) object explicitly.

```python
from configs import Config, DataConfig, TrainConfig, EncoderConfig

cfg = Config(
    data    = DataConfig(lang="korean", max_zips=5),
    encoder = EncoderConfig(arch="r2plus1d", out_dim=512),
    train   = TrainConfig(epochs=60, batch_size=8, accum_steps=2),
)
cfg.build()   # finalises vocab indices and device
```

Key sub-configs and their most useful fields:

| Sub-config | Field | Default | Notes |
|---|---|---|---|
| `DataConfig` | `lang` | `"english"` | `"english"` / `"korean"` / `"both"` |
| `DataConfig` | `max_zips` | `None` | `int` for debug subset |
| `EncoderConfig` | `arch` | `"r3d"` | `r3d` / `mc3` / `rmc3` / `r2plus1d` / `r2d` |
| `EncoderConfig` | `out_dim` | `256` | Encoder output feature size |
| `RecurrentConfig` | `arch` | `"lstm"` | `lstm` / `gru` / `transformer` / `none` |
| `TrainConfig` | `batch_size` | `4` | Micro-batch per GPU forward pass |
| `TrainConfig` | `accum_steps` | `4` | Effective batch = `batch_size × accum_steps` |
| `TrainConfig` | `scheduler` | `"onecycle"` | `onecycle` / `warmup_multistep` / `steplr` / `none` |
| `TrainConfig` | `lambda_ctc_start` | `0.50` | Initial CTC loss weight |
| `TrainConfig` | `lambda_ctc_min` | `0.20` | Final CTC loss weight (after annealing) |

---

## Checkpoints

Checkpoints are saved under `cfg.train.checkpoint_dir` (default `/kaggle/working/checkpoints/`):

| File | When saved | Contents |
|---|---|---|
| `best.pt` | Each time val CER improves | model + optimiser + scheduler + scaler + epoch + best CER |
| `latest.pt` | Every epoch | Same |
| `epoch_NNN.pt` | Every `save_frequency` epochs | Same |
| `phase1_export.pt` | After training (or `--export`) | Model weights + vocab metadata only (no optimiser) |

All checkpoint files store five state-dicts: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, and epoch/metric metadata. Resuming an interrupted run preserves the exact learning-rate schedule position.

---

## Extending for Phase 2 (Video Swin Transformer)

The encoder registry (`models/encoders/registry.py`) is the single extension point. To add a new backbone:

1. Implement your backbone in `models/encoders/swin3d.py` with the interface `__init__(cfg: EncoderConfig)` and `forward(x: [B, C, T, H, W]) → [B, T', out_dim]`.

2. Register it at the bottom of that file:
   ```python
   from models.encoders.registry import register_encoder

   @register_encoder("swin_t")
   class VideoSwinTiny(nn.Module):
       ...
   ```

3. Add `"swin_t"` to the `Literal` type in `EncoderConfig.arch`.

4. Pass `--arch swin_t` to `main.py`. Nothing else changes.

---

## Metrics

| Metric | Definition | Location |
|---|---|---|
| CER | Character Error Rate = edit\_distance(gt, pred) / len(gt) | `datasets/vocab.py` → `cer()` |
| WER | Word Error Rate = edit\_distance(gt\_words, pred\_words) / len(gt\_words) | `datasets/vocab.py` → `wer()` |

Both are computed with the `editdistance` library (Levenshtein distance).

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | Core deep learning |
| `torchvision` | ≥ 0.16.0 | VideoResNet, transforms |
| `Pillow` | ≥ 9.0.0 | Frame decoding |
| `numpy` | ≥ 1.24.0 | Tensor/array ops |
| `editdistance` | ≥ 0.6.3 | CER / WER |
| `huggingface_hub` | ≥ 0.20.0 | Dataset download |
| `hgtk` | ≥ 0.1.3 | Korean jamo decomposition (optional) |

See `requirements.txt` for the full list with pinned ranges.

---

## Roadmap

- **Phase 1** (current) — VideoResNet encoder, BiLSTM/GRU/Transformer head, hybrid CTC+Attention, English & Korean.
- **Phase 2** — Video Swin Transformer encoder via `models/encoders/registry.py`; beam-search decoding; language model rescoring.
- **Phase 3** — Real-time inference pipeline; ONNX / TorchScript export.

---

## License

This codebase is an independent re-implementation for research purposes. The model architecture is based on the WiTA paper; the dataset is hosted at `yewon816/WiTA` on HuggingFace. Please respect the original dataset licence when using this code.

# Stage 13B — Paper replication + joint CTC + attention decoder

This directory is a **patched fork** of Kim et al. 2023's official WiTA repo
(`https://github.com/Uehwan/WiTA`), adapted for a single A100 80 GB run on
Thunder Compute.  The patch adds a joint CTC + 2-layer Transformer attention
decoder head on top of the paper's `r3d_10_avg_eng_aug` encoder.

## Files

| File | Source | Status |
|---|---|---|
| `data.py`            | paper | unchanged |
| `utils.py`           | paper | unchanged |
| `resnet3d.py`        | paper | unchanged |
| `video_transforms.py`| paper | unchanged |
| `options.py`         | paper | **patched** — adds joint-decoder + loss-type flags |
| `model.py`           | paper | **patched** — adds `GestureTranslator.{token_embed, pos_embed, attn_decoder, attn_head, decode_attention}` |
| `train.py`           | paper | **patched** — `build_attn_io`, joint loss, attention-decode val path, best-checkpoint stable folder |
| `eval_test.py`       | new   | one-shot test eval; gated by `.stage13b_test_evaluated` marker |
| `sanity_check.py`    | new   | S1–S4 pre-flight checks |
| `layout_dataset.sh`  | new   | symlinks the Kaggle dataset into the paper's expected directory tree |
| `run_stage13b.sh`    | new   | launch script (train + eval, one shot) |

## On the A100 — full execution sequence

### 0. SSH in, set up the working tree

```bash
mkdir -p ~/wita-stage13 && cd ~/wita-stage13
git clone -b stage13b-paper-replication \
    https://github.com/Gaurs86/WiTA-v2.git wita-v2
cd wita-v2/paper_replication
```

### 1. Install deps

```bash
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0
pip install Pillow hgtk editdistance numpy
pip install kaggle huggingface_hub
```

### 2. Pull the dataset from Kaggle

Set your Kaggle credentials once:

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"<your_kaggle_username>","key":"<your_kaggle_key>"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

Then download.  The user's dataset is `gaurs86/wita-full-english-122signers`:

```bash
mkdir -p ~/wita-data && cd ~/wita-data
kaggle datasets download -d gaurs86/wita-full-english-122signers --unzip
ls .   # should show eng_train_lex/  eng_train_nonlex/  eng_val_lex/  ...
```

Wall-clock: ~10–20 min depending on Thunder's network.  No local upload needed.

### 3. Symlink into the paper's expected layout

```bash
cd ~/wita-stage13/wita-v2/paper_replication
SRC=~/wita-data DST=~/wita-data/english bash layout_dataset.sh
```

This creates `~/wita-data/english/{train,val,test}/{lex,nonlex}/` as symlinks
to the existing `eng_*_*` directories.  Zero disk usage.

### 4. Sanity checks (~5 min)

```bash
python sanity_check.py \
    --data_type=english \
    --model_type=r3d --num_res_layer=1 --img_size=112 \
    --use_joint_decoder=True --loss_type=joint \
    --lambda_ctc=0.5 --label_smoothing=0.1 \
    --data_path_train=$HOME/wita-data/english/train \
    --data_path_val=$HOME/wita-data/english/val \
    --data_path_test=$HOME/wita-data/english/test
```

You should see four `OK` lines.  If S3 (overfit) doesn't break below 0.5 in
50 iters, **stop and debug before training**.

### 5. Train + one-shot test eval (~4 hours)

```bash
DATA_ROOT=$HOME/wita-data/english SAVE_NAME=stage13b_joint \
BATCH=16 EPOCHS=175 LR=1e-3 NUM_WORKERS=8 \
    bash run_stage13b.sh
```

The script trains, saves the rolling best checkpoint at
`stage13b_joint/models/best/`, then runs `eval_test.py` exactly once on the
test set and writes:

* `stage13b_joint/models/best/test_eval_stage13b_joint_test.json`     (headline)
* `stage13b_joint/models/best/test_eval_stage13b_joint_test_full.json` (per-clip)
* `stage13b_joint/models/best/.stage13b_test_evaluated`                (marker)

Re-running `eval_test.py` against the same checkpoint will refuse with a
non-zero exit (delete the marker manually if you intentionally need to).

## Test discipline

The test set is evaluated **exactly once** per training run, on the best-val
checkpoint, with no hyperparameter changes in response to the test number.
The marker file enforces this mechanically; the JSON output is your
publishable evidence.

## Cost estimate (Thunder Compute A100 80 GB at ~$0.80/hr)

| Step | Wall-clock | Cost |
|---|---|---|
| Setup + deps + data | ~45 min | $0.60 |
| Training (175 epochs, batch 16) | ~3.5 h | $2.80 |
| Test eval (one-shot) | ~5 min | $0.10 |
| **Total** | **~4.5 h** | **~$3.50** |

Leaves ~5 h of budget for a second training run if needed (e.g. tune
`lambda_ctc` on val, then re-train and re-evaluate test once).

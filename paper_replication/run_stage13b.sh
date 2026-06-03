#!/usr/bin/env bash
# Stage 13B launch — paper recipe + joint CTC+attention decoder.
# Run from inside paper_replication/.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/wita-data/english}"
SAVE_NAME="${SAVE_NAME:-stage13b_joint}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-175}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_FRAMES="${MAX_FRAMES:-0}"   # 0 = no cap; set e.g. 96 to bound memory on long clips

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name="${SAVE_NAME}" \
    --model_type=r3d \
    --pooling_type=average \
    --img_size=112 \
    --data_type=english \
    --optimizer_type=adam \
    --num_res_layer=1 \
    --data_augment=True \
    --batch_size="${BATCH}" \
    --num_workers="${NUM_WORKERS}" \
    --learning_rate="${LR}" \
    --num_epochs="${EPOCHS}" \
    --scheduler_type=warmup \
    --scheduler_step_size=5 \
    --scheduler_gamma=0.9 \
    --use_joint_decoder=True \
    --loss_type=joint \
    --lambda_ctc=0.5 \
    --attn_decoder_layers=2 \
    --attn_decoder_heads=4 \
    --label_smoothing=0.1 \
    --attn_max_len=32 \
    --max_frames="${MAX_FRAMES}" \
    --save_frequency=25 \
    --log_interval=50 \
    --seed_number=0 \
    --data_path_train="${DATA_ROOT}/train" \
    --data_path_val="${DATA_ROOT}/val" \
    --data_path_test="${DATA_ROOT}/test" \
    2>&1 | tee "logs/${SAVE_NAME}.log"

# One-shot test eval against the rolling best checkpoint.
python eval_test.py \
    --model_name="${SAVE_NAME}_test" \
    --model_type=r3d \
    --num_res_layer=1 \
    --pooling_type=average \
    --img_size=112 \
    --data_type=english \
    --batch_size=1 \
    --num_workers="${NUM_WORKERS}" \
    --loss_type=joint \
    --use_joint_decoder=True \
    --lambda_ctc=0.5 \
    --attn_decoder_layers=2 \
    --attn_decoder_heads=4 \
    --attn_max_len=32 \
    --max_frames="${MAX_FRAMES}" \
    --load_dir="${SAVE_NAME}/models/best" \
    --data_path_test="${DATA_ROOT}/test" \
    2>&1 | tee "logs/${SAVE_NAME}_test.log"

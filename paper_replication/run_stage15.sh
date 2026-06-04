#!/usr/bin/env bash
# Stage 15 (Path B) — PURE CTC + Focal CTC retrain.
#
# vs Stage 13B (which hit 0.44 lex): drop the joint attention decoder
# (it split capacity and dragged the CTC head), and add Focal CTC
# (Feng et al. 2019) to attend to low-frequency characters (English
# letter frequency is skewed -> targets the lex subset).  Reuses the
# MAX_FRAMES=64 decode cache (no rebuild).  After training, evaluate
# with CTC greedy + lexicon-constrained decode.
#
# Run from inside paper_replication/.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/wita-data/english}"
CACHE_DIR="${CACHE_DIR:-$HOME/wita-cache}"
SAVE_NAME="${SAVE_NAME:-stage15_focalctc}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-120}"        # ~8 min/epoch with cache -> ~16h -> ~$5-6
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_FRAMES="${MAX_FRAMES:-64}" # reuse the existing cache (built at 64)
USE_AMP="${USE_AMP:-True}"
FOCAL_GAMMA="${FOCAL_GAMMA:-1.0}"   # 0 = standard CTC; 1.0 a safe default, tune on val
FOCAL_ALPHA="${FOCAL_ALPHA:-1.0}"

export PYTHONHASHSEED=0
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
    --loss_type=ctc \
    --use_joint_decoder=False \
    --focal_gamma="${FOCAL_GAMMA}" \
    --focal_alpha="${FOCAL_ALPHA}" \
    --max_frames="${MAX_FRAMES}" \
    --use_amp="${USE_AMP}" \
    --cache_dir="${CACHE_DIR}" \
    --save_frequency=25 \
    --log_interval=50 \
    --seed_number=0 \
    --data_path_train="${DATA_ROOT}/train" \
    --data_path_val="${DATA_ROOT}/val" \
    --data_path_test="${DATA_ROOT}/test" \
    2>&1 | tee "logs/${SAVE_NAME}.log"

echo
echo "=== training done. Next: evaluate on VAL (CTC greedy + lexicon) ==="
echo "  # CTC greedy val:"
echo "  python eval_test.py --eval_split=val --model_name=${SAVE_NAME} \\"
echo "      --model_type=r3d --num_res_layer=1 --pooling_type=average --img_size=112 \\"
echo "      --data_type=english --batch_size=1 --num_workers=4 \\"
echo "      --loss_type=ctc --use_joint_decoder=False --max_frames=${MAX_FRAMES} \\"
echo "      --cache_dir=${CACHE_DIR} --load_dir=${SAVE_NAME}/models/best \\"
echo "      --data_path_test=${DATA_ROOT}/val"
echo
echo "  # lexicon-constrained val:"
echo "  python lexicon_decode.py --eval_split=val --model_name=${SAVE_NAME} \\"
echo "      --model_type=r3d --num_res_layer=1 --pooling_type=average --img_size=112 \\"
echo "      --data_type=english --num_workers=4 --loss_type=ctc --use_joint_decoder=False \\"
echo "      --max_frames=${MAX_FRAMES} --cache_dir=${CACHE_DIR} --load_dir=${SAVE_NAME}/models/best \\"
echo "      --train_root=${DATA_ROOT}/train --wordfreq_topk=30000 --len_window=4 \\"
echo "      --data_path_test=${DATA_ROOT}/val"

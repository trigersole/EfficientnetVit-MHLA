#!/bin/bash
# One 6-hour TC2 training segment for one arm. Resubmit (or use tc2_submit_chain.sh)
# until final.pt exists; the last segment then runs the official full-resolution evaluator.
#
# Usage: sbatch scripts/slurm/tc2_train.sbatch <e0|e1|e2|e3> [extra trainer args, smoke tests only]
#SBATCH --job-name=effvit-train
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
EXPERIMENT="${1:?usage: tc2_train.sbatch <experiment> [extra args]}"
shift
EXTRA_ARGS=("$@")

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/protocol.env"
cd "$REPO_DIR"
RUN_NAME="${RUN_NAME:-${EXPERIMENT}_m${MHLA_NUM_GROUPS}_s${SEED}}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

echo "job=$SLURM_JOB_ID node=$(hostname) run_dir=$RUN_DIR start=$(date -Is)"
echo "git=$(git rev-parse HEAD 2>/dev/null || cat REVISION 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

SLICE_FLAG="--no-inherit-sliced-proj"
[ "$INHERIT_SLICED_PROJ" = "1" ] && SLICE_FLAG="--inherit-sliced-proj"

if [ ! -f "$RUN_DIR/final.pt" ]; then
  "$PYTHON" applications/efficientvit_seg/train_efficientvit_seg_model.py \
    --recipe "$RECIPE" --data-root "$DATA_ROOT" --dataset cityscapes \
    --model "$MODEL" --experiment "$EXPERIMENT" --mhla-num-groups "$MHLA_NUM_GROUPS" \
    --output-dir "$RUN_DIR" --resume \
    --init-checkpoint "$INIT_CHECKPOINT" --init-scope "$INIT_SCOPE" "$SLICE_FLAG" \
    --batch-size "$BATCH_SIZE" --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --crop-height "$CROP_HEIGHT" --crop-width "$CROP_WIDTH" \
    --backbone-lr-multiplier "$BACKBONE_LR_MULTIPLIER" \
    --lr "$LR" --weight-decay "$WEIGHT_DECAY" --warmup-steps "$WARMUP_STEPS" \
    --min-lr-ratio "$MIN_LR_RATIO" --grad-clip "$GRAD_CLIP" \
    --max-steps "$MAX_STEPS" --seed "$SEED" \
    --eval-interval "$EVAL_INTERVAL" --val-max-samples "$VAL_MAX_SAMPLES" \
    --save-interval "$SAVE_INTERVAL" --log-interval "$LOG_INTERVAL" --workers "$WORKERS" \
    "${EXTRA_ARGS[@]}" &
  TRAIN_PID=$!
  # SLURM sends USR1 to this batch shell 5 min before the limit; forward it so the trainer saves last.pt.
  trap 'echo "USR1 received $(date -Is); forwarding"; kill -USR1 $TRAIN_PID 2>/dev/null' USR1
  set +e
  wait $TRAIN_PID; STATUS=$?
  while kill -0 $TRAIN_PID 2>/dev/null; do wait $TRAIN_PID; STATUS=$?; done
  set -e
  echo "trainer exit=$STATUS at $(date -Is)"
  [ "$STATUS" -ne 0 ] && [ ! -f "$RUN_DIR/last.pt" ] && exit "$STATUS"
fi

if [ -f "$RUN_DIR/final.pt" ] && [ ! -f "$RUN_DIR/eval_full.txt" ]; then
  echo "final.pt found; running official evaluator (500 val images, 1024x2048)"
  "$PYTHON" applications/efficientvit_seg/eval_efficientvit_seg_model.py \
    --dataset cityscapes --crop_size 1024 --model "$MODEL" \
    --experiment "$EXPERIMENT" --mhla_num_groups "$MHLA_NUM_GROUPS" \
    --path "$DATA_ROOT/leftImg8bit/val" --weight_url "$RUN_DIR/final.pt" \
    --workers 8 2>&1 | tee "$RUN_DIR/eval_full.txt"
fi
echo "end=$(date -Is)"

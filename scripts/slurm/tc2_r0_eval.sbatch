#!/bin/bash
# R0: score the OFFICIAL released EfficientViT-B1 Cityscapes checkpoint with our evaluator.
# Pass rule (SPEC §6): mIoU within 80.547 +- 0.3, i.e. 80.2-80.8.
# Usage: sbatch scripts/slurm/tc2_r0_eval.sbatch
#SBATCH --job-name=effvit-r0
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/protocol.env"
cd "$REPO_DIR"
OUT_DIR="$RUNS_ROOT/r0_official"
mkdir -p "$OUT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sha256sum "$OFFICIAL_SEG_CHECKPOINT"

"$PYTHON" applications/efficientvit_seg/eval_efficientvit_seg_model.py \
  --dataset cityscapes --crop_size 1024 --model "$MODEL" --experiment e0 \
  --path "$DATA_ROOT/leftImg8bit/val" --weight_url "$OFFICIAL_SEG_CHECKPOINT" \
  --workers 8 2>&1 | tee "$OUT_DIR/eval_full.txt"
echo "end=$(date -Is)"

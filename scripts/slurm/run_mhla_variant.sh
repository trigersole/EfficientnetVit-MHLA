#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <experiment> <mhla_groups> <run_name>" >&2
    exit 2
fi

EXPERIMENT="$1"
MHLA_GROUPS="$2"
RUN_NAME="$3"

: "${MHLA_MODE:?MHLA_MODE is required}"
: "${MHLA_REPO_ROOT:?MHLA_REPO_ROOT is required}"
: "${MHLA_OUTPUT_ROOT:?MHLA_OUTPUT_ROOT is required}"
: "${MHLA_MODEL_NAME:?MHLA_MODEL_NAME is required}"
: "${MHLA_DATASET:?MHLA_DATASET is required}"

RUN_DIR="${MHLA_OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CHECKPOINT_PATH="${MHLA_CHECKPOINT_TEMPLATE:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
    CHECKPOINT_PATH='{run_dir}/best.pt'
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH//\{run_dir\}/${RUN_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH//\{run_name\}/${RUN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH//\{experiment\}/${EXPERIMENT}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH//\{groups\}/${MHLA_GROUPS}}"

export MHLA_EXPERIMENT="${EXPERIMENT}"
export MHLA_NUM_GROUPS="${MHLA_GROUPS}"
export MHLA_RUN_NAME="${RUN_NAME}"
export MHLA_RUN_DIR="${RUN_DIR}"
export MHLA_CHECKPOINT_PATH="${CHECKPOINT_PATH}"

cd "${MHLA_REPO_ROOT}"

write_metadata() {
    {
        echo "job_id=${SLURM_JOB_ID:-local}"
        echo "step_id=${SLURM_STEP_ID:-local}"
        echo "host=$(hostname)"
        echo "experiment=${EXPERIMENT}"
        echo "mhla_num_groups=${MHLA_GROUPS}"
        echo "model=${MHLA_MODEL_NAME}"
        echo "dataset=${MHLA_DATASET}"
        echo "mode=${MHLA_MODE}"
        echo "checkpoint=${CHECKPOINT_PATH}"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "started_at=$(date --iso-8601=seconds)"
    } > "${RUN_DIR}/run_metadata.txt"
}

run_training() {
    if [[ -z "${MHLA_TRAIN_COMMAND:-}" ]]; then
        cat >&2 <<'EOF'
Training was requested, but --train-command was not supplied.
The upstream EfficientViT repository does not include a segmentation trainer.
Pass your course/existing trainer command; it can read the exported variables:
MHLA_EXPERIMENT, MHLA_NUM_GROUPS, MHLA_MODEL_NAME, MHLA_DATASET,
MHLA_TRAIN_DATA, MHLA_EVAL_PATH, MHLA_RUN_DIR, MHLA_CHECKPOINT_PATH,
and MHLA_GPUS_PER_EXPERIMENT.
EOF
        exit 2
    fi

    echo "[${RUN_NAME}] starting training"
    # shellcheck disable=SC2016
    bash -lc "${MHLA_TRAIN_COMMAND}" 2>&1 | tee "${RUN_DIR}/train.log"
}

run_evaluation() {
    if [[ -z "${MHLA_EVAL_PATH:-}" ]]; then
        echo "--eval-path is required for evaluate/train-eval mode" >&2
        exit 2
    fi
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
        exit 2
    fi

    echo "[${RUN_NAME}] evaluating ${CHECKPOINT_PATH}"
    python applications/efficientvit_seg/eval_efficientvit_seg_model.py \
        --path "${MHLA_EVAL_PATH}" \
        --dataset "${MHLA_DATASET}" \
        --crop_size "${MHLA_CROP_SIZE}" \
        --model "${MHLA_MODEL_NAME}" \
        --experiment "${EXPERIMENT}" \
        --mhla_num_groups "${MHLA_GROUPS}" \
        --weight_url "${CHECKPOINT_PATH}" \
        --gpu all \
        --batch_size "${MHLA_EVAL_BATCH_SIZE}" \
        --workers "${MHLA_WORKERS}" \
        2>&1 | tee "${RUN_DIR}/evaluation.log"
}

run_benchmark() {
    echo "[${RUN_NAME}] benchmarking"
    python applications/efficientvit_seg/benchmark_mhla.py \
        --model "${MHLA_MODEL_NAME}" \
        --variants "${EXPERIMENT}" \
        --mhla_num_groups "${MHLA_GROUPS}" \
        --input_size "${MHLA_INPUT_HEIGHT}" "${MHLA_INPUT_WIDTH}" \
        --batch_size "${MHLA_BENCHMARK_BATCH_SIZE}" \
        --device cuda \
        --warmup "${MHLA_BENCHMARK_WARMUP}" \
        --iterations "${MHLA_BENCHMARK_ITERATIONS}" \
        --seed "${MHLA_SEED}" \
        --output "${RUN_DIR}/benchmark.json" \
        2>&1 | tee "${RUN_DIR}/benchmark.log"
}

write_metadata

case "${MHLA_MODE}" in
    train)
        run_training
        ;;
    evaluate)
        run_evaluation
        ;;
    train-eval)
        run_training
        run_evaluation
        ;;
    benchmark)
        run_benchmark
        ;;
    evaluate-benchmark)
        run_evaluation
        run_benchmark
        ;;
    train-eval-benchmark)
        run_training
        run_evaluation
        run_benchmark
        ;;
    *)
        echo "Unsupported mode: ${MHLA_MODE}" >&2
        exit 2
        ;;
esac

echo "completed_at=$(date --iso-8601=seconds)" >> "${RUN_DIR}/run_metadata.txt"
echo "[${RUN_NAME}] complete"

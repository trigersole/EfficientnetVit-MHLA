#!/usr/bin/env bash
#SBATCH --job-name=effvit-mhla
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:32
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=effvit-mhla-%j.out

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/slurm/run_mhla_experiments.sbatch [options]

Core options:
  --mode MODE                 benchmark, evaluate, train, train-eval,
                              evaluate-benchmark, or train-eval-benchmark
  --experiments LIST          Comma-separated subset of e0,e1,e2,e3
  --groups LIST               Comma-separated MHLA group counts (default: 4)
  --gpus-per-experiment N     GPUs assigned to each parallel run (default: 8)
  --cpus-per-experiment N     CPUs assigned to each parallel run (default: 8)

Model/data options:
  --model NAME                Model-zoo name
  --dataset NAME              cityscapes or ade20k
  --train-data PATH           Training dataset path exported to the trainer
  --eval-path PATH            Evaluation image directory
  --crop-size N               Evaluation crop size
  --output-root PATH          Root directory for checkpoints/logs/results
  --checkpoint-template PATH  Supports {run_dir}, {run_name}, {experiment}, {groups}

Training option:
  --train-command COMMAND     Existing trainer command executed once per run.
                              It receives the MHLA_* environment variables.

Benchmark/evaluation options:
  --input-height N
  --input-width N
  --eval-batch-size N         Batch size per GPU used by upstream evaluator
  --benchmark-batch-size N
  --workers N
  --warmup N
  --iterations N
  --seed N
  --dry-run                   Print the run matrix without starting srun steps

The default SBATCH header reserves 32 GPUs and launches four 8-GPU runs.
Override SBATCH resources on submission for fewer experiments, for example:
  sbatch --gres=gpu:8 --ntasks=1 ... --experiments e2
EOF
}

MODE="benchmark"
EXPERIMENTS="e0,e1,e2,e3"
GROUP_COUNTS="4"
GPUS_PER_EXPERIMENT=8
CPUS_PER_EXPERIMENT=8
MODEL_NAME="efficientvit-seg-b1-cityscapes"
DATASET="cityscapes"
TRAIN_DATA=""
EVAL_PATH=""
CROP_SIZE=1024
OUTPUT_ROOT=""
CHECKPOINT_TEMPLATE="{run_dir}/best.pt"
TRAIN_COMMAND=""
INPUT_HEIGHT=1024
INPUT_WIDTH=2048
EVAL_BATCH_SIZE=1
BENCHMARK_BATCH_SIZE=1
WORKERS=8
BENCHMARK_WARMUP=20
BENCHMARK_ITERATIONS=100
SEED=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --groups) GROUP_COUNTS="$2"; shift 2 ;;
        --gpus-per-experiment) GPUS_PER_EXPERIMENT="$2"; shift 2 ;;
        --cpus-per-experiment) CPUS_PER_EXPERIMENT="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --train-data) TRAIN_DATA="$2"; shift 2 ;;
        --eval-path) EVAL_PATH="$2"; shift 2 ;;
        --crop-size) CROP_SIZE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --checkpoint-template) CHECKPOINT_TEMPLATE="$2"; shift 2 ;;
        --train-command) TRAIN_COMMAND="$2"; shift 2 ;;
        --input-height) INPUT_HEIGHT="$2"; shift 2 ;;
        --input-width) INPUT_WIDTH="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --benchmark-batch-size) BENCHMARK_BATCH_SIZE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --warmup) BENCHMARK_WARMUP="$2"; shift 2 ;;
        --iterations) BENCHMARK_ITERATIONS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${MODE}" in
    train|evaluate|train-eval|benchmark|evaluate-benchmark|train-eval-benchmark) ;;
    *) echo "Invalid --mode: ${MODE}" >&2; exit 2 ;;
esac

case "${DATASET}" in
    cityscapes|ade20k) ;;
    *) echo "Invalid --dataset: ${DATASET}" >&2; exit 2 ;;
esac

if [[ "${MODE}" == *train* && -z "${TRAIN_COMMAND}" ]]; then
    echo "--train-command is required for ${MODE} mode" >&2
    exit 2
fi
if [[ "${MODE}" == *evaluate* && -z "${EVAL_PATH}" ]]; then
    echo "--eval-path is required for ${MODE} mode" >&2
    exit 2
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${REPO_ROOT}/applications/efficientvit_seg/benchmark_mhla.py" ]]; then
    echo "Submit this job from the EfficientViT-MHLA repository root" >&2
    exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/slurm_runs/${SLURM_JOB_ID:-local}"
fi
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export MHLA_MODE="${MODE}"
export MHLA_REPO_ROOT="${REPO_ROOT}"
export MHLA_OUTPUT_ROOT="${OUTPUT_ROOT}"
export MHLA_MODEL_NAME="${MODEL_NAME}"
export MHLA_DATASET="${DATASET}"
export MHLA_TRAIN_DATA="${TRAIN_DATA}"
export MHLA_EVAL_PATH="${EVAL_PATH}"
export MHLA_CROP_SIZE="${CROP_SIZE}"
export MHLA_CHECKPOINT_TEMPLATE="${CHECKPOINT_TEMPLATE}"
export MHLA_TRAIN_COMMAND="${TRAIN_COMMAND}"
export MHLA_INPUT_HEIGHT="${INPUT_HEIGHT}"
export MHLA_INPUT_WIDTH="${INPUT_WIDTH}"
export MHLA_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}"
export MHLA_BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE}"
export MHLA_WORKERS="${WORKERS}"
export MHLA_BENCHMARK_WARMUP="${BENCHMARK_WARMUP}"
export MHLA_BENCHMARK_ITERATIONS="${BENCHMARK_ITERATIONS}"
export MHLA_SEED="${SEED}"
export MHLA_GPUS_PER_EXPERIMENT="${GPUS_PER_EXPERIMENT}"

IFS=',' read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
IFS=',' read -r -a GROUP_LIST <<< "${GROUP_COUNTS}"

PIDS=()
RUN_NAMES=()
PLANNED=0

terminate_steps() {
    echo "Received scheduler signal; terminating active experiment steps" >&2
    for pid in "${PIDS[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    wait || true
    exit 99
}
trap terminate_steps USR1 TERM INT

launch_run() {
    local experiment="$1"
    local groups="$2"
    local run_name="$3"
    local run_dir="${OUTPUT_ROOT}/${run_name}"
    PLANNED=$((PLANNED + 1))
    mkdir -p "${run_dir}"
    echo "Launching ${run_name}: experiment=${experiment}, groups=${groups}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  srun: gpus=${GPUS_PER_EXPERIMENT}, cpus=${CPUS_PER_EXPERIMENT}, output=${run_dir}"
        return
    fi
    srun --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_EXPERIMENT}" \
        --gres="gpu:${GPUS_PER_EXPERIMENT}" \
        --output="${run_dir}/slurm-step-%j.out" \
        --error="${run_dir}/slurm-step-%j.err" \
        bash "${REPO_ROOT}/scripts/slurm/run_mhla_variant.sh" "${experiment}" "${groups}" "${run_name}" &
    PIDS+=("$!")
    RUN_NAMES+=("${run_name}")
}

for experiment in "${EXPERIMENT_LIST[@]}"; do
    case "${experiment}" in
        e0|e1)
            launch_run "${experiment}" "4" "${experiment}"
            ;;
        e2|e3)
            for groups in "${GROUP_LIST[@]}"; do
                if ! [[ "${groups}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Invalid MHLA group count: ${groups}" >&2
                    exit 2
                fi
                launch_run "${experiment}" "${groups}" "${experiment}_m${groups}"
            done
            ;;
        *)
            echo "Invalid experiment: ${experiment}" >&2
            exit 2
            ;;
    esac
done

if [[ ${PLANNED} -eq 0 ]]; then
    echo "No experiments selected" >&2
    exit 2
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Dry run complete: ${PLANNED} experiment run(s) planned"
    exit 0
fi

FAILED=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "${RUN_NAMES[$index]} completed"
    else
        status=$?
        echo "${RUN_NAMES[$index]} failed with status ${status}" >&2
        FAILED=1
    fi
done

if [[ ${FAILED} -ne 0 ]]; then
    exit 1
fi

echo "All selected experiments completed. Results: ${OUTPUT_ROOT}"

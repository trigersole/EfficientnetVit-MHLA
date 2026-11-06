#!/bin/bash
# Run several short runs back to back inside ONE 6-hour job.
# TC2 allows only 2 queued jobs per user, so this packs pilot/screening runs together.
# Each spec is "RUN_NAME|trainer args"; each run goes through tc2_train.sbatch (resume + final eval).
#
# Usage: RUNS_ROOT=$HOME/runs/recipe_search \
#          sbatch scripts/slurm/tc2_sequence.sbatch e0 "A|--max-steps 10000" "C|--max-steps 10000 --crop-height 1024"
#SBATCH --job-name=effvit-seq
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out

set -o pipefail
EXPERIMENT="${1:?usage: tc2_sequence.sbatch <experiment> \"NAME|args\" ...}"
shift
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CHILD=""
# Forward the pre-timeout USR1 to the current run's script, which forwards it to the trainer.
trap 'echo "USR1 received $(date -Is)"; [ -n "$CHILD" ] && kill -USR1 "$CHILD" 2>/dev/null' USR1

for SPEC in "$@"; do
  NAME="${SPEC%%|*}"
  ARGS="${SPEC#*|}"
  [ "$ARGS" = "$SPEC" ] && ARGS=""
  echo "=== run $NAME ($EXPERIMENT $ARGS) start=$(date -Is)"
  # shellcheck disable=SC2086
  RUN_NAME="$NAME" bash scripts/slurm/tc2_train.sbatch "$EXPERIMENT" $ARGS &
  CHILD=$!
  wait "$CHILD"; STATUS=$?
  while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; STATUS=$?; done
  echo "=== run $NAME exit=$STATUS end=$(date -Is)"
  CHILD=""
done

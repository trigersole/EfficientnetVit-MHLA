#!/bin/bash
# Submit N chained 6-hour segments for one arm; each starts after the previous ends (any state).
# A segment that finds final.pt only evaluates (or exits), so extra links are harmless.
# Usage: bash scripts/slurm/tc2_submit_chain.sh <experiment> <num_jobs> [after_job_id]
set -euo pipefail
EXPERIMENT="${1:?usage: tc2_submit_chain.sh <experiment> <num_jobs> [after_job_id]}"
NUM_JOBS="${2:?usage: tc2_submit_chain.sh <experiment> <num_jobs> [after_job_id]}"
PREVIOUS="${3:-}"
mkdir -p logs
for _ in $(seq 1 "$NUM_JOBS"); do
  DEPENDENCY=()
  [ -n "$PREVIOUS" ] && DEPENDENCY=(--dependency="afterany:$PREVIOUS")
  PREVIOUS=$(sbatch --parsable "${DEPENDENCY[@]}" --job-name="train-$EXPERIMENT" \
    scripts/slurm/tc2_train.sbatch "$EXPERIMENT")
  echo "$PREVIOUS"
done

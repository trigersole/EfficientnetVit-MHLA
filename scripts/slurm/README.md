# SLURM launcher for E0--E3

`run_mhla_experiments.sbatch` reserves at most six hours and, by default, one
32-GPU node. It launches E0, E1, E2, and E3 concurrently with eight GPUs per
experiment. The worker records the Git commit, resolved variant, group count,
checkpoint path, visible GPUs, logs, evaluation output, and benchmark JSON/CSV.

Submit commands from the repository root. Add cluster-specific `--partition`,
`--account`, `--qos`, module loading, or virtual-environment activation as
required by your system.

Preview the expanded experiment matrix without starting any SLURM steps:

```bash
bash scripts/slurm/run_mhla_experiments.sbatch \
  --mode benchmark --experiments e0,e2 --groups 1,2,4,8 --dry-run
```

## Choose experiments

Benchmark all four variants at Cityscapes resolution:

```bash
sbatch scripts/slurm/run_mhla_experiments.sbatch \
  --mode benchmark \
  --experiments e0,e1,e2,e3 \
  --groups 4 \
  --model efficientvit-seg-b1-cityscapes
```

Benchmark only E2 and E3. Override the allocation so it requests 16 rather
than 32 GPUs:

```bash
sbatch --gres=gpu:16 --ntasks=2 \
  scripts/slurm/run_mhla_experiments.sbatch \
  --mode benchmark \
  --experiments e2,e3 \
  --groups 4
```

Run the E2 group-count ablation as four parallel jobs:

```bash
sbatch scripts/slurm/run_mhla_experiments.sbatch \
  --mode benchmark \
  --experiments e2 \
  --groups 1,2,4,8
```

## Evaluate trained checkpoints

The default checkpoint location is `{run_dir}/best.pt`. A template may contain
`{run_dir}`, `{run_name}`, `{experiment}`, and `{groups}`.

```bash
sbatch --gres=gpu:16 --ntasks=2 \
  scripts/slurm/run_mhla_experiments.sbatch \
  --mode evaluate-benchmark \
  --experiments e0,e2 \
  --groups 4 \
  --eval-path /datasets/cityscapes/leftImg8bit/val \
  --output-root /scratch/$USER/effvit-mhla \
  --checkpoint-template '/scratch/'"$USER"'/effvit-mhla/{run_name}/best.pt'
```

The evaluator reports overall mIoU and every class IoU. Benchmarking records
parameters, estimated MACs/FLOPs, latency, FPS, and peak GPU memory.

## Train and then evaluate

The official EfficientViT repository does not contain its segmentation
trainer. Therefore the launcher accepts your existing/course trainer as a
command rather than silently replacing the paper's training protocol.

The command receives these environment variables for each selected run:

```text
MHLA_EXPERIMENT             e0, e1, e2, or e3
MHLA_NUM_GROUPS             token-group count
MHLA_MODEL_NAME             model-zoo name
MHLA_DATASET                cityscapes or ade20k
MHLA_TRAIN_DATA             value of --train-data
MHLA_EVAL_PATH              value of --eval-path
MHLA_RUN_DIR                unique output directory
MHLA_CHECKPOINT_PATH        resolved checkpoint target
MHLA_GPUS_PER_EXPERIMENT    normally 8
```

Example for a trainer that accepts the project's model flags:

```bash
sbatch scripts/slurm/run_mhla_experiments.sbatch \
  --mode train-eval-benchmark \
  --experiments e0,e1,e2,e3 \
  --groups 4 \
  --train-data /datasets/cityscapes \
  --eval-path /datasets/cityscapes/leftImg8bit/val \
  --output-root /scratch/$USER/effvit-mhla \
  --train-command 'torchrun --standalone \
    --nproc_per_node="$MHLA_GPUS_PER_EXPERIMENT" path/to/course_train.py \
    --experiment "$MHLA_EXPERIMENT" \
    --mhla_num_groups "$MHLA_NUM_GROUPS" \
    --model "$MHLA_MODEL_NAME" \
    --dataset "$MHLA_DATASET" \
    --data-root "$MHLA_TRAIN_DATA" \
    --output-dir "$MHLA_RUN_DIR" \
    --resume'
```

The trainer must periodically save `{run_dir}/best.pt` (or the path supplied
with `--checkpoint-template`). Because jobs are limited to six hours, its
`--resume` behavior and periodic checkpoints should be enabled. The job receives
`USR1` five minutes before termination and stops active `srun` steps cleanly.

For final scientific comparisons, keep every non-attention training option
identical across E0--E3: data, augmentations, loss, initialization policy,
optimizer, schedule, iterations, effective global batch size, and seed.

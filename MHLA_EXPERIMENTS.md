# EfficientViT + MHLA segmentation experiments

This branch adds a controlled E0--E3 experiment to the official
[MIT Han Lab EfficientViT](https://github.com/mit-han-lab/efficientvit) codebase.
The baseline is upstream commit `de7d7733cc0329f391b33f1f459271562ec27bd5`.
The MHLA math and locality-biased mixing initialization were adapted from the
authors' [DAGroup-PKU/MHLA](https://github.com/DAGroup-PKU/MHLA) implementation
at commit `275352cad3a38a51414afe45e91e2c9641aa485e`.

No accuracy improvement is claimed here. The implementation makes the proposed
comparison runnable; mIoU must be obtained by training and evaluating all four
variants under one fixed protocol.

## Controlled variants

| ID | Attention | Original multi-scale QKV | Constructed path |
|---|---|---:|---|
| E0 | LiteMLA | yes | QKV -> base + 5x5 aggregated QKV -> ReLU linear attention -> projection |
| E1 | LiteMLA | no | QKV -> ReLU linear attention -> projection |
| E2 | MHLA | no | QKV -> spatial token groups -> mixed KV summaries -> projection |
| E3 | MHLA | yes | QKV -> base + 5x5 aggregated QKV -> spatial token groups -> mixed KV summaries -> projection |

When multi-scale QKV is disabled, `LiteMLA.aggreg` is an empty `ModuleList` and
the projection input width is reduced accordingly. The convolutional branches
are therefore not constructed or executed in E1/E2.

## MHLA adaptation and tensor shapes

For one LiteMLA block, let the input be `[B, C, H, W]`, head dimension be `D`,
and EfficientViT feature-head count be `A`. The 1x1 projection produces
`[B, 3*A*D, H, W]`. With `S` original aggregation branches, E0/E3 concatenate
the base and aggregated tensors to obtain `A' = A*(1+S)` heads; E1/E2 use
`A' = A`.

MHLA reshapes Q/K/V as follows:

```text
[B, A', D, H, W]
    -> [B, A', N, D], N = H*W
    -> [B, A', M, Nb, D] for equal-sized 2-D token groups
```

It then computes:

```text
group KV summaries: [B, A', M, D, D]
group K sums:       [B, A', M, D]
mixing matrix:      [M, M] (learnable, shared over A')
output:             [B, A', D, H, W]
```

Groups are rectangular regions on the feature map, not chunks of raw image
pixels or flattened arbitrary windows. Uneven spatial sizes use a safe fallback
that preserves row-major spatial order. `M=1` is tested against the original
normalized linear-attention equation.

The optional LePE depthwise convolution found in the MHLA image-classification
code is deliberately omitted. Adding it to E2 would violate the isolation test
by introducing a new local convolution after QKV.

## Constructing models

```python
from efficientvit.seg_model_zoo import create_efficientvit_seg_experiment_model

model = create_efficientvit_seg_experiment_model(
    name="efficientvit-seg-b1-cityscapes",
    experiment="e2",
    mhla_num_groups=4,
    pretrained=False,
)
```

The equivalent low-level flags are `attention_type="lite_mla" | "mhla"`,
`use_multiscale_qkv=True | False`, and `mhla_num_groups=M`. The four model-only
configuration records live in `applications/efficientvit_seg/configs/mhla/`.

## Environment and sanity tests

Install the official requirements, then run:

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_mhla -v
```

The tests cover E0--E3 block and full-model forwards, output shapes, backward
gradients into QKV and MHLA mixing weights, finite outputs/gradients, arbitrary
batch/spatial sizes, M=1 equivalence, and CUDA fp16 autocast when CUDA exists.

## Parameter, MAC, latency, FPS, and memory measurement

Use the dataset's real evaluation resolution for reportable numbers:

```powershell
python applications/efficientvit_seg/benchmark_mhla.py `
  --model efficientvit-seg-b1-cityscapes `
  --input_size 1024 2048 `
  --mhla_num_groups 4 `
  --warmup 50 --iterations 200 `
  --output results/b1_cityscapes_1024x2048.json
```

JSON and CSV are both written. MAC counts from `torchprofile` are estimates; the
JSON's `macs_notes` field records unsupported operators, so report them as MAC
estimates rather than exact hardware FLOPs. Latency, FPS, and peak allocation are
measured directly on the selected device. Do not compare the included smoke
benchmark to published full-resolution numbers.

### Measured engineering baseline

The following is an inference-only measurement from this working tree on an
NVIDIA GeForce RTX 3050 Laptop GPU, PyTorch 2.5.1/CUDA 12.1, batch 1,
1024x2048 input, 5 warmups, and 20 timed iterations. MACs are profiler estimates
with the unsupported operations recorded in the result JSON. These are not
accuracy results and are not a substitute for repeated benchmark runs.

| Model | Params | Estimated MACs | FLOPs (2/MAC) | Latency | FPS | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 4,803,443 | 25.355G | 50.709G | 70.05 ms | 14.28 | 354,993,152 B |
| E1 | 4,318,963 | 23.592G | 47.184G | 56.82 ms | 17.60 | 353,055,232 B |
| E2, M=4 | 4,319,075 | 23.587G | 47.174G | 57.37 ms | 17.43 | 353,058,816 B |
| E3, M=4 | 4,803,555 | 25.345G | 50.690G | 70.02 ms | 14.28 | 354,996,736 B |

The raw data is in `results/b1_cityscapes_1024x2048.json` and `.csv`. On this
eager PyTorch implementation E2 is slightly slower than E1, while E3 and E0
are effectively tied in this short run. MHLA adds only 112 parameters to this
model. Kernel/runtime optimization remains separate from the accuracy hypothesis.

## Checkpoint initialization

E0 loads an original checkpoint strictly. E1--E3 use shape-compatible loading:
all unchanged backbone/head parameters are restored; removed aggregation or
resized projection tensors are reported in `model.pretrained_load_report`, and
new MHLA mixing matrices retain their locality-biased initialization. Save this
report with every run.

This was verified with the official B0 Cityscapes checkpoint: E0 loaded all 230
tensors strictly and produced bit-identical logits to the original model factory.
E1/E2 loaded 218 compatible tensors (four projection shape mismatches and eight
removed aggregation tensors); E3 loaded all 230 original tensors and reported
only its four new MHLA mixing matrices as missing.

## Training

The official repository does not release a semantic-segmentation training
script or its training configuration; it contains evaluation/demo code and
pretrained checkpoints only. Consequently this branch does not fabricate a
different trainer and call it the paper baseline. Connect the model factory
above to the course-provided/original training harness and keep dataset,
resolution, augmentations, loss, AdamW settings, cosine schedule, iterations,
seed, and evaluation protocol identical across E0--E3.

For each variant, the only model arguments that should change are:

```text
E0: experiment=e0
E1: experiment=e1
E2: experiment=e2 mhla_num_groups=4
E3: experiment=e3 mhla_num_groups=4
```

Run E0 first and require it to reproduce the accepted baseline tolerance before
interpreting E1--E3. Log the resolved configuration, git commit, seed, compatible
checkpoint-load report, loss, validation mIoU, per-class IoU, best epoch,
parameters, MAC estimate, latency, FPS, peak memory, and checkpoint path.

## Evaluation

After training, evaluate each checkpoint with the original evaluator plus the
variant selector. It now prints mIoU and every class IoU.

```powershell
python applications/efficientvit_seg/eval_efficientvit_seg_model.py `
  --path D:\datasets\cityscapes\leftImg8bit\val `
  --dataset cityscapes --crop_size 1024 `
  --model efficientvit-seg-b1-cityscapes `
  --experiment e2 --mhla_num_groups 4 `
  --weight_url checkpoints\e2_best.pt
```

Use the same command for E0/E1/E3, changing only `--experiment` and checkpoint.
For the group-count ablation, keep E2 fixed and run `M in {1, 2, 4, 8}`.

## Results template and hypothesis decisions

| Model | Params | MACs | Latency | FPS | Peak memory | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| E0 | | | | | | |
| E1 | | | | | | |
| E2, M=4 | | | | | | |
| E3, M=4 | | | | | | |

- H1 is supported if E1 < E0 under matched training.
- H2 is supported if E2 > E1; E2 approximately E0 would support replacement.
- H3 is supported if E3 > E2 beyond seed-to-seed variation; E3 approximately
  E2 would indicate that multi-scale QKV is largely redundant after MHLA.

Report multiple seeds for final E0/E2/E3 conclusions and inspect Cityscapes
small/local classes (person, rider, pole, traffic sign, bicycle) separately from
large/contextual classes (road, building, car).

## Modified files

- `efficientvit/models/nn/mhla.py`: standalone token-group MHLA operator.
- `efficientvit/models/nn/ops.py`: controlled LiteMLA routing and true removal
  of aggregation branches.
- `efficientvit/models/efficientvit/backbone.py`: propagates experiment flags.
- `efficientvit/seg_model_zoo.py`: E0--E3 factory and compatible checkpoint load
  reporting.
- `applications/efficientvit_seg/eval_efficientvit_seg_model.py`: experiment
  flags plus per-class IoU output.
- `applications/efficientvit_seg/benchmark_mhla.py`: reproducible profiling.
- `tests/test_mhla.py`: numerical, gradient, shape, full-model, and AMP tests.
- `efficientvit/models/nn/norm.py` and package `__init__.py` files: portable
  fallbacks so segmentation can run without Triton and unrelated optional apps.

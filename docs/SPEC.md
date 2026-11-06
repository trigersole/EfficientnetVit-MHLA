# SPEC: MHLA Rewrite, MALA Arms, 1-GPU Training, TensorRT Deployment

- **Status:** draft for team review
- **Date:** 2026-10-30 (aligned with PRD on 2026-11-01)
- **Goals, scope and why:** [`PRD.md`](PRD.md)
- **Existing study notes:** [`../MHLA_EXPERIMENTS.md`](../MHLA_EXPERIMENTS.md)

---

## 1. Notation

- **Image tokens:** one feature map position = one token. `N = H × W` tokens.
- **Heads:** `A'` attention heads per block, each with head size `D = 16` (B1).
- **Per head, after ReLU:** query `q_i`, key `k_j` (both length `D`, ≥ 0), value `v_j` (length `D`).
- **`ε`:** tiny number to avoid divide-by-zero (`1e-15`, kept in fp32).
- **`s`:** the scale `D^-1/2` (= 0.25 for B1).
- **Groups (MHLA):** `M` rectangles of equal size, `n = N / M` tokens each. `g(i)` = the group of token `i`.

## 2. The math

### 2.1 LiteMLA (today's EfficientViT attention)

- `S = Σ_j v_j k_jᵀ` (D×D summary), `z = Σ_j k_j` (key sum)
- `num_i = S q_i`, `den_i = zᵀ q_i`
- `out_i = num_i / (den_i + ε)`
- **Trick already in the code:** add a row of 1s to v (`v' = [v; 1]`).
  - Then `S' = Σ_j v'_j k_jᵀ` is (D+1)×D, and `S' q_i = [num_i; den_i]` in **one** matmul.

### 2.2 MHLA (2026)

- One summary per group: `S'_m = Σ_{j ∈ group m} v'_j k_jᵀ`.
- Mix the summaries: `S̃'_m = Σ_l W[m, l] · S'_l`.
  - `W` is a learnable M×M matrix.
  - It starts from a "nearby groups count more" cosine pattern, with columns summing to 1.
- Each query reads its own group's mixed summary: `[num_i; den_i] = S̃'_{g(i)} q_i`, then `out_i = num_i / (den_i + ε)`.
- With `M = 1` and `W = [1]` this is exactly LiteMLA (already unit-tested).

### 2.3 MALA (ICCV 2025), in the official-code form

- **Official code** (`qhfan/MALA`, `classfication_release/model.py`, lines 131–138):
  - `z = s·q·mean(k)`, `kv = (s/N)·Σ kᵀv`
  - `res = q·kv·(1 + 1/z) − z·mean(v)`
- **Same thing, rewritten:** `res_i = num_i/den_i + (s/N)·(num_i − den_i·v̄)`, where `v̄ = mean(v)`.
  - In words: **old output + a "magnitude" correction term**.
- **Our version adds a dial (gate):**
  - `out_i = num_i/(den_i + ε) + g_h · (s/N)·(num_i − den_i·v̄)`
  - `g_h` is one learnable number per head, **starting at 0**.
    - At step 0, E4 and E5 compute exactly what E1 and E2 do, so the pretrained backbone is not disturbed.
  - `g_h = 1` is exactly official MALA.
  - The learned `g_h` values are reported.
- **With MHLA (E5):** use the mixed group sums instead of global sums.
  - `N` → `Ñ_{g(i)}` (mixed token count)
  - `v̄` → `Σ̃v_{g(i)} / Ñ_{g(i)}` (mixed value mean)
  - `Ñ` is clamped to ≥ 1 for safety.
- **Kept from MALA:** the normalization only.
- **Not ported (on purpose):**
  - The ELU+1 kernel. We keep ReLU, which is pretrained and which the MALA paper's ablation shows is comparable.
  - The 5×5 LePE conv. It would add a local conv and break the isolation design.
  - RoPE and the output gate. They are not part of the attention formula.

### 2.4 One matmul gives every sum (used by E4 and E5)

- Add a row of 1s to **both** v and k: `v' = [v; 1]`, `k' = [k; 1]`.
- `P_m = Σ_{j ∈ group m} v'_j k'_jᵀ` is (D+1)×(D+1) and holds:
  - top-left `Σ v kᵀ`
  - top-right `Σ v`
  - bottom-left `Σ kᵀ`
  - bottom-right `n_m` (token count)
- Mixing `W` acts on all four parts at once (it's linear).
- `P̃[:, :D] · q_i = [num_i; den_i]`; `Σ̃v = P̃[:D, D]`; `Ñ = P̃[D, D]`.

### 2.5 Why MHLA and MALA (technical rationale)

**Two failure modes of ReLU linear attention.** Write `s_ij = φ(q_i)·φ(k_j)`; the weights are `a_ij = s_ij / Σ_l s_il`. In this section `σ` is the scale `D^-1/2` (called `s` in §1), so it doesn't clash with the score `s_ij`.

- **F1: magnitude neglect, so attention can't be sharp.**
  - ReLU is positively homogeneous: `φ(c·q) = c·φ(q)` for `c > 0`.
  - So `a_ij(c·q_i) = a_ij(q_i)`. The weights are degree-0 in `|q_i|`: only the query's direction matters, not its norm.
  - In softmax, `exp(c·s_ij)` makes `c` an inverse temperature, so a larger `|q|` gives a sharper distribution. Linear attention has no such control.
  - Contrast is also only polynomial (score ratios between non-negative vectors), not exponential, so the maps come out flat.
  - EfficientViT's paper states it directly: ReLU linear attention "cannot produce sharp attention distributions".
- **F2: one shared summary (global context collapse) and no locality.**
  - Every query reads the same `S = Σ_j v_j k_jᵀ` (D×D, D = 16) and the same `z`.
  - So every output `o_i = S·φ(q_i) / (z·φ(q_i))` lies in the column space of `S`, of rank ≤ D. That holds for N = 2,048–8,192 tokens.
  - The MHLA paper calls this global context collapse: the information exceeds the d×d capacity, and attention becomes more uniform.
  - The weights depend on content only, so a token 50 positions away is treated like a neighbour. EfficientViT's paper: "weak at capturing local information".
- **EfficientViT's answer: multi-scale QKV.**
  - A 5×5 depthwise conv plus a grouped 1×1 conv inject locality into Q/K/V before attention. This partly addresses F2's locality.
  - It costs +13.2 ms (+23%) per 1024×2048 image in our eager benchmark (E0 vs E1).
  - It does **not** address F1: E0's weights are still degree-0 in `|q|`.

**What each method changes**
- **MHLA addresses F2.**
  - M group summaries are mixed per query group by `W` (M×M).
  - Queries in different groups read different summaries `S̃_g`. Across the image the output can span up to M·D dimensions instead of D, and context depends on the query's position.
  - The cosine-initialized `W` (own 0.29 > side 0.25 > corner 0.21 for M = 4) is a learned locality prior **without convolution**. That makes MHLA a functional substitute for the multi-scale branch (H2: E2 vs E1; replacement: E2 vs E0).
  - Cost:
    - O(N·D² + M²·D²) FLOPs, negligible at M = 4.
    - M² parameters per block, so 112 in B1 (7 blocks × 16).
- **MALA addresses F1.**
  - `a_ij = s_ij / Σ_l s_il + g·(σ/N)·(s_ij − s̄_i)`, with `s̄_i` the mean score of query `i`.
  - The second term is degree-1 in `|q_i|`, so a larger query norm increases contrast. Keys scoring below the mean can get negative weight.
  - `Σ_j a_ij = 1` still holds, because the correction terms sum to zero.
  - It is orthogonal to the multi-scale branch, which leaves F1 untouched. So H4 (E4 vs E1) tests a weakness the original model never fixed; it is not a re-test of H1.
  - Cost: `Σv` and `N` come from the K-pad GEMM (§2.4), plus elementwise ops.
  - Parameters: 0 in the official form; 88 in our gated form (one `g` per head: 3 blocks × 8 heads + 4 blocks × 16 heads).

**Why they compose (E5)**
- MHLA changes *which* statistics a query uses: per-group mixed `(Σv kᵀ, Σk, Σv, N)`.
- MALA changes *how* the weights are normalized given those statistics. It needs exactly those four sums, which the same (D+1)×(D+1) GEMM supplies (§2.4).
- So E5 is MALA applied to MHLA's mixed statistics, with no extra structure.
- **Nested models:** M = 1 and `g = 0` both reduce to E1 exactly.
  - So E1 → E2, E1 → E4 and E1 → E5 add one mechanism each, starting from an identical function at step 0.
  - Under matched training, an mIoU difference can be attributed to the added mechanism (within seed noise; see PRD §9).

**Selection criteria.** Every one must hold:
1. It fixes a documented weakness of ReLU linear attention (F1 or F2).
2. **Exact start:** a neutral setting reproduces LiteMLA, so the ImageNet-pretrained backbone keeps its function at step 0.
3. ≤ ~100 extra parameters and **no convolution**. This keeps the comparison isolated from the multi-scale branch.
4. ONNX → TensorRT-exportable with MatMul and elementwise ops, compatible with the fp32 attention island (§9.1).
5. Recent (2025–2026) and published at an allowed venue.

| Method | Fixes | Exact start | Extra params (B1) | Conv-free | TRT-friendly | Venue |
|---|---|---|---|---|---|---|
| **MHLA** | F2 | M = 1 | 112 | yes | yes (after the §3.1 rewrite) | ICLR 2026 |
| **MALA** | F1 | g = 0 | 88 (0 official) | yes | yes | ICCV 2025 |

**Evidence of strength.** The MHLA paper's Table 2(a), ImageNet-1K top-1, DeiT-T:
- Linear 69.8, softmax 72.2, FLatten 74.1, InLine 74.5, MALA 75.1, MHLA 75.8.
- Caveat: the rows there use each method's full module. For example, the MALA row has 6.3M params vs 5.7M for linear, because MAViT's extra components are included. We port only the normalization.

**Runners-up rejected:**
- **RALA:** it targets rank, like MHLA, so it overlaps with F2 and would make the 2×2 less informative.
- **FLatten focus power:** it also targets F1, but it's from 2023 and the `Pow` op carries a high fp16 overflow risk.
- **PolaFormer, MirrorLA:** no exact start from the checkpoint. A Householder reflection is never the identity, and there is a sign split.
- **InLine, Agent attention:** they need a local conv or pooling, which breaks isolation from the multi-scale branch.
- **Delta-rule and recurrent mixers (Gated DeltaNet, LION, ViG):** they don't export to TensorRT without custom kernels.

**Falsifiable predictions** (reported whatever the outcome):
- **P1 (MHLA → F2):**
  - E2 > E1, with the gain concentrated in small or thin classes (pole, traffic sign, person, rider, bicycle).
  - Learned `W` becomes more diagonal than at init (`W_own − W_corner` grows).
- **P2 (MALA → F1):**
  - E4 > E1.
  - Learned `g_h` > 0 in most heads. If `g` → 0, the model rejected the mechanism; report that.
- **P3 (composition):** E5 ≥ max(E2, E4) if F1 and F2 are independent. E5 ≈ max(E2, E4) means they overlap.
- **P4 (replacement):** E2 or E5 within 0.5 mIoU of E0, while faster in TensorRT (S1). That means MHLA (+MALA) can replace the multi-scale branch.
- **Optional analysis:** mean attention entropy per block for E1 vs E4, on 64 sampled queries × 20 val images. A lower entropy is direct evidence for sharper attention (F1).

## 3. Model changes

### 3.1 `efficientvit/models/nn/mhla.py`: matmul-only MHLA

- **Goal:** same numbers, no einsum, no run-time index lookups, so TensorRT is happy.
- **Equal-group path** (used whenever H and W divide by the group grid; true for all our sizes):
  - Fold batch and heads: `[B·A', 3D, H, W]`.
  - View as `[B·A', 3D, gh, a, gw, c]` (6-D), permute, then reshape to `[B·A', M, 3D, n]`.
  - Split into q, k, v, each `[B·A', M, D, n]`.
  - Pad v with a ones row (and k too when MALA is on).
  - Group summaries: `P = v' @ k'ᵀ` → `[B·A', M, D+1, D(+1)]`.
  - Mixing: flatten the last two dims, then **one matmul** with `W` over the M axis.
  - Apply: `P̃[..., :D] @ q` → `[num; den]`, then divide (plus the MALA term if on).
  - Reverse permute/reshape → `[B, A'·D, H, W]`.
- **Unequal-group path** (odd sizes; PyTorch eval only):
  - Uses a fixed one-hot "token → group" matrix and plain matmuls.
  - No `nonzero`, `index_copy_` or Python loop.
  - About M× the equal path's cost. That's fine because it's rare.
- **Remove:** the `_spatial_group_ids` call on the equal path.
- **Keep:**
  - The parameter name `mixing_weight` and its cosine init (state_dicts stay compatible).
  - The group layout: longer group-grid axis along the longer image axis.
- **Interface:** `forward(q, k, v, mala_gate=None)`.
  - Input shapes as today: `[B, A', D, H, W]`.
  - `mala_gate` has shape `[A']` or is `None`.

### 3.2 `efficientvit/models/nn/ops.py`: MALA option in `LiteMLA`

- New argument `attention_norm: str = "div"`; allowed values `"div"`, `"mala"`.
- If `"mala"`: create `self.mala_gate = nn.Parameter(torch.zeros(A'))`.
- **`relu_linear_att` (E0/E1):**
  - **Unchanged** for `"div"`, so E0 stays bit-identical to the official model.
  - For `"mala"` (E4), call the shared MHLA helper with `M = 1`.
- **`mhla_att` (E2/E3/E5):** pass `self.mala_gate` (or `None`) to MHLA.
- **Precision:** all attention math stays in fp32. Keep the existing `.float()` casts and `autocast(enabled=False)`.

### 3.3 Plumbing

- `EfficientViTBlock` and `efficientvit/models/efficientvit/backbone.py`: pass `attention_norm` through, the same way as `attention_type`.
- `efficientvit/seg_model_zoo.py`:
  - Add to `EFFICIENTVIT_SEG_EXPERIMENTS`:
    - `"e4": {"attention_type": "lite_mla", "use_multiscale_qkv": False, "attention_norm": "mala"}`
    - `"e5": {"attention_type": "mhla", "use_multiscale_qkv": False, "attention_norm": "mala"}`
  - Strict loading only when `lite_mla` + multi-scale + `div` (i.e. E0).
- Configs: add `applications/efficientvit_seg/configs/mhla/e4_mala.yaml` and `e5_mhla_mala.yaml`.
- Allow `e0`–`e5` in:
  - `train_efficientvit_seg_model.py`
  - `eval_efficientvit_seg_model.py`
  - `benchmark_mhla.py`

## 4. Initialization (ImageNet backbone for every arm)

- **Checkpoint:** `efficientvit_b1_r224.pt` (ImageNet classification, from the official HuggingFace page).
- **New shared helper** in `seg_model_zoo.py`:
  - `load_compatible_state_dict(model, state, key_prefix=None, inherit_sliced_proj=True) -> report`
- **Backbone only:** the trainer passes `key_prefix="backbone."`, so classifier-head weights can't leak into the seg head.
- **Sliced projection (fairness fix):**
  - In no-add-on arms (E1, E2, E4, E5), each attention block's output projection takes half as many inputs.
  - Today those weights get thrown away and start random.
  - Instead, copy the matching half: `proj.conv.weight[:, :A·D]`.
  - This half belongs to the base (non-add-on) heads, because the concat order is `[base heads, add-on heads]`.
- **Report:** `initialization_report.json` records `loaded`, `sliced`, `missing`, `unexpected` and `shape_mismatch`.
- **Trainer flags:**
  - `--init-scope {backbone,all}`, default `backbone`. Use `all` for a seg checkpoint.
  - `--no-inherit-sliced-proj`, to turn the slicing off.
- The existing `load_compatible_weights` in the trainer is replaced by the helper (one code path).

## 5. Training protocol (identical for E0–E5)

| Setting | Value |
|---|---|
| Model | `efficientvit-seg-b1-cityscapes` (or `-ade20k` fallback) |
| Init | ImageNet B1 r224, backbone only, sliced proj on |
| Train crop | Cityscapes 512×1024 · ADE20K 512×512 |
| Augment | random scale 0.5–2.0, random crop, horizontal flip |
| Global batch | 16 = per-GPU batch × grad-accum (per-GPU batch from the TC2 memory probe: 8 or 4) |
| Optimizer | AdamW, lr 6e-4, weight decay 0.01 (none on norms/biases), grad clip 1.0 |
| Schedule | 500 warmup steps, then cosine down to 1% of lr |
| Precision | AMP for convs; attention always fp32 |
| Steps | `min(40000, ⌊12 h × 3600 × imgs_per_s / 16⌋)`, rounded down to 1000. `imgs_per_s` is measured on TC2 (on the slowest TC2 GPU model, if nodes differ) |
| Seed | 0, one seed per arm |
| Periodic val | every 2000 steps; Cityscapes all 500 val images (ADE20K: first 500), resized to the train crop |
| Reported checkpoint | `final.pt` (last step), **not** best-on-val, to avoid picking on the val set |

- **One source of truth:** `scripts/protocol.env` holds every value above. The TC2 runner and the optional local runner both `source` it.
- **New trainer outputs:**
  - `final.pt`: evaluator-compatible weights, written at `max_steps`.
  - `metrics.jsonl`, every log interval: `imgs_per_s`, `data_wait_frac`, `peak_mem_gb`.
  - `metrics.jsonl`, every eval: per block, the `mala_gate` values and `mixing_weight` min/max (early NaN warning).
- **Kept as-is:** `last.pt` (full resume state), `best.pt`, SIGUSR1/SIGTERM → save, then exit.

## 6. Baseline reference R0 (anti-penalty fix)

- **What:** the official released checkpoint, evaluated by **our** evaluator.
  - Cityscapes: `efficientvit_seg_b1_cityscapes.pt` (published 80.547)
  - ADE20K: `efficientvit_seg_b1_ade20k.pt` (published 42.840)
- **How:** `eval_efficientvit_seg_model.py --experiment e0`, with official weights, full val, official eval resolution.
- **Pass rule:** within 0.3 mIoU of the published number (Cityscapes 80.2–80.8, ADE20K 42.5–43.1). If not, our eval pipeline is wrong; fix it before trusting E0–E5.
- **Evaluation only:** R0 is not exported or timed; deployment covers E0–E5.
- **In the report:**
  - R0 is the first row of the main table, labelled "official weights, authors' unreleased recipe".
  - One bullet explains the gap (shorter recipe, 1 GPU, same recipe for every arm).
  - No tuning is done for any single arm.

## 7. Evaluation protocol

- **Tool:** `applications/efficientvit_seg/eval_efficientvit_seg_model.py` (official evaluator plus the variant flags).
- **Cityscapes:** all 500 val images at 1024×2048 (`--crop_size 1024`).
- **ADE20K:** all 2000 val images with the evaluator's default resize.
- **Outputs per arm:** mIoU and all per-class IoUs (already printed), saved to `<run_dir>/eval.txt`.
- **Class groups (Cityscapes):**
  - small = person, rider, pole, traffic sign, bicycle
  - large = road, building, car
  - Report the mean IoU of each group.

## 8. Compute runners

### 8.1 Cluster (TC2): `scripts/slurm/tc2_train.sbatch`

- **Header:**
  - `--partition=MGPU-TC2 --qos=normal --nodes=1 --gres=gpu:1`
  - `--cpus-per-task=10 --mem=30G --time=06:00:00 --signal=B:USR1@300`
- **Body:**
  - `module load anaconda`, activate the env, `source scripts/protocol.env`.
  - If `final.pt` exists, skip training and only evaluate (→ `eval.txt`).
  - Otherwise start the trainer in the background with `--resume`.
    - `trap` forwards USR1 to it; then `wait`.
    - When it finishes, evaluate.
- **Workers:** 8 data-loader workers.
- **`scripts/slurm/tc2_submit_chain.sh <exp> <num_jobs>`:**
  - Submits `num_jobs` copies, each `--dependency=afterany:<previous>`.
  - Prints the job IDs.
- **Leave alone:** the old 32-GPU launcher. Add a TC2 section to `scripts/slurm/README.md`.

### 8.2 Local, optional: `scripts/local/train_local.sh <exp>`

- For any team member's own GPU. **Coding and quick tests only.** Reported numbers come from TC2.
- Same trainer, same `protocol.env`, `--resume`, no time limit, then eval.
- Workers = `min(cores − 2, 8)`.

### 8.3 Probes

- **Smoke test:** `--max-steps 200`, crop 256×512, every arm, on TC2.
- **Throughput probe:**
  - `scripts/probe_throughput.py <run_dir>` reads `metrics.jsonl` from a 300-step run.
  - It averages `imgs_per_s` over the last 200 steps and prints the `max_steps` from §5.
- **Memory probe:** try per-GPU batch 8 on the TC2 GPU; if it runs out of memory, use 4 (with accum 4).

## 9. Deployment pipeline (ONNX → TensorRT, on TC2)

### 9.0 Where it runs

- **All TensorRT work runs on TC2**, the one machine the whole team shares.
- **One job does all the timing:** `scripts/slurm/tc2_trt.sbatch` builds and times E0–E5 inside a single job, so every arm is timed on the same GPU.
  - Engines only work on the GPU they were built on, so they are built inside the job.
  - The parity check (§9.4) runs in a TC2 job too, with engines built there.
- **Install without sudo:**
  - `trtexec` comes from the TensorRT tar package, unpacked into `$HOME`.
  - The `tensorrt` pip wheel goes into the conda env (Python API, used by `trt_parity.py`).
  - The tar and the wheel must be the same TensorRT version.
- **Check in the first probe job:**
  - GPU model and driver (`nvidia-smi`).
  - That the chosen TensorRT version supports that GPU and driver. If not, pick an older TensorRT version.
- **ONNX export** (§9.1) can run anywhere; the `.onnx` file is portable.

### 9.1 Export: `applications/efficientvit_seg/export_onnx_experiment.py`

- **Flags:**
  - `--model`, `--experiment e0..e5`, `--mhla-num-groups 4`
  - `--weights PATH` (optional; random weights are fine for latency)
  - `--input-size H W` (default 1024 2048; ADE20K 512 512)
  - `--precision {fp32,fp16-mixed}`, `--output`, `--opset 17`
- **Export settings:** batch 1, static shape, `torch.onnx.export(..., dynamo=False)`, then onnxsim.
- **fp16-mixed:**
  - Export `model.half().cuda()` with a half input.
  - The existing `.float()` / `.to(dtype)` casts become **Cast nodes**, so the attention core stays fp32 inside the file.
- **Graph audit:** fail if any of `Einsum`, `NonZero`, `Loop`, `If`, `ScatterND` or `ScatterElements` appear.
- **ORT parity (fp32):** ONNX Runtime vs PyTorch on one random input. Max abs diff must be < 1e-3.
- **Why not `assets/onnx_export.py`:** it ignores E1–E5, uses opset 11 and defaults to batch 16. Leave it untouched.

### 9.2 Build and benchmark: `scripts/trt/bench_trt.sh <onnx> <tag>`

- Detect the TensorRT major version.
- **Build** (once per model and precision):
  - `trtexec --onnx=… --saveEngine=… --profilingVerbosity=detailed --skipInference`
  - Add `--stronglyTyped` on TRT 10. TRT 11 is always strongly typed.
  - Precision therefore always comes from the ONNX file: one method for both versions.
- **Benchmark:**
  - `trtexec --loadEngine=… --warmUp=2000 --duration=0 --iterations=1000 --exportTimes=…`
  - TRT 10 also gets `--useCudaGraph --noDataTransfers --useSpinWait` (TRT 11 has these on by default).
- **Second run with data transfer included:** TRT 11 `--includeDataTransfers`; TRT 10 drop `--noDataTransfers`. This matches upstream's protocol.
- **Results:**
  - Median, p90 and p99 GPU compute time, written to `results/trt/<tag>.json`.
  - Also record GPU, driver, CUDA and TRT versions.
- **Fair-run rules:**
  - Run E0–E5 interleaved, 3 repeats.
  - The job has the GPU to itself (`--gres=gpu:1`).
  - Other users' jobs may share the node's CPUs, so the main metric is **GPU compute time**. The run that includes data transfers is secondary.
  - Clock locking needs sudo, so it isn't possible on TC2. Instead, record the GPU clocks and node name in the JSON, and rely on interleaving plus 3 repeats.

### 9.3 Where time goes: `scripts/trt/layer_breakdown.py` (should-have)

- Input: trtexec `--exportProfile` JSON.
- Groups layer times by name:
  - `context_module` (attention)
  - `local_module` (MBConv)
  - `head`
  - other
- Output: CSV plus a stacked bar chart, E0 vs E2.

### 9.4 Accuracy after deployment: `scripts/trt/trt_parity.py`

- **Inputs:** a trained arm's engine (fp16-mixed) and its PyTorch fp32 model, on the first 20 val images.
- **Metrics:**
  - % of pixels with the same predicted class (**pass ≥ 99.5%**)
  - max abs logit diff
- **Runs for:** E0–E5.

## 10. Results and figures

- **`scripts/collect_results.py`:**
  - Reads `eval.txt`, `metrics.jsonl`, the benchmark JSON and `results/trt/*.json`.
  - Writes `results/summary.csv` and `results/summary.md`.
- **Main table columns:**
  - ID, params, MACs
  - TRT fp32 ms, TRT fp16-mixed ms
  - mIoU, small-class mIoU, large-class mIoU
  - steps, GPU used
- **Figures** (`scripts/make_figures.py` → `results/figures/`):
  1. **Must have:** speed vs accuracy: TRT fp16-mixed latency vs mIoU, E0–E5, with R0's mIoU marked as a reference line. **The main take-away chart.**
  2. Should have: layer-time breakdown, E0 vs E2.
  3. Should have: learned MALA gate values per block (E4, E5).
  4. Should have: learned MHLA mixing matrices (E2, E5) as heatmaps.
- **Verdicts:** H1–H5 and S1 as in `PRD.md` §9, with the 0.5 mIoU "inconclusive" rule.

## 11. Survey document

- `docs/RESEARCH_SURVEY.md` is the source for the report's related-work section. It holds:
  - The verified table of about 15 linear-attention methods: venue, link, idea, TensorRT-friendly?, start-exact-from-checkpoint?, extra params, effort.
  - Why MALA was picked.
  - Runners-up (RALA, FLatten focus, PolaFormer, MirrorLA) and why they were not trained.

## 12. Tests and acceptance criteria

- **Unit tests** (`tests/test_mhla.py`; CPU, run locally in a `uv` venv):
  1. New MHLA = naive per-group loop reference, for M ∈ {1, 2, 4, 8}, sizes even and odd. Tolerance 1e-5.
  2. MHLA with M = 1 = LiteMLA.
  3. E4 with gate 0 = E1, and E5 with gate 0 = E2 (same weights, exact to 1e-6).
  4. Gate 1 = a transcription of official MALA lines 131–138 (ReLU kernel, no LePE).
  5. MALA + MHLA with M = 1 = MALA + LiteMLA.
  6. Gradients reach `mixing_weight` and `mala_gate` and are finite.
  7. E0–E5 full-model forward and backward work, with no NaN.
  8. Init: backbone-only loading never touches `head.*`; sliced proj copies the right columns; the report counts are correct.
  9. ONNX export of E0–E5 has no banned ops. Skipped if `onnx` is missing.
     - The CUDA-only check (attention MatMuls typed fp32 in the fp16-mixed file) is skipped on CPU.
- **Existing suites still pass:** `python -m unittest tests.test_mhla tests.test_seg_trainer -v`.
- **Trainer checks:**
  - `final.pt` loads in the evaluator.
  - Resume continues at the exact next step.
- **End-to-end acceptance:**

| Component | Done when |
|---|---|
| Model changes | Tests 1–7 pass; E0 is still bit-identical to the official model factory |
| Init | Test 8 passes; `initialization_report.json` shows sliced proj tensors for E1/E2/E4/E5 |
| TC2 runner | One real job: USR1 → `last.pt` saved → chained job resumes at the right step |
| Smoke runs | 200-step run on TC2 for every arm, finite loss |
| R0 | Within 0.3 mIoU of the published number |
| Export | Every arm exports in both precisions; audit clean; ORT diff < 1e-3 |
| TRT | Latency JSON for E0–E5 × 2 precisions × 3 repeats, all from one TC2 job |
| Parity | ≥ 99.5% pixel agreement for E0–E5 |
| Report data | `results/summary.md` and figure 1 generated from raw files (figures 2–4 if time allows) |

## 13. File change list

| File | Change |
|---|---|
| `efficientvit/models/nn/mhla.py` | Matmul-only rewrite, MALA support, loop-free unequal path |
| `efficientvit/models/nn/ops.py` | `attention_norm` arg, `mala_gate`, MALA paths |
| `efficientvit/models/efficientvit/backbone.py` | Pass `attention_norm` through |
| `efficientvit/seg_model_zoo.py` | e4/e5 entries, strict rule, `load_compatible_state_dict` helper |
| `applications/efficientvit_seg/train_efficientvit_seg_model.py` | e0–e5, init helper and flags, `final.pt`, new logs |
| `applications/efficientvit_seg/eval_efficientvit_seg_model.py` | e0–e5 choices |
| `applications/efficientvit_seg/benchmark_mhla.py` | e0–e5 choices |
| `applications/efficientvit_seg/configs/mhla/e4_mala.yaml`, `e5_mhla_mala.yaml` | New |
| `applications/efficientvit_seg/export_onnx_experiment.py` | New |
| `scripts/protocol.env` | New, the shared training recipe |
| `scripts/slurm/tc2_train.sbatch`, `scripts/slurm/tc2_submit_chain.sh` | New |
| `scripts/local/train_local.sh` | New (optional, local testing only) |
| `scripts/slurm/tc2_trt.sbatch` | New, TensorRT build, timing and parity on TC2 |
| `scripts/probe_throughput.py` | New |
| `scripts/trt/bench_trt.sh`, `layer_breakdown.py`, `trt_parity.py` | New |
| `scripts/collect_results.py`, `scripts/make_figures.py` | New |
| `tests/test_mhla.py` | New tests 1–9 |
| `docs/RESEARCH_SURVEY.md` | New |
| `MHLA_EXPERIMENTS.md`, `scripts/slurm/README.md` | Add E4/E5, R0, TC2 workflow, TRT protocol |

## 14. Out of scope

- Matching the published 80.5 mIoU, or reproducing the authors' unreleased training recipe.
- Other tasks in the repo (classification, SAM, DC-AE).
- Custom CUDA/Triton kernels or TensorRT plugins.
- INT8 or other quantization.
- Jetson and other edge devices. Latency is measured on one TC2 GPU only.
- Group-count ablation (M = 1, 2, 8). M = 1 ≡ E1 is proven by a test instead.
- Training any surveyed method other than MALA.
- A second seed, and an all-fp16 overflow demo.
- Changing the old 32-GPU launcher or `assets/onnx_export.py`.

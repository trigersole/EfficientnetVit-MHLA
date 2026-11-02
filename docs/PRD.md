# PRD: Faster EfficientViT Segmentation with Modern Linear Attention

- **Status:** draft for team review
- **Date:** 2026-10-30
- **Report due:** around 2026-11-07
- **Course track:** AI6103 Team Assignment, Option 2 (improve an existing codebase)
- **Companion doc:** [`SPEC.md`](SPEC.md) (how we build it)

---

## 1. One-line summary

- Make EfficientViT segmentation **faster** by removing its expensive "multi-scale" add-on. Try to **win back the lost accuracy** with two new attention ideas: MHLA (2026) and MALA (ICCV 2025). Then **prove the speed gain in real deployment** (TensorRT).

## 2. Background

- **Task:** semantic segmentation. The model colors every pixel of a street photo by class (road, car, person, pole, and so on).
- **Base model:** EfficientViT-B1, ICCV 2023 ([arXiv 2205.14756](https://arxiv.org/abs/2205.14756)), MIT Han Lab code.
- **How it understands context:** "linear attention". Every pixel writes into one shared summary, and every pixel reads from it.
  - This is fast, but **blurry**: everyone gets the same summary, and small objects get lost.
- **The authors' patch:** a "multi-scale" add-on (extra 5×5 filters) that sharpens local detail.
  - It costs about **18% of the run time**: 70 ms vs 57 ms per 1024×2048 image on our laptop GPU.
- **New fix 1, MHLA** ([ICLR 2026](https://iclr.cc/virtual/2026/poster/10011677), [arXiv 2601.07832](https://arxiv.org/abs/2601.07832)):
  - Splits the image into neighborhoods, each with its own summary.
  - Neighboring summaries are mixed.
  - Already built in this repo.
- **New fix 2, MALA** ([ICCV 2025, arXiv 2507.00698](https://arxiv.org/abs/2507.00698)):
  - Linear attention ignores how "strongly" a pixel asks its question (the query magnitude).
  - MALA puts that back, so confident pixels focus more sharply.
  - Adds no required parameters.

## 3. Problem

- The multi-scale add-on is accurate but slow.
- Prior-work check (searched 2026-10-30):
  - **MHLA has never been tested on segmentation.** Its paper covers classification, NLP and image/video generation only.
  - **MALA has been tested on segmentation**, but only in its own large backbone (MAViT, ADE20K). It has never been used in EfficientViT or a small CNN-hybrid, and never combined with MHLA.
  - **Nobody has replaced EfficientViT's multi-scale add-on with a newer linear attention.**
    - The closest work is MirrorLA ([arXiv 2602.04346](https://arxiv.org/abs/2602.04346), Feb 2026, preprint).
    - It builds its own backbone and lists EfficientViT only as a comparison row.
  - The fp16 overflow is known only as a GitHub issue, not studied in any paper.
- Speed measured in research code is not real deployment speed:
  - Real systems use **TensorRT** (NVIDIA's model compiler).
  - Our MHLA code uses an op (einsum) that TensorRT handles poorly.
  - EfficientViT-style attention can **overflow** in TensorRT's fast fp16 mode ([efficientvit issue #116](https://github.com/mit-han-lab/efficientvit/issues/116)).

### 3.1 Why MHLA and MALA

**EfficientViT's own authors name two weaknesses in their fast attention:**

- **It can't focus sharply.** Their paper says it "cannot produce sharp attention distributions".
- **It's weak at local detail.** Their paper says it is "weak at capturing local information".
- Their fix is the multi-scale add-on, which costs about 13 ms per image on our GPU.

**Each new method fixes one weakness, with no slow add-on:**

- **MHLA fixes the "everyone reads the same summary" problem (local detail).**
  - Today, the whole image shares **one** summary, so every pixel gets the same overview.
  - MHLA gives each region **its own** summary, mixed mostly with nearby regions. Each part of the image gets context that fits it.
  - That's the same job the add-on does (bringing in nearby information), done without extra filters.
  - So MHLA is the natural **replacement** for the add-on. E2 vs E0 tests exactly that.
- **MALA fixes the "can't focus" problem (sharpness).**
  - Every pixel's question has a strength: how big its numbers are.
  - Today, the attention **throws that strength away**. A pixel that is clearly a car focuses exactly as much as a blurry, unclear one.
  - MALA **keeps** the strength. Strong questions focus harder on the pixels that match and turn the rest down.
  - **The add-on does not fix this.** Even the original model throws the strength away. So MALA tests a second, separate weakness. E4 vs E1 tests it.

**Why test them together (E5):**

- They fix different things: MHLA changes *where* a pixel gets its context, MALA changes *how sharply* it focuses. So the gains could add up.
- They fit together cleanly. MALA needs a few totals that MHLA already computes per region.
- The 2×2 (E1, E2, E4, E5) shows each fix alone and both together, so we can tell which fix did what.

**Why these two, out of about 15 methods we surveyed:**

- **Strong results:** in the MHLA paper's ImageNet test (small DeiT-T model), plain linear attention scores 69.8%, the slow softmax version 72.2%, MALA 75.1%, and MHLA 75.8%. They are the top two in that table.
- **Cheap:**
  - MHLA adds 112 numbers to the whole model.
  - MALA adds no new layers, only one on/off dial per attention head (88 numbers).
- **Safe start:** with their switch at "off", both behave exactly like the old model, so the pretrained weights aren't wasted.
- **Deployable:** both are plain matrix math, so they export to TensorRT, which matches our speed goal.
- **A fair test:** neither uses convolution filters, so we test attention itself and don't sneak the add-on back in.
- **Runners-up and why we skipped them:** listed in `SPEC.md` §2.5.

**What we expect to see, if the ideas work:**

- **MHLA:** we can drop the add-on with little or no accuracy loss, and small objects (poles, signs, people) gain the most.
- **MALA:** sharper attention, and the learned dials end above "off" (0), meaning the model chose to use it.
- **Both:** better than either alone, if the two weaknesses really are separate.
- **These are predictions, not promises.** Neither method has been tested inside EfficientViT before. We will report the results either way.

## 4. Goals

- **G1:** Measure accuracy (mIoU) for 6 attention variants (E0–E5) under **one identical training recipe**.
- **G2:** Add MALA as a new option, alone and combined with MHLA.
- **G3:** Make every variant export to ONNX and TensorRT, and run safely in fp16.
- **G4:** Measure how long each variant (E0–E5) takes to process one image after it is converted with **TensorRT**, which is how a real product would run it. Do all the timing in **one TC2 job**, so every variant is timed on the same GPU.
- **G5:** Show the **official checkpoint's score as a reference row**, so the baseline is never hidden or weakened (see §7).
- **G6:** Produce a clear report and slides with one main take-away chart: speed vs accuracy.

## 5. Non-goals

- Beating the paper's published 80.5 mIoU. The rules say this is not required.
- Reproducing the authors' unreleased training recipe.
- Writing a custom CUDA/Triton kernel or TensorRT plugin.
- Edge devices (Jetson, phone). We measure on one TC2 cluster GPU only.
- Other tasks in the repo (classification, SAM, DC-AE).
- Testing other new attention methods. The survey lists them, but we do not train them.

## 6. Assignment fit

| Rule                                                             | How we meet it                                                                                                                                     |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Option 2: paper from 2019+ at an allowed venue, improve its code | EfficientViT is ICCV 2023; we modify its official codebase                                                                                         |
| Topic must be CV, NLP or tabular                                 | Computer vision (segmentation)                                                                                                                     |
| Novelty: newer work, not in lectures                             | MHLA (ICLR 2026) and MALA (ICCV 2025). First segmentation results for MHLA; first MHLA + MALA combination; first use of either inside EfficientViT |
| Difficulty: complex engineering                                  | New attention op, 2×2 controlled study, TensorRT deployment, 1-GPU cluster pipeline                                                               |
| Describe the technique precisely                                 | Exact formulas in `SPEC.md` §2; MALA checked against its paper and official code                                                                |
| Not required to match the paper                                  | We still report the official score (R0) for context                                                                                                |
| **Penalty for artificially lowering the baseline**         | Baseline gets the same recipe and budget as every arm; official checkpoint shown as R0 (see §7)                                                   |
| Presentation needs a take-away                                   | One chart: latency vs mIoU for E0–E5, plus R0                                                                                                     |

## 7. Baseline-reference fix (anti-penalty)

- **Risk:**
  - The authors trained their model with a long recipe they never published, and got 80.5 mIoU on Cityscapes
  - We retrain every version ourselves, with 1 GPU and a few days. So all our numbers will come out lower, including E0
  - Our E0 scores 74 while the paper says 80.5. A grader sees a baseline 6 points below the paper and may think we weakened it on
    purpose.
- **Fix:**
  - **R0 row:** evaluate the **official released E0 checkpoint** with our own evaluator on the full validation set.
    - Cityscapes: expect about 80.5.
    - ADE20K fallback: expect about 42.8.
  - **R0 has two jobs:**
    - It proves our evaluator is correct: it must reproduce the published number within about 80.5+-0.3.
    - It shows how much of the gap comes from the shorter training recipe.
  - **Equal treatment:**
    - E0 uses the **same** recipe, budget, seed and init as every other arm.
    - No per-arm tuning.

## 8. Experiments

| ID | Attention                  | Multi-scale add-on | Normalization | Role                              |
| -- | -------------------------- | ------------------ | ------------- | --------------------------------- |
| R0 | LiteMLA (official weights) | yes                | division      | reference only, not trained by us |
| E0 | LiteMLA                    | yes                | division      | baseline                          |
| E1 | LiteMLA                    | no                 | division      | cost of removing the add-on       |
| E2 | MHLA (4 groups)            | no                 | division      | MHLA as replacement               |
| E3 | MHLA (4 groups)            | yes                | division      | MHLA on top of the add-on         |
| E4 | LiteMLA                    | no                 | **MALA**      | MALA alone                        |
| E5 | MHLA (4 groups)            | no                 | **MALA**      | MHLA + MALA together              |

- All arms: EfficientViT-B1, ImageNet-pretrained backbone, same recipe, seed 0.
- **E1, E2, E4 and E5 test every on/off combination of MHLA and MALA.** The add-on is off in all four, and nothing else changes. So we can see what each idea adds on its own, and what they add together:

|              | MALA off | MALA on |
| ------------ | -------- | ------- |
| **MHLA off** | E1       | E4      |
| **MHLA on**  | E2       | E5      |

## 9. Hypotheses

- **H1:** E1 < E0. Removing the add-on hurts accuracy.
- **H2:** E2 > E1. MHLA wins some of that accuracy back.
- **H3:** E3 > E2. The add-on still helps on top of MHLA.
- **H4:** E4 > E1. MALA helps.
- **H5:** E5 ≥ max(E2, E4). The two fixes stack.
- **S1:** E2, E4 and E5 are faster than E0 in TensorRT fp16-mixed.
- **Reading rule:**
  - One seed only, so a gap under **0.5 mIoU** is called "inconclusive".
  - This threshold is a rule of thumb, not a measured variance.
- **Small vs large objects:** Cityscapes classes are split into small (person, rider, pole, traffic sign, bicycle) and large (road, building, car), to see *where* each fix helps.

## 10. Success metrics

- **Success means trustworthy, honestly reported results, not "our idea wins".** If MHLA or MALA don't help, the project still succeeds when every must-have below is done.
- **Must have**
  - **Our scoring code is correct:** it scores the official model (R0) within 0.3 mIoU of the published number (80.2–80.8 on Cityscapes).
  - **Fair training:** E0–E5 train for the same number of steps with the same settings.
  - **Final accuracy:** each of E0–E5 is scored on the full validation set at full image size, as an overall score (mIoU) and one score per class.
  - **Speed:** each of E0–E5 is timed in TensorRT, in full precision (fp32) and in mostly half precision (fp16-mixed), on the same GPU in one TC2 job.
  - **Conversion doesn't change the answers:**
    - ONNX vs PyTorch: outputs differ by less than 0.001.
    - TensorRT fp16-mixed vs PyTorch: at least 99.5% of pixels get the same class.
  - **The report contains:** the main results table, the speed-vs-accuracy chart, and a verdict for each hypothesis (H1–H5, S1).
- **Should have**
  - **Where the time goes:** the share of run time spent in attention, E0 vs E2, measured with TensorRT.
  - **What the model learned:** figures of the final MALA dial values and the MHLA mixing tables.

# Running E1–E3 on TC2 (one arm per account)

Each TC2 account gets 1 GPU, 6-hour jobs and at most 2 queued jobs. The recipe is **protocol v2**, fixed in
`scripts/protocol.env`; do not edit it. v2 = 1024×1024 crops + full backbone LR. On E0 it gives 76.78 mIoU,
vs 72.78 for v1; the paper's model scores 80.55.
One run (40,000 steps) takes about 7 h = a 2-job chain. Each arm needs 3 seeds (0, 1, 2), run one after another.

| Arm | Attention | Multi-scale | Who |
|-----|-----------|-------------|-----|
| E0  | LiteMLA   | yes         | kieu0003 (seed 0 done: 76.78) |
| E1  | LiteMLA   | no          | teammate A |
| E2  | MHLA (4 groups) | no    | teammate B |
| E3  | MHLA (4 groups) | yes   | teammate C |
| E4/E5 | MALA    | —           | not implemented yet |

## 0. Connect
- VPN: GlobalProtect `vpngate-student.ntu.edu.sg` (Linux: `sudo -E gpclient connect --browser manual vpngate-student.ntu.edu.sg`; the session drops every few hours, but queued jobs keep running).
- `ssh <your_username>@10.96.189.12`

## 1. Get the code (branch `ryan/tc2-training`; after the PR is merged, use `codex/mhla-efficientvit` from trigersole)
```bash
git clone -b ryan/tc2-training https://github.com/kieusontungwork-oss/EfficientnetVit-MHLA.git ~/EfficientnetVit-MHLA
cd ~/EfficientnetVit-MHLA
```

## 2. Set up env + checkpoints (~10 min, run once)
```bash
bash scripts/tc2/setup_account.sh        # must end with "SETUP_DONE" and two "OK" sha lines
```

## 3. Copy Cityscapes into your own home (accounts cannot read each other's folders)
From your laptop (the extracted Kaggle data, `leftImg8bit/` + `gtFine/`, ~7.7 GB, ~20 min over VPN):
```bash
rsync -a --partial --info=progress2 /path/to/cityscapes/ <your_username>@10.96.189.12:datasets/cityscapes/
```
Then on TC2:
```bash
~/envs/effvit/bin/python scripts/tc2/check_cityscapes.py ~/datasets/cityscapes   # must print DATASET_OK
```

## 4. Optional sanity check: R0 (1 min of GPU)
```bash
sbatch scripts/slurm/tc2_r0_eval.sbatch
# logs/effvit-r0-<job>.out must end with: mIoU = 80.547
```

## 5. Launch your arm
```bash
SEED=0 bash scripts/slurm/tc2_submit_chain.sh e1 2     # or e2 / e3 — exactly your assigned arm
squeue -u $USER
# when seed 0 has eval_full.txt (~7 h later):
SEED=1 bash scripts/slurm/tc2_submit_chain.sh e1 2
# then:
SEED=2 bash scripts/slurm/tc2_submit_chain.sh e1 2
```
- Job 1 trains ~5.9 h and saves `last.pt` 5 min before the limit. Job 2 resumes, finishes (~1 h), then runs
  the official evaluator.
- Only one seed at a time: a chain fills both of your queue slots.
- Progress: `tail -f logs/train-e1-<job>.out` or `~/runs/protocol_v2/e1_m4_s<seed>/metrics.jsonl`.
- The in-training `miou` uses half-size images and reads ~8 points low; only `eval_full.txt` counts.
- If both jobs end and `final.pt` is still missing, run `SEED=<seed> bash scripts/slurm/tc2_submit_chain.sh e1 1`
  again (it resumes from `last.pt`).

## 6. Report back
Send these files from each `~/runs/protocol_v2/<arm>_m4_s<seed>/`:
- `eval_full.txt` (the reported number: official evaluator, 500 val images, 1024×2048, `final.pt`)
- `metrics.jsonl`, `config.json`, `initialization_report.json`
- the first 3 lines of every `logs/train-<arm>-*.out` (job id, node, git revision)

Expected init report: E1 `loaded=247 sliced=7`, E2 `loaded=247 sliced=7` + 7 fresh MHLA weights,
E3 `loaded=261` + 7 fresh MHLA weights. Anything else: stop and tell Ryan.

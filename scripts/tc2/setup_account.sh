#!/bin/bash
# One-time setup of a TC2 account for the protocol_v1 runs: Python env + the two checkpoints.
# Run on the TC2 head node from the repo root:  bash scripts/tc2/setup_account.sh
# (The dataset is copied separately; see docs/TC2_TEAMMATE_GUIDE.md.)
set -eo pipefail
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
if [ ! -x ~/envs/effvit/bin/python ]; then
  # conda-forge only: avoids the defaults-channel Terms-of-Service prompt
  conda create -y -p ~/envs/effvit --override-channels -c conda-forge python=3.11
fi
conda activate ~/envs/effvit
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy "opencv-python-headless<4.12" pillow timm einops tqdm torchprofile scipy \
  omegaconf huggingface-hub onnx onnxsim onnxruntime matplotlib torchmetrics

mkdir -p ~/ckpt ~/datasets logs
cd ~/ckpt
[ -f efficientvit_b1_r224.pt ] || wget -q https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r224.pt
[ -f efficientvit_seg_b1_cityscapes.pt ] || wget -q https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b1_cityscapes.pt
sha256sum -c <<'EOF'
bf8798aa03ed2bba21fc8d58f43ca9a92096f3f03bd700e1df1dcde11559f499  efficientvit_b1_r224.pt
80c305bb2a94b9217187bfe4c029618503e0a7c6651edfe2e8a6cfc226426dd4  efficientvit_seg_b1_cityscapes.pt
EOF
python -c "import torch, cv2, timm, onnxsim; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo SETUP_DONE

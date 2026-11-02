#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/workspace/serum
cd "$ROOT"
exec > >(tee -a logs/setup.log) 2>&1
echo "[$(date -Is)] setup started"
if [ ! -d third_party/SERUM/.git ]; then git clone --depth 1 https://github.com/Hubizon/SERUM.git third_party/SERUM; fi
if [ ! -d third_party/CtrlRegen/.git ]; then git clone --depth 1 https://github.com/yepengliu/CtrlRegen.git third_party/CtrlRegen; fi
if [ ! -d third_party/diffusers/.git ]; then git clone --depth 1 https://github.com/huggingface/diffusers.git third_party/diffusers; fi
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e third_party/diffusers
python -m pip install -r third_party/diffusers/examples/text_to_image/requirements.txt
python -m pip install controlnet-aux lpips scikit-learn scipy pandas
python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
{
 echo "SERUM $(git -C third_party/SERUM rev-parse HEAD)"
 echo "CtrlRegen $(git -C third_party/CtrlRegen rev-parse HEAD)"
 echo "diffusers $(git -C third_party/diffusers rev-parse HEAD)"
 python -m pip freeze
} > manifests/environment.txt
printf '{"setup":"complete","created_at":"%s"}\n' "$(date -Is)" > manifests/setup_status.json
echo "[$(date -Is)] setup completed"
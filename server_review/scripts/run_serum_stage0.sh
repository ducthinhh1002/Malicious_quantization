#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/workspace/serum
cd "$ROOT"
exec > >(tee -a logs/serum_stage0.log) 2>&1
while tmux has-session -t quant-assets 2>/dev/null; do sleep 15; done
while tmux has-session -t quant-sd21 2>/dev/null; do sleep 15; done
if [ ! -x .venv-serum/bin/python ]; then
  python3 -m venv .venv-serum
  .venv-serum/bin/python -m pip install --upgrade pip setuptools wheel
  .venv-serum/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  .venv-serum/bin/python -m pip install -r third_party/SERUM/requirements.txt
fi
source .venv-serum/bin/activate
cd third_party/SERUM
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.data.dataset --config configs/config_sd21.yaml --mode both --batch-size 32
python -m src.training.train --config configs/config_sd21.yaml --dont-load-checkpoint
python -m src.evaluation.generate --config configs/config_sd21.yaml --generate --prompts-path ../../prompts/adapt_train.txt --clean-path ../../data/raw/clean_train --watermarked-path ../../data/raw/wm_train --num-samples 5000 --batch-size 32
python -m src.evaluation.generate --config configs/config_sd21.yaml --generate --prompts-path ../../prompts/adapt_val.txt --clean-path ../../data/raw/clean_val --watermarked-path ../../data/raw/wm_val --num-samples 1000 --batch-size 32
python -m src.evaluation.generate --config configs/config_sd21.yaml --generate --prompts-path ../../prompts/detector_calib.txt --clean-path ../../data/raw/calib_clean --watermarked-path ../../data/raw/calib_wm_unused --num-samples 5000 --batch-size 32
cd "$ROOT"
python - <<'PY'
from pathlib import Path
import json,hashlib
root=Path('/root/workspace/serum/third_party/SERUM')
items={}
for p in (root/'results/SERUM_sd21/models').glob('*.pt'):
 items[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('/root/workspace/serum/manifests/serum_artifacts.json').write_text(json.dumps(items,indent=2)+'\n')
PY
echo "[$(date -Is)] SERUM Stage 0 and teacher datasets complete"
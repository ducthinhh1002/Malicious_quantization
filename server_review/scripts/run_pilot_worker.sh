#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
KIND=$1
GPU=$2
export CUDA_VISIBLE_DEVICES="$GPU"
.venv/bin/python scripts/serum_scores.py --images "data/pilot/$KIND" --out "metrics/pilot/${KIND}_raw_scores.npy"
for step in 0.25 0.50 0.75 1.00; do
  out="data/pilot/${KIND}_${step}"
  .venv-ctrlregen/bin/python scripts/launder_ctrlregen.py --ctrlregen-repo third_party/CtrlRegen --input-dir "data/pilot/$KIND" --output-dir "$out" --step "$step" --base-seed 10000 --spatial-control third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000 --semantic-control third_party/CtrlRegen/public_checkpoints/semanticnet_ckp
  .venv-student/bin/python scripts/pilot_metrics.py --raw "data/pilot/$KIND" --regen "$out" --prompts data/pilot/prompts.txt --out "metrics/pilot/${KIND}_${step}.json"
  .venv/bin/python scripts/serum_scores.py --images "$out" --out "metrics/pilot/${KIND}_${step}_scores.npy"
done

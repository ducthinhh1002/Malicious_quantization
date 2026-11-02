#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
: "${GPU:=0}"
.venv/bin/python scripts/check_stage_gates.py
.venv/bin/python - <<'PY'
from pathlib import Path
import os
r=Path('/root/workspace/serum'); d=r/'data/pilot'; d.mkdir(parents=True,exist_ok=True)
for kind in ('clean','wm'):
    out=d/kind; out.mkdir(exist_ok=True)
    for src in sorted((r/'data/raw'/f'{kind}_train').glob('*.png'))[:500]:
        dst=out/src.name
        if not dst.exists(): os.link(src,dst)
(d/'prompts.txt').write_text('\n'.join((r/'prompts/adapt_train.txt').read_text().splitlines()[:500])+'\n')
PY
for kind in clean wm; do
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/serum_scores.py --images "data/pilot/$kind" --out "metrics/pilot/${kind}_raw_scores.npy"
done
for step in 0.25 0.50 0.75 1.00; do
  for kind in clean wm; do
    out="data/pilot/${kind}_${step}"
    CUDA_VISIBLE_DEVICES="$GPU" .venv-ctrlregen/bin/python scripts/launder_ctrlregen.py --ctrlregen-repo third_party/CtrlRegen --input-dir "data/pilot/$kind" --output-dir "$out" --step "$step" --base-seed 10000 --spatial-control third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000 --semantic-control third_party/CtrlRegen/public_checkpoints/semanticnet_ckp
    CUDA_VISIBLE_DEVICES="$GPU" .venv-student/bin/python scripts/pilot_metrics.py --raw "data/pilot/$kind" --regen "$out" --prompts data/pilot/prompts.txt --out "metrics/pilot/${kind}_${step}.json"
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/serum_scores.py --images "$out" --out "metrics/pilot/${kind}_${step}_scores.npy"
  done
done
echo 'Pilot finished. Choose mild/strong from preservation metrics and record the decision; do not use student detector results.'

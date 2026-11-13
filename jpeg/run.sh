#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bash setup.sh
PY="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>run.lock
flock -n 9 || { echo 'This JPEG run is already active'; exit 1; }
mkdir -p logs
"$PY" -u prepare_inputs.py >logs/00_inputs.log 2>&1
if ! test -f run/teacher_fp/refined/trained.json; then
  "$PY" -u prepare_codec.py >logs/01_carrier.log 2>&1
  "$PY" -u train.py --run "$PWD/run" --smoke >logs/02_smoke.log 2>&1
  "$PY" -u train.py --run "$PWD/run" >logs/03_train.log 2>&1
fi
"$PY" -u generate_test.py >logs/04_generate.log 2>&1
exec "$PY" -u evaluate_jpeg.py >logs/05_evaluate.log 2>&1

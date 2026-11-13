#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if test -z "${DCT_PYTHON:-}"; then bash setup.sh; DCT_PYTHON="$PWD/.venv/bin/python"; fi
PY="$DCT_PYTHON"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BASELINE_CACHE="$PWD/.cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>run.lock
flock -n 9 || { echo 'DCT pipeline already active'; exit 1; }
mkdir -p logs
"$PY" -u prepare.py >logs/00_prepare.log 2>&1
"$PY" -u -m benchmark.scripts.latent32_student_job_parallel >logs/01_pipeline.log 2>&1
"$PY" -u -m benchmark.scripts.latent32_agg32 >logs/02_aggregate32.log 2>&1
printf 'DCT teacher, student and aggregate32 evaluation completed.\n'

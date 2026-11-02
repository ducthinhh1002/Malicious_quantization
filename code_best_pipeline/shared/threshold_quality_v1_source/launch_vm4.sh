#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BITMARK_RUNTIME_ROOT=/root/logs/runtime
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /root/logs/gate_env/bin/python "$ROOT/main.py" \
  --teacher-objective threshold_cfg_v1 --threshold-quality \
  --teacher-kappas 2 1.5 1 --teacher-min-tpr 0.99 \
  --key-seed 0 --seed 0 --num-bits 8 --stable-candidate-count 16 \
  --carrier-pc-min 8 --carrier-pc-max 64 --fpr 0.01 \
  --teacher-rank 32 --teacher-lr 1e-4 --teacher-train-steps 3000 \
  --teacher-batch 4 --teacher-micro-batch 4 --teacher-precision bf16 \
  --teacher-checkpoint-every 500 --semantic-t-max 500 --cycle-every 1 \
  --generation-batch 16 --score-batch 8 --inference-batch 16 --inference-steps 50 \
  --ranks 32 --adapter-scopes attention_conv --no-dora --lora-only \
  --batch-size 4 --learning-rate 1e-4 --max-train-steps 7500 \
  --checkpoint-steps 2500 5000 7500 --precision fp16 \
  --train-n 1000 --validation-n 256 --validation-generation-n 128 \
  --calibration-n 5000 --test-n 1000 --validation-draws 2 \
  --teacher-snapshot /root/logs/runtime/cache/sd2-community--stable-diffusion-2-1-base/4e63672c03103b6c636b8fb4119ba982469b2955 \
  --student-snapshot /root/logs/runtime/cache/CompVis--stable-diffusion-v1-4/133a221b8aa7292a167afc5127cb63fb5005638b \
  --cache /root/logs/runtime/cache --min-free-gb 70 --output "$ROOT/run" "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CUDA_VISIBLE_DEVICES=1 STRUCTURE32_EXPERIMENT=1
export PYTHONPATH="$ROOT:$ROOT/core:/var/lib/vllm/harness"
export BITMARK_RUNTIME_ROOT=/var/lib/vllm/runtime
export HF_HOME=/var/lib/vllm/runtime/huggingface TORCH_HOME=/var/lib/vllm/runtime/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec /var/lib/vllm/env/bin/python "$@"

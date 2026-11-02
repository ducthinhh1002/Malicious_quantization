#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${NEW_WATERMARK_PYTHON:-python3}"
export PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT_DIR"
if ! "$PYTHON_BIN" -c 'import torch, diffusers, peft, datasets, lpips, transformers, safetensors' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-learning-channel.txt"
fi
exec "$PYTHON_BIN" -u -m new_watermark.learning_channel.run "$@"

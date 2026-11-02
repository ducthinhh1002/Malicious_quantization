#!/usr/bin/env bash
# Native semantic-relation replacement pilot; server assets must already exist.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${SEMANTIC_PYTHON:-/root/logs/gate_env/bin/python}" -u -m new_watermark.learning_channel.semantic_experiment "$@"

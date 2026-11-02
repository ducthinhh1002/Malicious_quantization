#!/usr/bin/env bash
# Owner-only weighted-denoising teachers; ordinary students are not modified.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${NEW_WATERMARK_PYTHON:-python3}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
GATE_ROOT="${DISTRIBUTION_GATE_ROOT:-${BITMARK_RESULTS_ROOT:-$ROOT_DIR}/distribution_gate_$(date +%Y%m%d_%H%M%S)_$$}"
exec "$PYTHON_BIN" -u "$ROOT_DIR/new_watermark/distribution.py" --output "$GATE_ROOT" "$@"

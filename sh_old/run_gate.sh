#!/usr/bin/env bash
# Paired C3 off/on pilot: 500 training/test images, teacher 250 steps.
# Each arm: clean/marked LoRA r4, 2500 steps, snapshots 500/1000/2500.
# Total: 10000 student updates. Runtime has not been benchmarked.
# Usage: bash run_gate.sh [--plan-only] [--output-root /path/new_gate]
# Read gate_summary.csv/json: paired CI plus teacher/control/quality gates.
# PCA feature drift and C3 gain are diagnostics, not standalone pass/fail rules.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${NEW_WATERMARK_PYTHON:-${SERUM_PYTHON:-${BITMARK_PYTHON:-}}}"
if [[ -z "$PYTHON_BIN" ]]; then
  RUNTIME_DIR="${BITMARK_RUNTIME_ROOT:-/content/bitmark_runtime}"
  if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$RUNTIME_DIR/.venv/bin/python"
  elif [[ -x "$ROOT_DIR/bitmark_lora/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/bitmark_lora/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" -u "$ROOT_DIR/new_watermark/gate.py" "$@"

#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export USE_TORCH=1 USE_TF=0 USE_FLAX=0
# Defaults make a full run one command; explicit CLI arguments override defaults.
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec python "$SCRIPT_DIR/wmq_blind.py" "$@"
  fi
done
python - <<'PY'
import torch
import diffusers
import transformers
import safetensors
import PIL
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Activate the prepared wmq environment on a GPU node.")
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
PY

RUN_ROOT="${WMQ_ROOT:-$SCRIPT_DIR/wmq_runs}"
ATTACK_OUTPUT_ROOT="$SCRIPT_DIR/output_attack"
DEFAULT_MODEL="$RUN_ROOT/marked_sd21"
CUSTOM_MODEL=0
for arg in "$@"; do
  if [[ "$arg" == "--model" || "$arg" == --model=* ]]; then
    CUSTOM_MODEL=1
  fi
done
mkdir -p "$RUN_ROOT"
if [[ "$CUSTOM_MODEL" == "0" ]]; then
  echo "Preparing or reusing public Stable Signature fixture..."
  python "$SCRIPT_DIR/prepare_marked_fixture.py" --output "$DEFAULT_MODEL" --reuse
fi
mkdir -p "$ATTACK_OUTPUT_ROOT"
DEFAULT_OUTPUT="$ATTACK_OUTPUT_ROOT/blind_$(date +%Y%m%d_%H%M%S)_$$"
LOG_PATH="$(mktemp "$ATTACK_OUTPUT_ROOT/blind_run_XXXXXX.log")"
echo "Default output: $DEFAULT_OUTPUT (overridden by --output if supplied)"
echo "Log: $LOG_PATH"
python "$SCRIPT_DIR/wmq_blind.py" \
  --model "$DEFAULT_MODEL" \
  --prompts "$SCRIPT_DIR/prompt.txt" \
  --output "$DEFAULT_OUTPUT" \
  "$@" 2>&1 | tee "$LOG_PATH"

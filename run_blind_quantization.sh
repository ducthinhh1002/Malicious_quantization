#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${WMQ_ROOT:-$SCRIPT_DIR/wmq_runs}"
ATTACK_OUTPUT_ROOT="${WMQ_ATTACK_OUTPUT_ROOT:-$SCRIPT_DIR/output_attack}"
IMAGE_OUTPUT_ROOT="${WMQ_IMAGE_OUTPUT_ROOT:-$SCRIPT_DIR/output_image}"

if [[ -z "${CONDA_PREFIX:-}" || ! -d "$CONDA_PREFIX/conda-meta" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
  CONDA_BIN=""
  for candidate in "${CONDA_EXE:-}" "${HOME:-}/miniconda3/bin/conda" "${HOME:-}/anaconda3/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$CONDA_BIN" ]] && type -P conda >/dev/null 2>&1; then
    CONDA_BIN="$(type -P conda)"
  fi
  if [[ -z "$CONDA_BIN" ]]; then
    echo "Conda was not found. Install the prepared environment on the login node first." >&2
    exit 1
  fi
  echo "Entering Conda environment ${WMQ_CONDA_ENV:-wmq}..." >&2
  exec "$CONDA_BIN" run --no-capture-output -n "${WMQ_CONDA_ENV:-wmq}" bash "$0" "$@"
fi
PY="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false USE_TORCH=1 USE_TF=0 USE_FLAX=0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV

mkdir -p "$ATTACK_OUTPUT_ROOT"
LOG_PATH="$(mktemp "$ATTACK_OUTPUT_ROOT/blind_run_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "Log (preflight, fixture, attack, owner evaluation): $LOG_PATH"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PY" "$SCRIPT_DIR/wmq_blind.py" "$@"
fi

"$PY" - <<'PY'
import os
import sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    raise SystemExit("Python does not belong to the active Conda environment")
import torch, diffusers, transformers, safetensors, PIL, scipy
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Run this launcher inside the prepared Conda environment on a GPU node.")
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; diffusers={diffusers.__version__}", flush=True)
PY

DEFAULT_MODEL="$RUN_ROOT/marked_sd21"
DEFAULT_OUTPUT="$ATTACK_OUTPUT_ROOT/blind_$(date +%Y%m%d_%H%M%S)_$$"
MODEL_PATH="$DEFAULT_MODEL"
ATTACK_OUTPUT="$DEFAULT_OUTPUT"
CUSTOM_MODEL=0
HAS_OUTPUT=0
HAS_IMAGE_OUTPUT=0
HAS_BITS=0
HAS_NATURAL=0
TRAIN_N=32
SEARCH_N=20
DATA_SEED=3407
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  argument="${arguments[index]}"
  case "$argument" in
    --model)
      ((index + 1 < ${#arguments[@]})) || { echo "--model requires a path" >&2; exit 2; }
      MODEL_PATH="${arguments[index+1]}"; CUSTOM_MODEL=1; ((index+=1)) ;;
    --model=*) MODEL_PATH="${argument#--model=}"; CUSTOM_MODEL=1 ;;
    --output)
      ((index + 1 < ${#arguments[@]})) || { echo "--output requires a path" >&2; exit 2; }
      ATTACK_OUTPUT="${arguments[index+1]}"; HAS_OUTPUT=1; ((index+=1)) ;;
    --output=*) ATTACK_OUTPUT="${argument#--output=}"; HAS_OUTPUT=1 ;;
    --image-output|--image-output=*) HAS_IMAGE_OUTPUT=1 ;;
    --bits|--bits=*) HAS_BITS=1 ;;
    --natural-images|--natural-images=*) HAS_NATURAL=1 ;;
    --train-n|--search-n|--seed)
      ((index + 1 < ${#arguments[@]})) || { echo "$argument requires a value" >&2; exit 2; }
      value="${arguments[index+1]}"
      case "$argument" in
        --train-n) TRAIN_N="$value" ;; --search-n) SEARCH_N="$value" ;; --seed) DATA_SEED="$value" ;;
      esac
      ((index+=1)) ;;
    --train-n=*) TRAIN_N="${argument#*=}" ;;
    --search-n=*) SEARCH_N="${argument#*=}" ;;
    --seed=*) DATA_SEED="${argument#*=}" ;;
  esac
done
ATTACK_OUTPUT="$(realpath -m "$ATTACK_OUTPUT")"

mkdir -p "$RUN_ROOT"
if [[ "$CUSTOM_MODEL" == "0" ]]; then
  echo "[1/4] Preparing or reusing the pinned public Stable Signature fixture..."
  "$PY" "$SCRIPT_DIR/prepare_marked_fixture.py" --output "$DEFAULT_MODEL" --reuse
else
  [[ -n "${SS_KEY:-}" ]] || {
    echo "A custom --model requires SS_KEY; the public fixture key must not be assumed for another model." >&2
    exit 2
  }
  echo "[1/4] Using custom marked model: $MODEL_PATH"
fi

attack=("$PY" "$SCRIPT_DIR/wmq_blind.py")
if [[ "$CUSTOM_MODEL" == "0" ]]; then attack+=(--model "$DEFAULT_MODEL"); fi
attack+=(--prompts "$SCRIPT_DIR/prompt.txt")
if [[ "$HAS_OUTPUT" == "0" ]]; then attack+=(--output "$DEFAULT_OUTPUT"); fi
if [[ "$HAS_IMAGE_OUTPUT" == "0" ]]; then attack+=(--image-output "$IMAGE_OUTPUT_ROOT/$(basename -- "$ATTACK_OUTPUT")"); fi
if [[ "$HAS_BITS" == "0" ]]; then attack+=(--bits 8 4); fi
if [[ "${WMQ_MODEL_ONLY:-0}" == "1" && "$HAS_NATURAL" == "1" ]]; then
  echo "WMQ_MODEL_ONLY=1 conflicts with --natural-images" >&2; exit 2
fi
if [[ "${WMQ_MODEL_ONLY:-0}" != "1" && "$HAS_NATURAL" == "0" ]]; then
  [[ "$TRAIN_N" =~ ^[1-9][0-9]*$ && "$SEARCH_N" =~ ^[1-9][0-9]*$ && "$DATA_SEED" =~ ^[0-9]+$ ]] || {
    echo "Natural pool preparation requires positive train/search counts and a nonnegative seed" >&2; exit 2;
  }
  NATURAL_COUNT=$((10#$TRAIN_N + 10#$SEARCH_N))
  NATURAL_POOL="$RUN_ROOT/public_data/coco2017_n${NATURAL_COUNT}_seed${DATA_SEED}"
  echo "Preparing public COCO natural images for the additional branches..."
  "$PY" "$SCRIPT_DIR/prepare_natural_images.py" --output "$NATURAL_POOL" --count "$NATURAL_COUNT" --seed "$DATA_SEED"
  attack+=(--natural-images "$NATURAL_POOL")
fi
attack+=("$@")

echo "[2/4] Running blind Stable Signature quantization (default comparison: W8 and W4)..."
echo "Result directory: $ATTACK_OUTPUT"
"${attack[@]}"
[[ -f "$ATTACK_OUTPUT/report.json" && -f "$ATTACK_OUTPUT/selection_frozen.json" ]] || {
  echo "Attack did not produce a complete frozen run at $ATTACK_OUTPUT" >&2
  exit 1
}

OFFICIAL_EXTRACTOR_SHA256="77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04"
if [[ -n "${SS_EXTRACTOR:-}" ]]; then
  EXTRACTOR="$(realpath -m "$SS_EXTRACTOR")"
  [[ -n "${SS_EXTRACTOR_SHA256:-}" ]] || {
    echo "A custom SS_EXTRACTOR requires SS_EXTRACTOR_SHA256." >&2
    exit 2
  }
  EXTRACTOR_SHA256="$SS_EXTRACTOR_SHA256"
else
  EXTRACTOR="$RUN_ROOT/owner_assets/dec_48b_whit.torchscript.pt"
  EXTRACTOR_SHA256="$OFFICIAL_EXTRACTOR_SHA256"
  echo "[3/4] Preparing or reusing the checksum-pinned official owner extractor..."
  "$PY" "$SCRIPT_DIR/prepare_owner_evaluator.py" --output "$EXTRACTOR"
fi

if [[ "$CUSTOM_MODEL" == "0" ]]; then
  KEY="${SS_KEY:-111010110101000001010111010011010100010000100111}"
  KEY_SOURCE="${SS_KEY_SOURCE:-facebookresearch/stable_signature public SD2 decoder example key}"
else
  KEY="$SS_KEY"
  KEY_SOURCE="${SS_KEY_SOURCE:-user-supplied key for custom marked model}"
fi
[[ ${#KEY} -eq 48 && "$KEY" != *[!01]* ]] || { echo "SS_KEY must contain exactly 48 binary digits." >&2; exit 2; }

echo "[4/4] Running owner-only evaluation after selection freeze..."
evaluation=("$PY" "$SCRIPT_DIR/evaluate_blind_watermark.py"
  --run "$ATTACK_OUTPUT" --extractor "$EXTRACTOR"
  --expected-extractor-sha256 "$EXTRACTOR_SHA256"
  --key "$KEY" --key-source "$KEY_SOURCE"
  --batch-size "${WMQ_OWNER_BATCH_SIZE:-0}"
  --fpr "${FPR:-0.001}" --detector "${DETECTOR:-double}" --min-reference-tpr "${MIN_REFERENCE_TPR:-0.9}")
if [[ "${WMQ_EVAL_LPIPS:-1}" == "1" ]]; then evaluation+=(--lpips); fi
"${evaluation[@]}"

"$PY" - "$ATTACK_OUTPUT/owner_evaluation.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
validation = report["reference_validation"]
print(f"Reference validation: valid={validation['valid']}; "
      f"TPR={report['reference_tpr']:.4f}; detected={report['baseline_detected_count']}")
if report["baseline_warning"]:
    print("WARNING:", report["baseline_warning"])
    print("Attack rows were retained, but must not be interpreted as successful watermark suppression.")
PY

echo "Finished: $ATTACK_OUTPUT"
echo "Owner results: $ATTACK_OUTPUT/owner_evaluation.json"
echo "Plot-ready table: $ATTACK_OUTPUT/watermark_retention.csv"
echo "All search candidates: $ATTACK_OUTPUT/search.csv"
echo "Quality diagnostics: $ATTACK_OUTPUT/quality_summary.csv"

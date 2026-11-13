#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
JPEG_CODE_DIR="${JPEG_CODE_DIR:-$(cd ../jpeg && pwd)}"
export JPEG_CODE_DIR
bash "$JPEG_CODE_DIR/setup.sh"
PY="$JPEG_CODE_DIR/.venv/bin/python"
export PYTHONPATH="$PWD/Janus${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>run.lock
flock -n 9 || { echo 'This Janus run is already active'; exit 1; }
if ! test -f "${JPEG_TEACHER_RUN:-$JPEG_CODE_DIR/run}/images/teacher_jpeg600_test/image_hashes.json"; then
  if test -n "${JPEG_TEACHER_RUN:-}"; then echo 'Supplied teacher run is incomplete'; exit 1; fi
  bash "$JPEG_CODE_DIR/run.sh"
fi
"$PY" -u prepare_model.py
exec "$PY" -u job.py

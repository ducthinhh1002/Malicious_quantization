#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${BITMARK_PYTHON:-python3}" -u "$ROOT_DIR/bitmark_lora/run.py" "$@"

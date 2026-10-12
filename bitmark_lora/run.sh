#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
BOOTSTRAP_PYTHON="${BITMARK_PYTHON:-python3}"
if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo 'Python 3 is required. Set BITMARK_PYTHON to a Python executable.' >&2
  exit 1
fi
# These modes only inspect the protocol/results, regardless of argument order.
for arg in "$@"; do
  case "$arg" in
    --help|-h|--plan-only|--analyze|--analyze=*)
      exec "$BOOTSTRAP_PYTHON" "$HERE/run.py" "$@" ;;
  esac
done
exec "$BOOTSTRAP_PYTHON" "$HERE/bootstrap_env.py" "$@"

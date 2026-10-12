#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${BITMARK_PYTHON:-python3}" -u "$HERE/run.py" "$@"

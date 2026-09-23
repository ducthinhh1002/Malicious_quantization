#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
PY="${WMQ_SUITE_PYTHON:-$(type -P python3 || type -P python || true)}"
[[ -x "$PY" ]] || { echo 'Python 3 is required to orchestrate the suite.' >&2; exit 1; }
exec "$PY" "$SCRIPT_DIR/run_blind_suite.py" "$@"

#!/usr/bin/env bash
# Compatibility entry point. The canonical one-command pipeline is kept in one place.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_blind_quantization.sh" "$@"

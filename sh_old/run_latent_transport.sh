#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# One complete run: long joint teacher, learning-channel optimization, ordinary
# students/clean controls if the final teacher gate passes. No tiny-fit gate.
exec bash "$ROOT_DIR/run_learning_channel_gate.sh" --profile pilot --ablation transport "$@"

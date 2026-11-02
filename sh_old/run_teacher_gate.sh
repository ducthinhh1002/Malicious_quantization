#!/usr/bin/env bash
# Screen native teacher embedding BEFORE spending student-training compute.
# Uses the same data, key, DDIM sampler, image penalty and budgeted carrier in
# both arms; only the teacher training proxy vs final-sample objective changes.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GATE_ROOT="${TEACHER_GATE_ROOT:-${BITMARK_RESULTS_ROOT:-$ROOT_DIR}/teacher_gate_$(date +%Y%m%d_%H%M%S)_$$}"
COMMON=(--teacher-only --detector-basis budgeted --carrier-rank 64 --stable-rank 8
  --generation-sampler ddim --inference-steps 20 --rollout-grad-steps 4
  --teacher-checkpoints 250 --teacher-lr 5e-5 --image-preservation-weight 2
  --channel-warmup 50 --no-double-channel --design-fit-n 256 --teacher-train-n 500
  --calibration-n 1000 --pilot-eval-n 128 --student-kinds lora --ranks 4
  --train-n 500 --test-n 500 --validation-n 128 --validation-generation-n 32
  --skip-utility --min-free-gb 30)
for mode in proxy rollout; do
  bash "$ROOT_DIR/run_new.sh" "${COMMON[@]}" "$@" --teacher-training-mode "$mode" --output "$GATE_ROOT/$mode"
done
printf 'Teacher-only results (no student training): %s/{proxy,rollout}/result/teacher_selection.json\n' "$GATE_ROOT"

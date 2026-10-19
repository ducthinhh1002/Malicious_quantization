#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# This experiment checkpoint was recorded on vm_4; override for other machines.
OWNER="${NATIVE_OWNER_CHECKPOINT:-/root/logs/runtime/lalt_probe_artifacts/lalt_step20000_key221_probe_20261018_231633Z/owner_step20000.pt}"
HASH="${NATIVE_OWNER_SHA256:-bce50d0a1a450ae315a90c4bfcfafe22c93b6bb79ceea85980a9ddef357e8eb0}"
exec bash "$ROOT_DIR/run_new.sh" \
  --owner-checkpoint "$OWNER" --owner-sha256 "$HASH" \
  --exclude-prior-prompts "${NATIVE_PRIOR_PROMPTS:-/root/logs/results/lalt_stable_reader_20261018_111112Z/prompts.json}" \
  --teacher-checkpoints 500 1000 2000 --teacher-lr 5e-5 \
  --teacher-tail-steps 4 --teacher-accumulation 2 \
  --inference-steps 20 --inference-batch 4 \
  --meta-rank 16 --meta-every 25 --meta-warmup 250 --meta-weight 0.1 \
  --score-margin 0.25 --transfer-margin 0.25 \
  --student-scope attention_conv --ranks 16 32 --learning-rate 1e-4 \
  --precision bf16 \
  --checkpoint-steps 2500 5000 7500 10000 "$@"

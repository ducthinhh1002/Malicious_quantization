#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
.venv/bin/python -m py_compile scripts/queue_stage12.py scripts/select_pilot_strengths.py
bash -n scripts/run_pilot_worker.sh
bash -n scripts/preflight_stage12.sh
if ! tmux has-session -t quant-stage12-preflight 2>/dev/null && ! test -f metrics/stage12_preflight.json; then
  tmux new-session -d -s quant-stage12-preflight 'bash scripts/preflight_stage12.sh > logs/stage12_preflight.log 2>&1'
fi
if ! tmux has-session -t quant-stage12-queue 2>/dev/null; then
  tmux new-session -d -s quant-stage12-queue '.venv/bin/python scripts/queue_stage12.py > logs/stage12_queue.log 2>&1'
fi
tmux list-sessions

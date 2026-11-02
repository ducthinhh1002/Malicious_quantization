#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
test ! -f STOP_CORE_QUEUE
.venv/bin/python -m py_compile scripts/queue_core_seed0.py scripts/supervise_core_queue.py scripts/find_resume_checkpoint.py
.venv-student/bin/python -m py_compile scripts/generate_student.py scripts/launder_ctrlregen.py
bash -n scripts/run_core_condition.sh
if ! tmux has-session -t quant-core-seed0 2>/dev/null; then
  tmux new-session -d -s quant-core-seed0 '.venv/bin/python scripts/supervise_core_queue.py >> logs/core_supervisor.log 2>&1'
fi
tmux list-sessions

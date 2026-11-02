#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
.venv-student/bin/python -m pip install --no-deps torch-fidelity==0.3.0
.venv-student/bin/python -m pip install --no-deps scipy==1.15.3
.venv-student/bin/python -m py_compile scripts/student_utility.py scripts/queue_core_seed0.py scripts/audit_laundered_pairing.py scripts/summarize_core_seed0.py scripts/generate_student.py scripts/launder_ctrlregen.py
.venv-student/bin/python -c 'import torch_fidelity, scipy, matplotlib; print("Utility dependencies ready")'
bash -n scripts/run_core_condition.sh
if ! tmux has-session -t quant-core-seed0 2>/dev/null; then
  tmux new-session -d -s quant-core-seed0 '.venv/bin/python scripts/queue_core_seed0.py > logs/core_queue.log 2>&1'
fi
tmux list-sessions

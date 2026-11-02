#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
CUDA_VISIBLE_DEVICES=0 .venv-student/bin/python scripts/generate_student.py --unadapted --prompts prompts/student_eval.txt --out student_outputs/SD14_unadapted
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serum_scores.py --images student_outputs/SD14_unadapted --out metrics/SD14_unadapted_scores.npy
.venv/bin/python scripts/summarize_baseline_controls.py

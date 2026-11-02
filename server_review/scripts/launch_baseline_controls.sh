#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
if ! tmux has-session -t quant-sd14-control 2>/dev/null; then
  tmux new-session -d -s quant-sd14-control 'bash scripts/run_sd14_control.sh > logs/sd14_control.log 2>&1'
fi
tmux pipe-pane -t quant-sd14-control -o 'cat >> /root/workspace/serum/logs/sd14_control_pane.log'
if ! tmux has-session -t quant-c00 2>/dev/null; then
  tmux new-session -d -s quant-c00 'bash scripts/run_c00_baseline.sh > logs/c00_baseline.log 2>&1'
fi
tmux list-sessions

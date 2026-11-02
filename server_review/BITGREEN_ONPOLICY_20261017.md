# Deployment: on-policy supervision / low-capacity carrier

Server: `vm4`, H200. Immutable code snapshot: `/root/logs/bitgreen_onpolicy_1017`.
Python: `/root/logs/gate_env/bin/python`.
Source FP32 targets: `/root/logs/results/bitgreen_repair_shared_1017`.

| Pilot | tmux / basename | Result folder |
|---|---|---|
| Full 20-step terminal gradient, 4 train + 4 internal holdout, 60 updates, LR 1e-6 | `bitgreen_onpolicy_terminal_1017` | `/root/logs/results/bitgreen_onpolicy_terminal_1017` |
| Content-conditioned 148-parameter conv carrier, 64 train + 32 internal holdout, 200 updates, LR 3e-4 | `bitgreen_onpolicy_conv_1017` | `/root/logs/results/bitgreen_onpolicy_conv_1017` |

Launcher/log/exit paths: `/root/logs/<basename>.sh`, `.log`, `.exit`.
An exit file is written only when that process ends. Other users' tmux sessions
were left untouched. No previous run was deleted.

Checks: 26 unit tests passed on server; `bash -n` passed. First native probe
exactly matches old FP32 clean references (PSNR numerical floor 120, relative
target MSE 1). First full-gradient update completed: finite loss 1 before update,
gradient norm 49.1069, 84.9 seconds, peak allocated 7.018 GiB. This is an execution
check, not evidence of convergence. At that check, carrier cache was 61/96.

Next check: `result/training.jsonl`, `result/probe_fp32_0020.json` for terminal,
`result/probe_fp32_0050.json` for conv, and `.exit`/`complete.json`.
Conv results are carrier-only, never native teacher/student success. Terminal
4-image fit is only a trainability test; independent ordinary LoRA remains unrun.

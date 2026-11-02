# Constrained owner objective deployment

- Snapshot: `/root/logs/bitgreen_constrained_1017`.
- Run: `/root/logs/results/bitgreen_constrained_20261017_100852`.
- tmux: `bitgreen_constrained`.
- Launcher: `/root/logs/start_bitgreen_constrained.sh`.
- Log: `/root/logs/bitgreen_constrained.log`; exit file appears on termination:
  `/root/logs/bitgreen_constrained.exit`.
- Per-file source hashes are frozen in the run's `source.json`.

Default constrained objective, full 20-step gradients, teacher LR 1e-5,
250 updates per arm, 24 dB training penalty target, unchanged 22 dB evaluation
gate, ordinary clean/Plain/Robust student LoRA. Calibration reused with integrity
checks from the original completed BitGreen run. Original runs preserved.

Validation: 12 unit tests passed on server; Python/Bash syntax checked.
Two real robust-arm GPU updates passed with finite nonzero gradients and
13.32 GiB peak allocated memory. Full-gradient norm before clipping was large
(~33222 at step 1); clipping to norm 1 remains enabled. No claim of convergence
or scientific improvement from this two-step smoke check.

Deployment verified alive in asset validation with intended configuration.
See `survey/BitGreen_Constrained_Objective_VI.md` for objective and limits.

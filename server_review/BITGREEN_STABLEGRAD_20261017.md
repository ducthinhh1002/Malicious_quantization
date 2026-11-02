# BitGreen stable-gradient pilot deployment

Server: vm4. Immutable code snapshot: `/root/logs/bitgreen_stablegrad_release_1017`.
Tmux: `bitgreen_stablegrad`.
Launcher: `/root/logs/start_bitgreen_stablegrad.sh`.
Log: `/root/logs/bitgreen_stablegrad.log`; exit: `/root/logs/bitgreen_stablegrad.exit`.

Queued sequentially:
1. `/root/logs/results/bitgreen_stablegrad_1017_acc4`: SD1.4, K=4, accumulation=4, 100 updates per arm = 400 owner samples per arm. Standard student gate enabled.
2. `/root/logs/results/bitgreen_stablegrad_1017_acc1`: SD1.4, K=4, accumulation=1, 100 updates per arm = 100 owner samples per arm. Teacher-only ablation.

Both use frozen existing clean calibration; sample counts/time differ and must be reported. No edits to running SD1.4/SD2.1 full-gradient snapshots. Existing jobs were not stopped.

Validation: 16 unit tests pass in GPU environment; robust CUDA smoke 2 updates / 8 samples succeeds, peak allocated 13.365 GiB. First/second norm 6.512 / 8.213. This validates mechanics, NOT watermark effectiveness. Per-process allocator budget is 20 GiB; device is shared.

Audit artifacts (gradient/latent probes, smoke records) copied into `server_review/bitgreen_gradient_audit_1017`.
Research/limitations: `survey/BitGreen_Gradient_Stability_VI.md`.

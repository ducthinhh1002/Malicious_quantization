# SD2.1-base ablation deployment

- Code snapshot: `/root/logs/bitgreen_sd21_1017` (per-file hashes in run/source.json).
- Run: `/root/logs/results/bitgreen_sd21_20261017_101804`.
- SD1.4 comparison run: `/root/logs/results/bitgreen_constrained_20261017_100852`.
- tmux: `bitgreen_sd21`.
- Launcher: `/root/logs/start_bitgreen_sd21.sh`.
- Log: `/root/logs/bitgreen_sd21.log`; exit file `/root/logs/bitgreen_sd21.exit`.

Frozen `result/comparison_protocol.json` confirms identical prompts and objective/
update-budget fields. Model assets explicitly separate teacher SD2.1-base from
student SD1.4. The original SD1.4 job keeps its own unchanged code snapshot.

Verified public mirror: `sd2-community/stable-diffusion-2-1-base`, revision
`4e63672c03103b6c636b8fb4119ba982469b2955`. FP32 weights downloaded and LFS hashes
checked. Config verifies 512px, 1024 text/cross-attention width, epsilon prediction.

15 unit tests pass, including owner/student separation, 768-v rejection and
prompt/budget mismatch rejection. AST and shell syntax checks pass. Main run has
started generating its own 1,000 clean SD2.1 calibration images.

GPU integration smoke is isolated at `/root/logs/bitgreen_sd21_smoke`, log
`/root/logs/bitgreen_sd21_smoke.log`. Its temporary detector threshold is only a
mechanical gradient test; the main run does not read it and calibrates independently.
The two real robust-arm updates completed successfully with finite nonzero
gradients through all 20 steps and peak allocated memory 14.19 GiB. These smoke
metrics are not scientific results. Last main-run check: calibration 80/1000.

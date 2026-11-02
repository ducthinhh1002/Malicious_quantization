# BitGreen owner quality calibration — deployment

- Server: vm4, H200; tmux `bitgreen_quality`.
- Code snapshot: `/root/logs/bitgreen_quality_c4b3dba`.
- Archive SHA256: `c4b3dba47d6af29234affb61a8b096b478096b96f078606c578abcdea676378d`.
- Source completed run: `/root/logs/results/bitgreen_20261017_064252_359192`.
- New run: `/root/logs/results/bitgreen_quality_20261017_082439_942427`.
- Launcher: `/root/logs/start_bitgreen_quality.sh`.
- Log: `/root/logs/bitgreen_quality.log`; eventual exit status: `/root/logs/bitgreen_quality.exit`.
- Heavy artifacts: `/root/logs/runtime/bitgreen_artifacts/bitgreen_quality_20261017_082439_942427`.

Verified local and server unit tests: 8 passed. Bash syntax and Python syntax passed.
GPU integration smoke: two fresh design images at owner strength 0.4, real frozen
BSQ reader, selection/fallback, native export/reload. PNG hashes matched exactly
between selection-time fused owner and exported native owner. These two images
are an engineering check, not evidence of stable quality or watermark transfer.

Main run was verified alive in `select_plain`. No new main-run scientific results
were available at this deployment check. Student training runs automatically only
after the separate teacher gate passes. Prior completed experiment is preserved.

See `survey/BitGreen_Quality_Calibration_VI.md` for fresh data splits, fixed
calibration, owner-only intervention, quality targets and interpretation.

## Metadata repair and resume

Plain sweep completed, but `select_robust` stopped: Diffusers reordered the
set-like `_use_default_values` scheduler metadata across Python processes.
Fixed `freeze_generation` compares this list in sorted order while retaining
the original metadata file; all other identity fields must still match exactly.
Nine tests passed, including rejection of changed seed, prompt order and beta.

Resumed in tmux `bitgreen_quality` from `/root/logs/bitgreen_quality_metadatafix`
using `/root/logs/resume_bitgreen_quality.sh`. Run remains unchanged. The old
source manifest and exact two-file patch hashes are saved in
`result/source_before_metadata_fix.json` and `result/source_metadata_fix.json`.
Original exit status 1 is retained at `/root/logs/bitgreen_quality.first_exit`.

Plain design sweep (64 images): strength 1.0 gives TPR 100%, PSNR 17.39;
0.65 gives 79.69%, 19.81; 0.4 gives 14.06%, 22.68; 0.3 gives 4.69%, 24.38.
No candidate satisfies both selection targets. These are design results, not
held-out teacher or student results. Scalar attenuation alone has not solved
the signal/fidelity trade-off for Plain.

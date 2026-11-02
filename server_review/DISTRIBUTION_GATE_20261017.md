# Distribution fingerprint teacher pilot deployment

- Server: vm_4, H200. Started after GPU integration smoke passed (exit 0).
- Code: `/root/logs/distribution_gate_v2`
- Archive SHA256: `5af32de862ee6529579580a9a34f125f385387b618fca445ed90656177b2ce33`
- Runtime: `/root/logs/gate_env/bin/python`
- Tmux: `distribution_gate`
- Entrypoint: `/root/logs/start_distribution_gate.sh`
- Log: `/root/logs/distribution_gate.log`
- Exit status when complete: `/root/logs/distribution_gate.exit`
- Run: `/root/logs/results/distribution_gate_20261017_001814`
- Heavy artifacts: `/root/logs/runtime/distribution_artifacts/distribution_gate_20261017_001814`
- Launcher metadata: `/root/logs/distribution_current.json`

Three teacher arms, 500 updates each: control, clip, unet. Teacher-only; no
student training or inheritance claim. Shares 2048 pool images, 512 fit images,
1000 clean calibration and 256 evaluation images. Updates reports after each
completed arm. Reports use both fixed carriers, wrong keys and same-step
unweighted control; query statistics are exploratory, not calibrated model FPR.

Local and remote numerical unit tests passed. Real H200 integration smoke
verified feature extraction, carrier fitting, weighted/control update, native
adapter merge/export/reload, generation and report writing. Smoke fixture reuse
is explicitly non-scientific and its numbers must never enter paper tables.
Smoke completion log: `/root/logs/distribution_smoke_report.log`, exit file 0.

Initial native merge check failed with TF32 enabled. Disabling TF32 without
changing equivalence tolerance produced max error 5.0664e-6, RMSE 1.0614e-6.
Production GPU stages disable TF32 and use explicit BF16 autocast for training.

Last launch check: generating fit images, 12/512; H200 6321 MiB used, 99% utilization;
tmux alive, no exit file. This is a launch observation, not a completion claim.

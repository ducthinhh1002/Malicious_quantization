# Centered teacher pilot deployment

- Run: `/root/logs/results/centered_teacher_20261017_060553`
- Heavy artifacts: `/root/logs/runtime/distribution_artifacts/centered_teacher_20261017_060553`
- Code: `/root/logs/centered_teacher`
- Archive SHA256: `adec503cf4f694a3c84c2d4a30781d9a6838434fa4de7b8b1a06c729faa18fee`
- Tmux: `centered_teacher`
- Log: `/root/logs/centered_teacher.log`
- Exit file: `/root/logs/centered_teacher.exit`
- Entrypoint: `/root/logs/start_centered_teacher.sh`
- Python: `/root/logs/gate_env/bin/python`
- Launcher metadata: `/root/logs/centered_current.json`

Reuses verified design data and frozen detectors from
`/root/logs/results/distribution_gate_20261017_001814`. Starts fresh base teacher
weights for control/CLIP/UNet. Signed residual targets remove ordinary SFT
gradient at the base. Beta2, signal gain2, all8 views per prompt/batch, common
noise/timestep per prompt, explicit unconditional/CFG preservation. Checkpoints
250/500/1000. Native export, frozen image thresholds, unchanged quality gates.
Teacher allocator cap40 GiB. Existing unrelated GPU processes were left running.

Six local and remote numerical tests passed. H200 smoke passed with exit0:
control loss/gradient exactly zero for two steps; UNet step2 grad norm0.00924,
target RMS0.22167; merged prediction max error4.4107e-6. These are implementation
checks, not evidence of watermark effectiveness or student inheritance.

No students launched. Reused design evaluations are adaptive development data,
not an independent confirmation set. Current run started after smoke success.

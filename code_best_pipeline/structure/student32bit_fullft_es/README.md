# Full UNet fine-tuning with held-out validation and early stopping

Server `.21`: `/var/lib/structure32_student_fullft_es`, GPU1.
Migrates at a completed1250-step periodic checkpoint from the original full-FT
run, preserving dense UNet, AdamW, LR scheduler and CPU/CUDA RNG state. Original
run and identity are retained. Training data/objective, BF16, batch4 and LR
schedule remain unchanged; no watermark loss, LoRA or augmentation is added.

Validation uses1000 held-out teacher32 images and2 fixed denoising draws per
image, seed314159. Cache is built only from validation images; the same draws
are reused at every check. Validation restores training mode and CPU/CUDA RNG.
Checks occur at the migration checkpoint and every500steps thereafter.

Early stopping: min_delta1e-4 absolute MSE; patience4 checks, counted from2500
steps; maximum12500. This detects lack of validation improvement, not a proof
that overfitting is absent. Each validation saves full resumable state and logs
the loss. Best-validation checkpoint is used for final generation/evaluation,
not the last checkpoint. Test images/bit metrics never select the checkpoint.

`run/metrics/train_loss.jsonl`: full-precision values from Accelerator logging
after migration. Earlier rounded train loss remains in the old run's metrics.
`run/metrics/validation_loss.jsonl`: step, loss, best step/loss, bad checks and stop.
`run/early_stopping.json`: current policy and selected best full-state directory.
`deployment/migration.json`: boundary and migration state.

Validation-state checkpoints are kept separately from periodic checkpoint
retention so a best state cannot be removed by the latest2 checkpoint policy.
Results report the actual selected step. Statistical key acceptance remains
>=24of32 under the documented independent-bit theoretical FPR model.

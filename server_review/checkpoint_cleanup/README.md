# Checkpoint cleanup on vm4

Completed: 2026-10-18T09:26:09.706588+00:00

- Deleted 82 weight files from 28 explicitly reviewed historical checkpoint folders, one file at a time.
- Logical size removed: 112.12 GiB.
- Observed available-space increase: 112.10 GiB.
- Space available afterward: 272.78 GiB.
- Retained every active LALT artifact, hub/base models, environments, prefix caches, source code, JSON/CSV reports, image outputs.
- Retained strong-signal BitGreen pair, best quality-robust diagnostic, representative stablegrad-acc1 robust negative, centered step500 control/clip/unet triplet, and small LCF comparison checkpoints.
- The deleted exports cannot be reloaded without retraining/recreating their weights; their configs and result evidence remain.

[Plan with per-checkpoint evidence](checkpoint_cleanup_20261018_092513Z.plan.json)

[Per-file deletion log](checkpoint_cleanup_20261018_092513Z.plan.deleted.jsonl)

[Space summary](checkpoint_cleanup_20261018_092513Z.plan.summary.json)

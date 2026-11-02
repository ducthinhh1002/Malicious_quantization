# Watermark evaluation policy (user instruction, 2026-10-29)

For new watermark experiments, report bit accuracy and exact full-key recovery
after aggregating groups of1,8,32,64,256 images. Do not use TPR as the evaluation metric.

Average the normalized continuous carrier coordinates across images in each group,
then decode each bit from its sign. A full key is recovered only when all payload bits
match. Declare zero ties (`>=0` maps to+1), group construction, sample counts and seeds.
Distinguish per-image bit accuracy, group bit accuracy and a single aggregate over all
images. The latter is not a substitute for the grouping table.

Current implementation: `structure32_student12500/key_metrics.py`. It uses100 seeded
random partitions without replacement within a group, discards incomplete groups,
and averages partition metrics. Repeated partitions do not create new independent data.
Keep quality metrics (LPIPS/FID/KID/CLIP) and validation-loss checkpoint selection.
Clean-score calibration remains internal to the existing teacher loss where required.
Update2026-10-29: students now use the fixed12,500-step endpoint by user request.
For32bit/rank32 and32bit/rank4, skip validation checkpoint generation, loss comparison,
utility evaluation and checkpoint selection. Keep final test key-recovery and quality
evaluation, and periodic full training-state checkpoints for recovery.

Historical reports retain their historical definitions; recomputed grouped metrics
must be explicitly labeled with the new evaluation protocol.

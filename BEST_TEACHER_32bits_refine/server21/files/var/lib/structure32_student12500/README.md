# Full Structure 32 bits -> student 12,500

Server `/var/lib/structure32_student12500`, GPU1, process name `VLLM:Enginecore`.

Reference Structure12500 uses **8 payload bits**, not4; LoRA rank32 is unrelated to payload length.
This experiment keeps the original Full Structure six refinement modules byte-for-byte,
the kappa1 objective, PNDM50 CFG7.5, seed0, base models, 5,000/1,000/1,000 data budget,
600 refinement updates, independent SD1.4 rank32-conv student and12,500-step cooldown schedule.
It does not substitute Nash refinement or the autoregressive student.

Reuse the existing clean-only PCA/stability fit, all clean RGB/captions, base checkpoints
and unadapted SD1.4 images via immutable hardlinks. Select32 PCs with the original RNG rule
from the top32 stable PCs (the8bit reference used a top16 candidate pool). Keep PC range8..64.
Recalibrate clean score thresholds for the32bit teacher loss. These two changes are necessary
to make the original algorithm accept32 bits; this is not literally a one-number ablation.
Train kappa1 teacher from clean initialization for3,000 updates, then refine600; do not warm
start from8bit watermarked weights or use8bit watermarked training images.

## Evaluation policy

Primary metrics are **bit accuracy** and **exact full-key recovery rate** for groups of
1,8,32,64,256 images. Average the normalized continuous carrier coordinates within each group,
then decode signs with `>=0 => +1` (zero tie rule declared explicitly). Compare all32 bits with
the fixed owner key. Full key is recovered only if all32 bits match. This is distinct from
the historical `bit_accuracy` scalar that decoded a single aggregate over all1,000 images.

For group size1 use all1,000 images once. For larger groups use100 seeded permutations,
partition each without replacement, discard the remainder, compute group metrics and average
over partitions. There is no repeated image within a group; reused data across partitions
does not increase the independent sample size. Partition SD is grouping variability, not CI.
Keep all1,000 images aggregated as an additional diagnostic, not the primary result.

Do not compute/report TPR in teacher/student/validation summaries. Frozen clean thresholds
remain an internal component of the unchanged threshold_cfg_v1 teacher objective.
Keep LPIPS/FID/KID/CLIP quality metrics. Student checkpoint selection remains minimum
validation denoising MSE; payload metrics/test/utility never select checkpoints.

Full teacher optimizer/RNG checkpoints are kept at original intervals; student full state
every1,250 steps. No quality/detection stopping gates. Supervisor resumes completed stages.

Progress: `deployment/status.json`, per-stage logs. Final payload tables are in
`runs/structure32/result/teacher_key_metrics.json` and the student result summaries under
`student12500/runs/structure32_student12500_seed0/student12500/result/`.

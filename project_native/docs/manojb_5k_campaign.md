# Manojb / 5000-image kappa-1 revision

Requested order: `k1_scale085`, `k1_evolution`, `k1_scale100`, `k1_scale095`.
Driver: `manojb_5k_campaign.py`. VM4 output: `/root/logs/threshold_quality_20261022/kappa1_manojb_5k`.

The preceding `kappa1_campaign` was stopped before student training. Its artifacts
remain available. The new run is separate: it does not overwrite old frozen
configurations, images or reported results.

## Model and data

- Teacher: `Manojb/stable-diffusion-2-1-base`, pinned revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`.
- All selected model weights **and configuration files** match the SHA-256 hashes
  of the previously used sd2-community snapshot. Mirror naming is therefore not
  an explanation for the previous quality gap. Verified identical weights are
  hardlinked into the requested pinned repository directory.
- Original disjoint COCO split function, seed 0: train 5000, calibration 5000,
  test 1000, validation 1000. Student checkpoint image evaluation also uses 1000.
- Increasing train size shifts the original split offsets. Reusing the old
  calibration would overlap new training data; therefore PCA/carrier and both
  null thresholds are recomputed. Key seed, original carrier construction rule,
  FPR 0.01, sampler and 50 inference steps stay fixed. The new detector is shared
  unchanged by all four branches. This is not reuse of the historical key file.
- Clean-image reuse requires identical model hashes, prompts, seeds, batch
  composition/order, dtype, CFG, steps and resolution. No watermarked image or
  score from the previous carrier is reused.
- Fresh teacher kappa=1: original `threshold_cfg_v1` objective, rank 32, 3000
  updates, batch 4, microbatch 4, BF16, full checkpoints every 500 updates.
  Microbatch was increased from 1 to 4 at the user's request before carrier
  construction or teacher training started. Completed clean images are retained;
  previous configuration and code manifests are archived under `revisions/`.
- Students: original `lora_r32_conv_wm` trainer, rank 32 attention+convolution,
  batch 4, FP16, learning rate 1e-4, 7500 updates. Save 2500/5000/7500; select by
  held-out denoising MSE. No quality cutoff terminates training.

The source teacher preparation necessarily precedes the ordered student trials.
“Original” means the fresh kappa-1 teacher at scale 1.0, not a recovered historical
checkpoint that already attained detection 99.70% / FID 62.676.

## Research choice and implementation

[MPaGE, Ha et al., with Huynh Thi Thanh Binh (2025), sections 4.2–4.5](https://arxiv.org/html/2507.20923v1)
uses a Pareto-front grid and evolutionary variation to maintain diverse heuristic
candidates. We adapt the grid/elitism idea to six adapter coefficients; we do not
reproduce its LLM program generation, feedback reflection or semantic clustering.

The six variables are down/mid/up crossed with attention/convolution. Only LoRA B
factors are scaled, leaving A unchanged. Levels are 0.80, 0.85, 0.90, 0.95, 1.0.
Objectives: minimize paired LPIPS and CLIP drop, maximize the tenth-percentile
detector-score margin. The margin matters because saturated teacher TPR can hide
loss of watermark strength and does not establish transfer to a student.

Eight common initial candidates (0.85, 1.0 and six single-group restorations)
are measured on the same 64 validation prompts. Pareto-grid crossover/mutation
and a random-search control each receive 16 additional evaluations. The control
is diagnostic and does not secretly contribute candidates to the evolutionary
selection. Measurements are cached; two independent candidate workers may run
concurrently on GPU 0. No test score or prior branch test result enters search.

Up to three Pareto candidates plus 0.85/1.0 controls are reevaluated on all 1000
validation prompts. Prefer candidates losing no scale-1 positives and retaining
at least 90% of its positive p10 margin; choose the lowest LPIPS, breaking ties
with CLIP and margin. If none qualify, record failure and still train the best
detection candidate. A control winning the search is reported as a fallback,
not an improvement. An exact already-trained vector is reused to avoid duplicate
7500-step training. Final quality and transfer claims use the separate test set.

[MF-LTGA, Binh et al. (2020), section 4](https://arxiv.org/html/2005.03090v1)
was also reviewed. Its task-specific linkage trees and transfer are less directly
suited to this single six-variable task and limited population. We do not label
LPIPS and detection as two multitasking problems or claim to implement MF-LTGA.

## Integrity and storage

Frozen campaign/code/model/data provenance accompanies the new run. Unit checks
cover detection-preserving selection, explicit failure reporting, bounded and
reproducible proposals, disjoint splits and the requested execution order.

Old reconstructible merged exports were released only after checking their full
hash manifest and retained adapter hash. All original adapters, full checkpoints,
images and metrics remain. Reconstruction proofs and recipes are recorded in
`storage_revisions/20261024_manojb_5k.json`; free space rose to approximately 56 GiB.

Results include detection, margin, LPIPS, CLIP, FID and KID. Final paired reports
use the new scale-1 student. One training seed remains a limitation; image
bootstrap intervals do not quantify training-seed uncertainty.

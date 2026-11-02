# Scale 0.85 detection analysis and controlled recovery

Update after the scale-1 evaluation completed: student scale 1.0 has test
TPR 75.0%, LPIPS 0.5504473, FID 54.44424, KID 0.00166651, CLIP 0.7876815.
Scale .85 improves paired LPIPS by only 3.05% while losing 32.3 percentage points
of detection. The document below records the earlier launch-time analysis;
the current research review and paired comparison are in
[transfer_research_20261024.md](transfer_research_20261024.md).

Status at 09:24 ICT, 24 October 2026. The user cancelled scale 0.95. The original
scheduler PID 394276 was stopped; the already-running scale-1 evaluation PID
623931 is allowed to finish. Cancellation is recorded in
`kappa1_manojb_5k/cancelled_jobs.json`. Do not relaunch the old scheduler.
Replacement driver: `scale085_recovery.py`; outputs under
`/root/logs/threshold_quality_20261022/scale085_recovery_20261024`.

## What the measurements establish

All following teacher comparisons use the same 1000 validation prompts/seeds,
same freshly trained kappa-1 checkpoint and fixed detector threshold.

| Teacher adapter | LPIPS | CLIP drop | Detection | p10 score minus threshold |
|---|---:|---:|---:|---:|
| Uniform 1.0 | 0.4365073 | 0.00341065 | 98.9% | 0.3144539 |
| Uniform 0.85 | 0.3931558 | 0.00188880 | 94.3% | 0.0721064 |
| Restore down-conv 1.0 / up-conv 0.9; others 0.85 | 0.4007470 | 0.00178715 | 96.1% | 0.1263073 |

Uniform 0.85 reduces paired teacher LPIPS by 9.93%. Paired mean difference
(1.0 minus 0.85) is 0.0433515, image-bootstrap 95% interval
[0.0411950, 0.0456164]. This measures closeness to the clean counterpart, not a
human-rated quality percentage. Student comparison against 1.0 awaits its test
evaluation; the teacher result must not be relabelled as a student result.

On the separate existing test panel, uniform 0.85 teacher detection is 92.6%,
mean score 1.21267, p10 0.88981. Its student detects 42.7%, mean 0.80589,
median 0.80231, p10 0.44851, below the student threshold 0.85413. The clean student
detects only 0.8%, supporting the intended null behaviour rather than an obvious
threshold bug. The teacher null threshold is separately calibrated at 0.85093.

Student 0.85 test quality: LPIPS 0.5336590 against paired unadapted SD1.4,
FID 54.56305, KID 0.00146374 against clean teacher images, CLIP 0.7874365.
These LPIPS references differ from teacher LPIPS; do not compare their absolute
values as if both used the same reference model.

## Why detection decreased: evidence versus hypotheses

1. **Teacher attenuation reduces the safety margin.** Uniform scaling loses
   about 77.1% of the teacher p10 margin, even though aggregate teacher TPR still
   looks high. The student loses further signal during ordinary denoising
   fine-tuning, placing its score distribution around/below the frozen threshold.
   Teacher detection does not guarantee transfer to an independently trained SD1.4.
2. **The historical 99.70% reference used selected kappa=2**, as recorded in
   `t2ifing_radioactive/result_vast/RESULT.md`. The current teacher is kappa=1,
   followed by 0.85 attenuation. Historical per-run checkpoint/carrier recovery
   remains unavailable. Its student metrics are not measurements of a kappa-1
   student. The different carrier and data split also prevent an isolated causal
   comparison with that historical run.
3. **Limited student exposure is a testable hypothesis.** At batch 4, 7500 updates
   over 5000 images equal six data passes. The previous 1000-image setup had thirty
   passes at the same update count. Validation detection still rises across
   2500/5000/7500: 39.1/42.6/48.2%, with mean scores 0.7504/0.7962/0.8405.
   This does not prove that more updates will solve the problem; the curve may
   saturate, and image quality may deteriorate.
4. **No skipped optimizer updates explain the failure:** the audit records
   7500 attempted, 7500 applied, zero skipped. Native training images number 5000.
   The student trainer retains its original MSE objective and configuration.
5. **“Bit accuracy 100%” is aggregate recovery.** `SemanticCarrier.bit_metrics`
   takes the sign of coordinates averaged over the dataset. It is not per-image
   detection and cannot contradict the 42.7% threshold-based TPR.

The Manojb and preceding sd2-community snapshots have identical verified model
weights and required configuration files, so mirror naming does not explain the
observed quality gap. The detector threshold is not lowered to inflate TPR.

[SIREN, Li et al.](https://arxiv.org/abs/2410.10437) explicitly targets learnability
of protection signals under personalization. This motivates separating teacher
visibility from student transfer. It does not prove these specific interventions
will work, and this experiment does not implement SIREN's coating method.

## Implemented trials

- **Convolution restoration:** vector `[0.85,1.0,0.85,0.85,0.85,0.9]` in ordered
  groups `[down_attention,down_conv,mid_attention,mid_conv,up_attention,up_conv]`.
  This measured teacher candidate costs 1.93% more LPIPS than uniform 0.85 while
  improving teacher p10 margin by approximately 75%. Student budget stays 7500,
  so its transfer comparison uses the same budget. This is an exploratory
  recovery candidate, not a candidate that preserves every scale-1 detection.
- **Exposure ablation:** exactly the original uniform-0.85 images, cache and seed,
  but 22500 student updates (18 passes), snapshots 7500/15000/22500. All 7000
  generated teacher images and validation encodings are reused with checked
  manifests. The old snapshots lack optimizer/RNG state, so the new trial starts
  from clean SD1.4, not a falsely claimed exact optimizer resume. Only the training
  budget changes. Increased-budget results must not be labelled a teacher improvement.

Both use the same key/carrier/null thresholds, sampler, rank-32 attention+conv
student, original loss, learning rate and batch 4. Checkpoint selection remains
validation denoising MSE only. Quality stopping gates stay disabled; failures and
quality/detection tradeoffs are reported. The long-budget trial runs alongside the
retained control evaluation; convolution restoration starts after that evaluation
finishes, keeping at most two heavy jobs in this replacement schedule.

Three tests pass locally and on VM4: unchanged training command except budget,
the intended grouped adapter change, and reuse/selection-before-test ordering.
Server preflight verifies 5000/1000/1000 reused images, the validation-cache hash,
and unchanged student-threshold hash.

This is an adaptive follow-up using an already observed test panel for comparison.
Any apparent success needs confirmation on a fresh holdout and additional training
seeds before a robust improvement claim.

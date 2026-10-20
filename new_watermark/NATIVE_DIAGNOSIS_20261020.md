# Diagnosis of the completed native UNet run

Inspected via SSH on 2026-10-20. Run:
`/root/logs/results/native_unet_20261019_162350_r16r32`.
Source snapshot: `native_unet_20261019_production`, commit `679931a`.
Raw reports and frozen feature logits were copied locally under
`output_analysis/native_unet_20261019_162350_r16r32`.

## Observations

| Generator | Detected / 1000 | Detection rate | Bit accuracy | Exact 8-bit accuracy |
|---|---:|---:|---:|---:|
| Clean base / unadapted student | 3 | 0.3% (false positives) | 32.04% | 0% |
| Merged marked teacher | 72 | 7.2% | 52.50% | 0% |
| Attention + convolution LoRA rank 16 | 19 | 1.9% | 40.23% | 0% |
| Attention + convolution LoRA rank 32 | 19 | 1.9% | 41.48% | 0% |

Teacher TPR Wilson 95% CI: 5.76–8.97%; each student: about 1.22–2.95%.
The score combines presence and soft bit agreement, so passing detection is
NOT equivalent to decoding every bit correctly. The correct-key threshold is
-1.8832511608130547, frozen before test; nominal FPR .01 is split over two keys.

Teacher paired PSNR is 18.4475 dB, below the declared 25 dB threshold.
FID versus **clean teacher generated images** is 59.6287, KID .0075823.
Student FIDs are 49.9258 / 50.8524 against that same reference. These are
not FID against real COCO and cannot be compared directly with paper headline FIDs.

No NaN-skipped teacher steps: **2000/2000 applied**. 1998/2000 finite
gradient norms exceed the clipping threshold of 1. Clipping alone is not a bug.
During the last 500 steps, mean image MSE is .01691 (budget .001),
LPIPS .36671 (budget .1), quality loss 74.411 and detection-margin loss 1.861.
Loss magnitudes alone do not establish gradient magnitudes or directions.

On pilot evaluation, teacher TPR at steps 500/1000/2000 is
0.39% / 0.78% / 4.69%; PSNR 19.24 / 18.49 / 18.35 dB.
All three corresponding 500-step pilot students have TPR **0/256**.
Step 2000 was selected only by the declared diagnostic fallback.
The report correctly records `valid_design_winner=false`, `teacher_valid=false`.

Both final students select step 10000 by validation denoising MSE, as intended.
Validation detection is 6.25% / 5.47% on 128 images; final test is 1.9% each
on a different 1000-image set. This difference is not itself evidence of a bug.

## Reader findings

For little-endian key 221, target bits are `[1,0,1,1,1,0,1,1]`.
Teacher per-bit accuracies are
`[.212,.111,.863,.786,.821,.183,.670,.554]`.
Mean presence logit shifts from -6.045 on clean images to -.102 on teacher
images, but three bit positions remain mostly wrong. Neither teacher nor
student decodes a complete key in this test.

This is consistent with learning a generic reader-recognizable perturbation
without reliable key encoding. It does not prove the reader is implemented
incorrectly or that these bit positions are intrinsically unlearnable.
The reader was trained for an earlier latent-transport carrier and frozen;
compatibility with new native-UNet perturbations is an experimental limitation.
On a single fixed key, clean-image bit accuracy need not equal 50%.

## What plausibly went wrong

1. The teacher positive control failed. Student weakness cannot isolate a LoRA
   capacity bottleneck when the data source is already poorly marked.
2. The teacher changes all 20 denoising steps while receiving output gradients
   through only four. This is an intentional biased estimator, not a syntax bug.
   It can miss the effect of changed earlier steps, but a controlled ablation is
   needed to establish whether it caused this failure.
3. Image losses did not control drift and there was no clean denoiser penalty.
   Lower LR/warmup and a prediction-preservation term are reasonable tests.
4. Detection confidence improved more than key decoding. Mean-bit losses can
   conceal hard bits. Add a symmetric smooth worst-bit term and report per-bit
   accuracy rather than treating presence confidence as ownership.
5. Seventy two-step transfer-proxy observations have mean gain .000698;
   24 are nonpositive. This proxy is a predicted-x0 score, not downstream TPR.
   There is no demonstrated return for its substantial memory cost here.

## Research and implemented response

- [DRaFT, ICLR 2024](https://arxiv.org/html/2309.17400v2) evaluates full and
  truncated gradient methods; full backprop can explode. Added an explicit
  full-gradient ablation (`--teacher-tail-steps 0`) with a finite-difference
  regression check; retained tail-4 as the default.
- [AquaLoRA, ICML 2024](https://arxiv.org/abs/2405.11135) and its
  [official PPFT code](https://github.com/Georgefwt/AquaLoRA/blob/master/train/ppft_train.py)
  preserve base noise predictions while learning watermarked latents.
  Added a simpler same-input, clean-denoiser preservation loss across uniformly
  sampled timesteps. This is NOT an AquaLoRA reproduction or a radioactivity proof.
- [HMARK](https://arxiv.org/html/2512.00094v1) motivates investigating signals
  in semantic h-space instead of only texture. Its main radioactivity evaluation
  uses pixel-space DDPM on CelebA-HQ, unlike our text-conditioned SD1.4 study.
  Its strong numbers cannot be assumed transferable to this setup.
- Default native teacher LR now warms up over 100 updates to 1e-5 and decays
  to 1e-6. Added the bit-tail loss and separate reader/training audit reports.
  Disabled the costly two-step transfer proxy by default; it remains opt-in.
  Student ranks, 10000-step horizon, frozen detector and quality gates are unchanged.

All changes are experimental. No improved TPR is claimed without a new run.
The failed metrics continue to be reported rather than aborting the experiment.
The GPU was running another project's job during inspection; no new full GPU
experiment was launched and that job was not modified.

Validation: 20 native/shared integration tests plus a new four-update teacher
loop test passed on CPU, and 39 shared BitMark/SERUM regressions passed.
The teacher loop test exercises a real tiny Diffusers UNet with PEFT adapters,
full sampling gradients, prior loss, bit-tail loss, stochastic VAE-cycle path,
snapshot serialization and audit reporting; its VAE and perceptual metric are
small stand-ins. This is not a full-size H200 runtime or effectiveness test.
Shell syntax, Python compilation, whitespace checks and wrapper defaults passed.
The union of both prior studies excludes 17,700 COCO IDs, leaving 22,804 for
the new 8,920-image split allocation; this was checked against the server cache.

## Next evaluation

First compare teacher strength/quality at the declared snapshots; run remaining
stages regardless of quality failure, as requested. Use ablations of LR, prior,
bit-tail and trajectory depth to attribute any improvement. Freeze thresholds
and choose checkpoints using only development/validation data.

Use a fresh final test: the October 19 outputs are now development evidence.
Keep an independent matched clean-trained LoRA control in a future confirmatory
study; differences from the unadapted base alone do not isolate inheritance.
Do not claim a novel watermark or reliable LoRA radioactivity from this run.

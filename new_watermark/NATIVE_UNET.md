# Native UNet + frozen 8-bit reader experiment

Run `bash run_native_unet.sh` in the prepared environment. On vm_4 the wrapper
uses the recorded step-20,000 LALT checkpoint, pinned SHA256
`bce50d0a1a450ae315a90c4bfcfafe22c93b6bb79ceea85980a9ddef357e8eb0`.
On another machine override `NATIVE_OWNER_CHECKPOINT`, `NATIVE_PRIOR_PROMPTS`
and `NATIVE_DEVELOPMENT_PROMPTS` with local copies of the recorded artifacts.
The reader architecture is kept exactly compatible with `TransportReader` in the
server snapshot `lalt_stable_reader_20261018_111112Z/…/latent_transport.py`.
The old latent transport weights are **not** loaded into the new teacher.

## Implemented protocol

- SD1.4 UNet, attention + convolution rank-32 teacher adapter, later fused into
  ordinary UNet weights. No inference key, latent transport or custom pipeline hook.
- Fixed development key 221, complementary wrong-key control 34. The reader is
  frozen, not retrained on generated students. This is not unseen-key evidence.
- Exclude all COCO image IDs in the prior study's prompt manifest from every new
  split. Train/selection/calibration/test splits are distinct; captions/noise paired
  across compared outputs. The wrapper also excludes the October 19 run now
  used for development, through `--exclude-additional-prompts`.
  The old reader still carries prior-study knowledge.
- Teacher samples the complete current-adapter DDIM trajectory (20 steps); only
  the last four steps retain gradients. Its prefix is recomputed, not taken from
  cached clean-model latents. This is biased truncated backprop, not full BPTT.
  `--teacher-tail-steps 0` enables a full-gradient ablation; this is not assumed
  superior (DRaFT reports exploding full-chain gradients).
- Loss: bit BCE + softplus margin above the frozen detection threshold + .5
  correct-vs-wrong-key margin + paired clean-image MSE/LPIPS quality budgets.
  Quality uses .001 MSE and .1 LPIPS budgets, small in-budget penalties and stronger
  per-example penalties when exceeded. Inputs use an RGB8 straight-through view.
  The updated wrapper adds a smooth worst-bit BCE (weight 1, temperature .25,
  symmetric across bit positions), so easy bits cannot entirely mask difficult
  ones. This changes training only, never the calibrated detection score.
- A clean-denoiser trust region samples a uniform DDPM timestep and compares
  both unconditional/conditional predictions to the frozen base at the SAME
  noisy clean latent. Relative prediction MSE is normalized per branch. For
  `r = relative_MSE / .01`, the penalty is `.01*r + .1*relu(r-1)^2`, weight 1.
  It is backpropagated separately after the sampling loss to limit peak memory.
  This is ordinary preservation regularization, not AquaLoRA's shifted-latent PPFT.
- Every fourth update adds .25 watermark loss through a stochastic VAE posterior
  round trip. This only addresses the VAE bottleneck, not complete student learning.
- Optional `--meta-weight 0.1`: after 250 updates, every 25th update adds short-horizon LoRA-transfer
  loss through two ordinary AdamW denoising updates of a fresh rank-16
  attention+convolution surrogate. The clean rollout uses the same initialization
  and noise. The disjoint predicted-x0 probe is explicitly an approximation.
  Default is now **0**: the completed experiment's 70 proxy evaluations had
  mean gain only .000698 and all three real 500-step pilots detected 0/256.
  This is not evidence that meta-learning in general fails.
- Teacher 2,000 updates, snapshots 500/1,000/2,000; FP32, two microbatches of one,
  norm clipping 1. LR warms up for 100 steps to 1e-5, then cosine to 1e-6.
  This lower LR and the regularizer are hypotheses motivated by observed drift,
  not experimentally established improvements.
- Three 500-step marked-data pilot students + one matched clean pilot select the
  teacher by transfer under native teacher/quality checks. Failed checks still
  select a diagnostic fallback and run final students. Corrupt assets/runtime
  exceptions remain errors, never disguised as scientific results.
- Final student attention+convolution ranks 16 and 32, each 10,000 updates on 5,000
  selected-teacher images, LR 1e-4, constant schedule, batch 4. Checkpoints
  2,500/5,000/7,500/10,000 selected by held-out denoising MSE, never test detection.
- The same frozen reader measures TPR, bit accuracy and wrong-key detection.
  Thresholds are calibrated on 1,000 clean images; nominal total FPR .01 is split
  between two keys (.005 each), using a conservative order statistic. Empirical
  FPR and Wilson intervals are reported, not assumed to equal the nominal target.
- FID/KID and CLIP run after generation. No final clean-trained students are added
  silently: final unadapted comparisons cannot isolate generic fine-tuning effects.

Reports live in `RUN/result/README.md`, `summary.csv`, `report.json`,
`checkpoint_metrics.csv`, `teacher_selection.json`, and `teacher_strength.json`.
Per-update losses, applied/skipped updates and GPU peak are in
`RUN/teacher/training.jsonl`. Export validation is `RUN/models/teacher_STEP/export.json`.
`result/teacher_training_audit.json` reports loss windows, clipping, skipped
updates and proxy gains. `result/reader_diagnostics.json` separates presence
logits, per-bit accuracy and exact-message accuracy. Neither file changes selection.
Weak metrics never stop the study. A completed run does not mean successful transfer.

## Evidence behind the optimization choices

[AlignProp](https://arxiv.org/abs/2310.03739) establishes the practical use of
differentiable output rewards with diffusion LoRA and checkpointing. Our fixed
tail, paired quality objective, frozen watermark reader and short learner proxy
are this experiment's choices, not a claimed reproduction or proof of radioactivity.
[Diffusers LoRA documentation](https://huggingface.co/docs/diffusers/main/training/lora)
uses 1e-4 as the student LR default; it does not establish optimality for this dataset,
convolution scope, or watermark. All extra teacher/pilot compute must be reported.

[DRaFT](https://arxiv.org/html/2309.17400v2) supports both full and truncated
sampling gradients and specifically warns that the full gradient can explode.
[AquaLoRA's official PPFT implementation](https://github.com/Georgefwt/AquaLoRA/blob/master/train/ppft_train.py)
provides a useful example of preserving base predictions while implanting a
watermark, but uses shifted latent inputs and a pretrained watermark encoder.
Our same-input regularizer does not reproduce that mechanism.

For an ablation against the October 19 optimization settings (use a new output):

```bash
bash run_native_unet.sh --teacher-lr 5e-5 --teacher-warmup-steps 0 \
  --native-prior-weight 0 --native-bit-tail-weight 0 --meta-weight 0.1
```

Change one factor at a time for causal comparisons. Reusing the old test for
development makes it unsuitable as the final independent test of improvements.
The launcher already supports a new `--seed` and `--exclude-prior-prompts`.
`--exclude-additional-prompts path1 path2` unions further manifests with the
original `--exclude-prior-prompts` manifest. The wrapper already includes the
October 19 run. This changes the new data splits as well as the optimization:
compare ablations within the new splits, not as a paired old-vs-new test.

CPU tests verify score compatibility, calibration ties, differentiable frozen-reader
loss, and current-adapter use in the no-grad prefix. The GPU smoke protocol uses
tiny image counts, four teacher updates (including the VAE and meta path), and two
updates per student. Its FPR .4 and sample size make it a software test only.

On 2026-10-19, the H200 smoke completed the full pipeline, including both final
student ranks and reporting after failed teacher quality/detection checks. The
teacher applied 4/4 updates, its meta/cycle step peaked at 83.24 GiB, and merged
prediction error was 5.13e-6. The migrated reader matched original-reader CPU
logits bitwise on the compatibility input. Fifteen native/shared-integration tests
and 39 shared protocol regressions passed. These are implementation checks, not
evidence of strong watermark inheritance. Raw smoke evidence is stored on vm_4 at
`/root/logs/results/native_unet_20261019_smoke/`.

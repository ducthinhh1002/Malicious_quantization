# Native UNet + frozen 8-bit reader experiment

Run `bash run_native_unet.sh` in the prepared environment. On vm_4 the wrapper
uses the recorded step-20,000 LALT checkpoint, pinned SHA256
`bce50d0a1a450ae315a90c4bfcfafe22c93b6bb79ceea85980a9ddef357e8eb0`.
On another machine override `NATIVE_OWNER_CHECKPOINT` and `NATIVE_PRIOR_PROMPTS`.
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
  across compared outputs. The old reader still carries prior-study knowledge.
- Teacher samples the complete current-adapter DDIM trajectory (20 steps); only
  the last four steps retain gradients. Its prefix is recomputed, not taken from
  cached clean-model latents. This is biased truncated backprop, not full BPTT.
- Loss: bit BCE + softplus margin above the frozen detection threshold + .5
  correct-vs-wrong-key margin + paired clean-image MSE/LPIPS quality budgets.
  Quality uses .001 MSE and .1 LPIPS budgets, small in-budget penalties and stronger
  per-example penalties when exceeded. Inputs use an RGB8 straight-through view.
- Every fourth update adds .25 watermark loss through a stochastic VAE posterior
  round trip. This only addresses the VAE bottleneck, not complete student learning.
- After 250 warmup updates, every 25th update adds .1 short-horizon LoRA-transfer
  loss through two ordinary AdamW denoising updates of a fresh rank-16
  attention+convolution surrogate. The clean rollout uses the same initialization
  and noise. The disjoint predicted-x0 probe is explicitly an approximation.
- Teacher 2,000 updates, snapshots 500/1,000/2,000; FP32, two microbatches of one,
  norm clipping 1. LR cosine 5e-5 -> 5e-6. This is a conservative starting choice
  for native UNet changes, not an experimentally established optimum.
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
Weak metrics never stop the study. A completed run does not mean successful transfer.

## Evidence behind the optimization choices

[AlignProp](https://arxiv.org/abs/2310.03739) establishes the practical use of
differentiable output rewards with diffusion LoRA and checkpointing. Our fixed
tail, paired quality objective, frozen watermark reader and short learner proxy
are this experiment's choices, not a claimed reproduction or proof of radioactivity.
[Diffusers LoRA documentation](https://huggingface.co/docs/diffusers/main/training/lora)
uses 1e-4 as the student LR default; it does not establish optimality for this dataset,
convolution scope, or watermark. All extra teacher/pilot compute must be reported.

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

# Projected native Transfer90

Independent continuation aimed at **single-image student TPR ≥90%**, using the historical frozen semantic verifier at the clean-student threshold (target FPR1%). The outcome remains an empirical hypothesis.

On VM4, after code/tests are installed:

```bash
bash /root/logs/project_native_transfer90/run.sh
```

The command queues the complete workflow behind the preserved `structure_margin_native` experiment. The cancelled `margin_native` is excluded. All new code/results are in this folder; the historical harness is a pinned, read-only dependency. `PN90_BASE_ROOT` overrides its location and `PN90_PYTHON` selects the already prepared Python. This launcher does not install libraries inside a job.

Teacher: warm-start the completed Projected native adapter;800updates, LR5e-6, batch4. Original watermark objective plus LPIPS/pixel/MS-SSIM preservation and margin repair across native RGB, stochastic SD1.4 VAE roundtrip, and frozen historical SD1.4 student denoising views. Margin increases gradually over200updates. Weak views receive extra weight; a small signed-coordinate margin discourages a few axes from dominating the scalar score. Gradients are balanced separately and the AdamW displacement is projected onto local descent constraints. These are truncated, local approximations, not downstream guarantees.

Student: independently initialized clean SD1.4; rank32 attention+conv;5000trainingimages;15000steps/batch4. Ordinary denoising loss, LR1e-4 through10000, cosine decay to2e-5 at15000. Select7500/10000/12500/15000 by held-out denoising MSE only. Save trainable parameters plus optimizer/scheduler/scaler/RNG every2500, avoiding duplicate frozen UNet weights. Interrupted training preserves state and requires an explicit audited continuation; it never silently restarts or overwrites it.

`runs/run_<UTC+7 timestamp>/status.json` records queue progress. Detailed logs are in `transfer90/logs/`. Outputs:

- `transfer90/result/verification.json`: single-image TPR/Wilson CI,90% point and lower-CI criteria, per-image bit accuracy, clean-control false positives, paired historical comparisons and warnings.
- `transfer90/result/per_image.csv` and `validation_curve.csv`.
- `transfer90/result/SUMMARY.md`, historical-format utility/teacher/student reports, and selected checkpoint evidence.

Poor scientific metrics are report-only. Nonfinite tensors, changed pinned inputs, incomplete datasets and technical failures are errors. Smoke uses the actual models before teacher training; completed metrics are never inferred from smoke.

The historical test has already been viewed during development; this is an exploratory single-seed pilot. A fresh holdout and multiple training seeds are needed for confirmatory claims. The frozen channel proxy is trained on historical training images, not final student outputs. It is a surrogate, not an unrolled downstream training process. Changing teacher, step budget and LR together does not isolate each contribution. Historical clean-trained control is7500steps, not a matched15000step control. Aggregate key recovery is reported separately from single-image TPR. FID/KID reference clean teacher generations, not real COCO.

Research motivating this pilot:

- [HMARK](https://arxiv.org/html/2512.00094v1): semantic watermark inheritance and LoRA; its reported metrics/false positives are not directly comparable to this frozen TPR criterion.
- [Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models](https://arxiv.org/html/2506.23731v1): examine VAE and noising/denoising channels when studying inheritance.
- [BitMark](https://arxiv.org/abs/2506.21209): a distinct autoregressive teacher mechanism; its results are not evidence that this diffusion pilot reaches90%.
- [Diffusers LoRA training](https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/lora.md): LR1e-4 as a starting point, not an optimum guarantee.

The finite-view mean/max loss is not a population CVaR estimator. Min-SNR is deliberately not added to this pilot because its convergence evidence does not establish improved watermark inheritance.

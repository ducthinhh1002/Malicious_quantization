# Hyper-parameter guide

This note summarizes how the three core hyper-parameters control the
trade-off between robustness (M-ACC) and image quality (FID).

<!--
  ASSET: assets/hyperparams.png
  Suggested content: Figure 5 of the paper — three side-by-side plots
  for alpha, eta, r_start (FID vs M-ACC).
-->
<p align="center"><img src="../assets/hyperparams.png" alt="Hyperparameter sweeps" width="90%"/></p>

## Watermark strength `α` (`--strength`, `GrowConfig.watermark_strength`)

`α` is the amplitude of each mid-frequency DCT coefficient assigned to
the watermark. Larger `α` means a stronger signal and higher robustness
under attacks, at the cost of more visible artifacts.

| α   | Avg. M-ACC | FID  |
|----:|:----------:|:----:|
| 0.3 | 0.92       | 9.5  |
| 0.5 | 0.98       | 12.3 (default) |
| 0.7 | 0.99       | 28.2 |

**Recommendation.** Start at `α = 0.5`. Increase only if your downstream
pipeline applies aggressive distortions.

## Guidance scale `η` (`--guidance`, `GrowConfig.guidance_scale`)

`η` controls how aggressively each per-step gradient modifies the
predicted clean latent `\hat z_0` toward the target signal.

| η    | Avg. M-ACC | FID  |
|-----:|:----------:|:----:|
|  50  | 0.90       | 17.8 |
| 100  | 0.99       | 16.2 |
| 125  | 0.99       | 15.1 |
| 200  | 0.98       | 12.3 (default) |

**Recommendation.** `η ∈ [100, 200]`. Excessively large values saturate
M-ACC while harming image quality.

## Start ratio `r_start` (`--start-ratio`, `GrowConfig.watermark_start_ratio`)

`r_start` defines the fraction of sampling steps that pass *before*
guidance kicks in. Starting earlier gives more steps to integrate the
watermark into semantic structure, improving robustness; starting too
late leaves the watermark surface-level.

| r\_start | Avg. M-ACC |
|---------:|:----------:|
| 0.3      | 0.97       |
| 0.5      | 0.98 (default) |
| 0.7      | 0.96       |
| 0.9      | 0.92       |

**Recommendation.** Keep `r_start = 0.5`. Lower it to ≈ 0.3 when you need
extra robustness on highly textured prompts.

## Secondary parameters

| Name            | Default     | Notes                                                              |
|-----------------|:-----------:|--------------------------------------------------------------------|
| `dct_min`, `dct_max` | 0.2, 0.5 | Mid-frequency band carrying the payload (as a fraction of the crop). |
| `channels`      | `[0,1,2,3]` | VAE latent channels used. More channels ⇒ higher capacity.         |
| `center_ratio`  | 1.0         | Fraction of the central latent processed with DCT (1.0 = whole).   |
| `cfg_scale`     | 7.5         | Classifier-free-guidance scale of the backbone pipeline.           |
| `num_inference_steps` | 50    | Number of DDIM steps.                                              |

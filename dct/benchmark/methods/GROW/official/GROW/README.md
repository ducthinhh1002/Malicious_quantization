<div align="center">

# GROW: Watermark Generation with Progressive Guidance for Diffusion Models

[![Paper](https://img.shields.io/badge/Paper-CVPR-1f6feb.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97-Diffusers-ff9d00.svg)](https://huggingface.co/docs/diffusers)

**Pengcheng Luo¹, Zexi Jia²\*, Yijia Zhong³, Jinchao Zhang², Jie Zhou²**

¹Peking University · ²WeChat AI, Tencent · ³Fudan University · \*Corresponding author

[**Paper**](#) · [**Project Page**](#) · [**BibTeX**](#citation)

<!--
  ASSET: assets/teaser.png
  Suggested content: a side-by-side of (non-watermarked image) vs (GROW-watermarked image)
  from the owl or cyborg prompt. Width ≈ 1200 px. Keep no caption inside the crop.
-->
<img src="assets/teaser.png" alt="GROW teaser" width="85%"/>

</div>

---

## TL;DR

**GROW** is a **training-free** watermarking framework for latent diffusion
models that reframes watermarking from a *one-shot imprint on the initial
noise* into a *progressive growth during denoising*. By steering the
generation with frequency-domain gradients, the watermark is **woven into
the image**, and extraction reduces to a single VAE-encode + DCT — no
DDIM inversion needed.

* **Inversion-free extraction** → **~100× faster** than Tree-Ring / WIND / Gaussian Shading.
* **Superior imperceptibility** (PSNR ≈ 27.5, FID 12.3 on MS-COCO).
* **State-of-the-art robustness** (Avg. M-ACC ≈ 97.8%) across rotation, JPEG, crop, blur, noise, brightness and adversarial purification.
* **Plug-and-play**: no retraining, no model surgery — drop in on top of SD 1.5 / SD 2.1 / SDXL.

---

## Motivation

<!--
  ASSET: assets/paradigm.png
  Suggested content: Figure 1 of the paper — "Initial-Noise Watermark" vs
  "Progressive Guidance Watermark" (the two-panel dashed-box diagram).
-->
<p align="center"><img src="assets/paradigm.png" alt="Paradigm comparison" width="70%"/></p>

Existing training-free methods (Tree-Ring, RingID, WIND, Gaussian Shading,
…) embed a static pattern into the initial noise `z_T` and recover it by
**DDIM inversion** — an expensive optimization that dominates detection
time. Directly injecting the watermark into the final latent `z_0`
collapses the image distribution. GROW resolves both issues by treating
watermarking as *optimization concurrent with denoising*: small
gradient-guided updates accumulate along the data manifold, letting the
U-Net render the watermark into plausible image textures.

---

## Method

<!--
  ASSET: assets/framework.png
  Suggested content: Figure 2 of the paper — overall framework with
  (a) watermark generation (progressive guidance) and (b) extraction.
-->
<p align="center"><img src="assets/framework.png" alt="GROW framework" width="92%"/></p>

At each denoising step `t` after `T · r_start`, GROW performs four
operations (see Algorithm 1 of the paper):

1. **Predict the clean latent** `\hat z_0` from the noisy `z_t` and the
   conditional noise estimate (Eq. 1).
2. **Measure the watermark error** — the MSE between `DCT(\hat z_0)` and
   a deterministic mid-band target `S` on the masked locations (Eq. 5):
   `L_wm = ‖(DCT(\hat z_0) − S) ⊙ M‖²`.
3. **Guide** `\hat z_0` with a single gradient step of scale `η` (Eq. 6).
4. **Back-convert** to a new effective noise prediction that the
   scheduler can consume (Eq. 7), then blend with the unconditional
   branch using standard classifier-free guidance.

**Extraction** (inversion-free): VAE-encode the image, take the DCT,
read the sign of each coefficient at the masked locations, and majority
vote across redundant copies.

---

## Why progressive guidance?

<!--
  ASSET: assets/ablation_one_step.png
  Suggested content: Figure 6 of the paper — (a) clean (b) GROW (c) one-step injection.
-->
<p align="center"><img src="assets/ablation_one_step.png" alt="One-step vs progressive" width="55%"/></p>

A naïve one-step injection into `z_0` pushes the latent off the data
manifold and collapses image quality (**FID 157.8 / PSNR 11.4 / SSIM
0.19**), while GROW preserves fidelity (**FID 12.3 / PSNR 27.5 / SSIM
0.85**). Progressive guidance succeeds because each perturbation is
small enough for the U-Net to interpret it as high-frequency detail and
render it as plausible texture (grain, fur, wood, water ripples…).

---

## Results

### Quantitative

| Method (SD 2.1-base) | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ | Clean | Rot. | JPEG | Crop | Blur | Noise | Bright. | Adv. | **Avg.** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DwtDctSvd | 42.05 | 0.99 | 24.65 | 0.04 | 1.00 | 0.47 | 0.06 | 0.50 | 0.45 | 0.72 | 0.14 | 0.56 | 0.487 |
| HiDDeN    | 23.75 | 0.80 | 23.21 | 0.09 | 0.94 | 0.76 | 0.57 | 0.93 | 0.75 | 0.79 | 0.71 | 0.67 | 0.765 |
| Tree-Ring | 16.48 | 0.27 | 22.38 | 0.43 | 0.99 | 0.05 | 0.46 | 0.08 | 0.64 | 0.50 | 0.44 | 0.87 | 0.504 |
| RingID    | 15.73 | 0.51 | 23.63 | 0.48 | 1.00 | **1.00** | **1.00** | 0.57 | 0.97 | **0.98** | 0.97 | 0.88 | 0.921 |
| Gaussian Shading | 26.21 | 0.83 | 22.45 | 0.07 | **1.00** | 0.59 | **1.00** | 0.92 | 0.94 | 0.97 | 0.96 | 0.93 | 0.913 |
| WIND      |  7.44 | 0.26 | 21.89 | 0.44 | **1.00** | 0.75 | **1.00** | 0.84 | 0.98 | 0.97 | 0.96 | 0.90 | 0.925 |
| **GROW (Ours)** | **27.54** | **0.85** | **12.32** | **0.05** | **1.00** | **1.00** | **1.00** | **1.00** | **0.99** | 0.91 | **0.98** | **0.95** | **0.978** |

> Dataset: **MS-COCO**, 1,000 generated images, 16-bit payload. Robustness numbers are Message Accuracy (M-ACC). Full table on Stable-Diffusion-Prompts, SD 1.5 and SDXL is in the paper.

### Extraction speed

| Method            | Generation (s) | Extraction (s) |
|-------------------|:---:|:---:|
| Tree-Ring         | 21.03 / 0.04 | 19.70 |
| RingID            | 21.07 / 0.08 | 15.30 |
| Gaussian Shading  | 21.24 / 0.25 | 12.50 |
| WIND              | 21.17 / 0.18 | 21.70 |
| **GROW (Ours)**   | 21.16 / 0.17 | **0.24** |

> Times are seconds per 512×512 image on a single NVIDIA T4 GPU with 50 sampling steps.

### Qualitative comparison

<!--
  ASSET: assets/visual_comparison.png
  Suggested content: Figure 4 — grid of prompts × methods (Hidden / Tree-Ring / WIND / GROW).
-->
<p align="center"><img src="assets/visual_comparison.png" alt="Visual comparison" width="92%"/></p>

### Attack gallery

<!--
  ASSET: assets/attacks_panel.png
  Suggested content: Figure 3 — the 8-panel grid (clean, rotation, JPEG, crop, blur, noise, brightness, adversarial).
-->
<p align="center"><img src="assets/attacks_panel.png" alt="Attack gallery" width="92%"/></p>

---

## Installation

```bash
git clone https://github.com/<your-handle>/GROW.git
cd GROW

# Recommended: isolated environment
conda create -n grow python=3.10 -y
conda activate grow

pip install -r requirements.txt
pip install -e .        # optional: `import grow` from anywhere
```

A modern GPU (≥ 10 GB VRAM for SD 2.1, ≥ 16 GB for SDXL) is recommended.
CPU inference is possible but impractically slow.

---

## Quick start

### 1. Generate a watermarked image and benchmark its robustness

```bash
python scripts/run_demo.py \
  --prompt "a sharp wildlife photo of an owl with feather details, perched on an old tree branch" \
  --message OKOK \
  --secret-key watermark \
  --output-dir outputs/owl
```

This produces `outputs/owl/`:

```
reference.png            # non-watermarked reference
watermarked.png          # GROW-watermarked image
attack_Clean.png ...     # distorted copies used in the robustness table
```

and prints a per-attack message-accuracy table.

### 2. Extract the watermark from an arbitrary image

```bash
python scripts/extract.py \
  --image outputs/owl/watermarked.png \
  --message OKOK \
  --secret-key watermark
```

### 3. Use it from Python

```python
from grow import GrowConfig, GrowWatermarker

cfg = GrowConfig(
    model_id="stabilityai/stable-diffusion-2-1-base",
    secret_key="watermark",
    message="OKOK",
    watermark_strength=0.5,   # alpha
    guidance_scale=200.0,     # eta
    watermark_start_ratio=0.5 # r_start
)

wm = GrowWatermarker(cfg)
image = wm.generate_with_watermark(
    "A portrait of a cyborg in a golden suit, D&D sci-fi, artstation, concept art"
)
image.save("watermarked.png")

recovered = wm.extract(image)
print("Recovered:", recovered)  # "OKOK"
```

---

## Repository layout

```
GROW/
├── grow/                  # library source
│   ├── __init__.py        # public API: GrowConfig, GrowWatermarker, ...
│   ├── config.py          # hyper-parameter dataclass
│   ├── watermark.py       # progressive-guidance generation + extraction
│   ├── codec.py           # message <-> bits
│   ├── attacks.py         # robustness evaluation suite
│   └── utils.py           # tensor / DCT helpers
├── scripts/
│   ├── run_demo.py        # end-to-end demo (generate + attack + extract)
│   └── extract.py         # stand-alone extractor
├── configs/
│   └── default.yaml       # paper defaults
├── assets/                # README figures (see assets/README.md)
├── docs/
│   ├── hyperparameters.md
│   └── robustness.md
├── requirements.txt
├── setup.py
├── CITATION.cff
└── LICENSE
```

---

## Key hyper-parameters

| Name (CLI / `GrowConfig`)      | Paper symbol | Default | Meaning                                                                   |
|--------------------------------|:---:|:---:|----------------------------------------------------------------------------|
| `--strength` / `watermark_strength`      | α        | 0.5   | Amplitude of DCT target coefficients.                                      |
| `--guidance` / `guidance_scale`          | η        | 200.0 | Per-step gradient guidance scale.                                          |
| `--start-ratio` / `watermark_start_ratio`| r\_start | 0.5   | Fraction of sampling elapsed before guidance begins.                       |
| `dct_min`, `dct_max`                     | —        | 0.2, 0.5 | Mid-frequency band used to carry the payload.                            |
| `channels`                               | —        | [0,1,2,3] | Which VAE latent channels carry the watermark.                           |
| `center_ratio`                           | —        | 1.0   | Fraction of the central latent region processed with DCT (≤ 1).            |

See [`docs/hyperparameters.md`](docs/hyperparameters.md) for the FID/M-ACC trade-off curves.

---

## Limitations

* Robustness degrades under *random aspect-ratio stretching*.
* Capacity is bounded by the size of the mid-frequency band in the VAE
  latent.
* As a DCT-domain method, it is susceptible to aggressive frequency
  filtering — a learned, attack-resilient embedding space is a natural
  next step (see paper §6).

---

## Acknowledgements

This work was completed at the Pattern Recognition Center, WeChat AI,
Tencent. We thank Prof. **Anil K. Jain** for his generous guidance and
insightful discussions. We build on the excellent open-source work of
[🤗 diffusers](https://github.com/huggingface/diffusers),
[Tree-Ring](https://github.com/YuxinWenRick/tree-ring-watermark),
[RingID](https://github.com/showlab/RingID) and
[Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading).

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{luo2026grow,
  title     = {GROW: Watermark Generation with Progressive Guidance for Diffusion Models},
  author    = {Luo, Pengcheng and Jia, Zexi and Zhong, Yijia and Zhang, Jinchao and Zhou, Jie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE).

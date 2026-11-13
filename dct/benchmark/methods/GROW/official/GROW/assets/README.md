# `assets/` — image placeholders for the repository

This folder hosts the figures referenced from the top-level `README.md`.
Place each file below at exactly the indicated path so the Markdown
links in the README and in `docs/` resolve automatically. All figures
are available in the Overleaf source of the paper.

| File name                    | What to put here (source: Overleaf)                                                                                       | Where it is used                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `teaser.png`                 | A visually striking side-by-side: *clean image* vs *GROW-watermarked image* (ideally the owl or cyborg prompt).            | README top banner (under the title).              |
| `paradigm.png`               | **Figure 1** — "Initial-Noise Watermark" vs "Progressive Guidance Watermark" comparison.                                   | README §Motivation.                               |
| `framework.png`              | **Figure 2** — overall framework: watermark generation (progressive guidance) + inversion-free extraction.                 | README §Method.                                   |
| `attacks_panel.png`          | **Figure 3** — watermarked image under 7 attacks (rotation, JPEG, crop, blur, noise, brightness, adversarial).             | README §Robustness / `docs/robustness.md`.        |
| `visual_comparison.png`      | **Figure 4** — qualitative comparison across Hidden / Tree-Ring / WIND / GROW on 3 prompts.                                | README §Qualitative Results.                      |
| `hyperparams.png`            | **Figure 5** — ablation of α, η, r_start (FID vs M-ACC trade-off).                                                         | `docs/hyperparameters.md`.                        |
| `ablation_one_step.png`      | **Figure 6** — one-step injection vs GROW (shows one-step collapse).                                                       | README §Why Progressive Guidance.                 |
| `logo.png` *(optional)*      | A small square logo / wordmark if you'd like one next to the title.                                                        | README title.                                     |

Recommended export settings from Overleaf:

* Width: ≥ 1200 px for full-width figures (framework, visual_comparison,
  attacks_panel), ≥ 700 px for half-width ones.
* Format: PNG (lossless) preferred; JPEG is acceptable for photographic
  figures.
* Keep the captions off the exported crop — the README provides its own
  short captions.

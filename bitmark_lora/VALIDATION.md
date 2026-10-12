# Validation, 2026-10-12

Executed locally with Python 3.12, torch 2.7.1+cpu, torchvision 0.22.1+cpu,
diffusers 0.35.1, transformers 4.51.3, peft 0.17.1, accelerate 1.10.1,
datasets 3.6.0, pyarrow 19.0.1, NumPy 1.26.4:

- 11 protocol/integrity tests passed, including a synthetic end-to-end report,
  paired clean adjustment, altered-image rejection, ties at the FPR threshold,
  nonfinite rejection, stage failure propagation, and matched trainer commands.
- Official BitMark detector and logits processor passed CPU checks on alternating
  and constant bit sequences. This validates API wiring and score direction,
  not watermark detection on real generated images.
- Both vendored Diffusers trainers completed **two actual optimizer steps** on
  a tiny randomly initialized SD model created locally. LoRA up-projections
  changed; full UNet weights changed. Both checkpoints saved, reloaded and
  generated a 32×32 image. The input imagefolder included auxiliary JSON
  manifests, as in the real pipeline.
- Official Infinity import chain and adapter arguments checked on CPU with an
  explicit import-only FlashAttention stub. No Infinity forward pass was claimed.
- Shell syntax, Python compilation, root `--help` and `--plan-only` checked.
- Upstream source hashes and pinned model revisions/LFS hashes checked against
  official repositories. Large model download/verification runs on the GPU host.

Runtime issues found and fixed during these checks:

1. Unconstrained PyArrow 26 requires NumPy ≥2, despite resolving alongside
   NumPy 1.26. Pin PyArrow 19.0.1, verified by real imagefolder training.
2. Accelerate 1.10 rejects `log_with="none"`. Both trainer copies explicitly
   map that CLI value to Python `None`; source changes are hash-recorded.
3. LoRA parameter iterator exhaustion would disable repeated gradient clipping.
   The same `list(filter(...))` correction as SERUM is retained.

The deployment environment is Python **3.11/CUDA 12.8**, not this CPU test
environment. The local machine has no NVIDIA GPU and insufficient disk for all
checkpoints. Full Infinity image generation, FlashAttention CUDA execution,
real SD2.1 training, and final CLIP/FID/KID have **not** been run here. The launcher
exercises FlashAttention on CUDA before downloading models. No experimental
TPR or conclusion about LoRA radioactivity is fabricated or promised.

# Teacher 32 bit / rank32: manipulation refinement v2

This isolated experiment warm-starts the **Structure 32-bit teacher after its original 600 refinement steps**. It does not train a student. Its display name is `VLLM::Enginecore`; its real experiment identity, PID, GPU and source checkpoint remain recorded in deployment metadata.

## What the old experiment did not solve

The previous 8-bit experiment trained rotation within ±15°, but the selected historical Rotation result used ±90°. Training blur used kernel9/σ0.5–1.5, while the best historical blur row used kernel15/σ0.1–2.0. These are concrete distribution mismatches, not evidence that augmentation is ineffective.

Its clean/attacked objectives shared a gradient budget and one attacked view per image. A good mean score can coexist with incorrect individual bits. The fixed semantic extractor is not inherently rotation/crop invariant; we cannot assume extra augmentation alone will fix every failure. The prior 100% crop result was 32-bit statistical acceptance with at least24 correct bits, not exact8/8 recovery.

## Literature and chosen adaptation

| Work | Relevant mechanism | Use here |
|---|---|---|
| [HiDDeN, ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.pdf) | Distortion layers between embedding and decoding teach robustness. | Differentiate through the attacked teacher output rather than augmenting student training data. |
| [Stable Signature, ICCV2023](https://arxiv.org/abs/2303.15435) | Fine-tune the diffusion VAE decoder against a pre-trained watermark extractor, preserving image appearance. | Retain a frozen reader and explicit perceptual objective. We keep UNet LoRA for this experiment; changing the VAE would change the embedding mechanism and needs its own transfer experiment. |
| [Watermark Anything, ICLR2025](https://arxiv.org/html/2411.07231v1) | Joint embedding/extraction, localized decoding and geometric/valuemetric augmentations. | Train both geometry and photometric views. Pixel-localized decoding is a possible future reader redesign, not something the current global carrier already provides. |
| [SERUM, ICLR2026](https://adam-dziedzic.com/static/assets/papers/serum.pdf) | A learned detector consumes augmented images/latents. | Match transform definitions precisely. We do not train a fixed-key-only reader: that risks a constant-key solution and invalidates the original detector comparison. |
| [Guidance Watermarking, ICLR2026](https://arxiv.org/html/2509.22126v2) | Backpropagates through transformed outputs of a fixed reader and combines conflicting gradients with PCGrad. | Adapt the multi-view gradient idea to teacher LoRA refinement. This is not a reproduction of their inference-time guidance. |

## Method

1. Keep source key/carrier, base SD2.1, LoRA rank32, 50-step native PNDM generation, CFG7.5 and prompt/seed pairing unchanged. Freeze VAE, text encoder and semantic reader. Source SHA256: `2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8`.
2. Add1200 optimizer steps, LR5e-6, batch4, microbatch2. Every second step generates a full native trajectory and differentiates its last10 denoiser calls. This truncation saves memory but is an approximation to the full trajectory derivative.
3. Each native image has a geometry view and a photometric view. Rotation/crop get more exposure than horizontal flip. Train rotation curriculum±15→±90 over200 optimizer steps. All remaining steps use the full target angle distribution.
4. For signed normalized coordinates `s`, use `d=relu(0.1-s)^2`, `Lbit=0.5*mean(d)+0.5*mean(top8(d))`. This repairs weak bits while keeping pressure on all32. It does not change the acceptance threshold or reduce payload.
5. Compute the two view gradients with respect to the generated image; apply symmetric two-task PCGrad per image, then propagate the merged gradient into LoRA. **Image-space conflict correction does not guarantee non-conflicting parameter-space updates.** Log the conflicts and gradient norms.
6. Keep original threshold loss, clean per-image score floor, clean weak-bit loss, LPIPS+0.1pixel+0.25MS-SSIM. Bound quality, clean margin and robust gradient norms against the original gradient. Project the actual AdamW proposal against three local descent constraints: original loss, clean margin, quality. These are first-order/truncated-gradient protections, not guarantees of finite-step quality or detection.
7. Save adapter, optimizer, data order and RNG every200 steps. Select the predeclared final1200; save all results, no scientific early-stop gate. A technical failure stops and records the error. Existing source checkpoint remains unchanged.

## Frozen test panel: exactly the settings in the selected best table

| Transform | Setting | Channel |
|---|---|---|
| Original | None | Tensor |
| Flip | Horizontal, fixed | Tensor |
| Rotation | Uniform[-90°,90°], nearest, gray0.5 fill | RGB8 truncation |
| Crop75 | Random square position;75%area; bilinear resize, antialias | Tensor |
| Drop64 | Independently keep64%pixels; replace36%withgray0.5; oneRGB-shared mask | Tensor |
| Gaussian | σ0.1 on[0,1], clip; independent seed per image | Tensor |
| Brightness | Factor0.7 | RGB8 truncation |
| Blur | Kernel15, σUniform[0.1,2.0] | RGB8 truncation |

No JPEG or salt–pepper. Evaluation transforms run on CPU with the old per-image torch RNG policy; before/after use identical draws. Training uses differentiable operations and a declared straight-through RGB8 quantizer. A new 32-bit baseline is measured using this same panel; the old32-bit SERUM-like numbers must not be reused across incompatible settings.

Use1000validation and1000test images per teacher. The test is for reporting, not checkpoint/setting selection. Historical best settings were selected on old results, so this is a declared targeted benchmark, not an unbiased new attack-discovery benchmark. Independent future attack settings/seeds are needed for a general robustness claim.

Report BA, **exact32/32 FK**, and **key acceptance≥24/32** separately, each for groups1/8/32/64/256. Group32 has3100 group evaluations and group256 has300 from100 random partitions; image reuse between partitions precludes treating these as independent trials. The theoretical binomial FPR for≥24/32 is0.350018%, conditional on independent fair matches under the null; it is not an empirically measured FPR. Also save per-bit accuracies/margins to expose systematic errors.

Quality uses paired validation LPIPS and CLIP for both original-refine600 and the new teacher. LPIPS is saved before CLIP runs, so a missing CLIP cache cannot block manipulation reporting. No improvement is claimed before these comparisons finish. Student transfer remains untested by this teacher-only experiment.

## Run

`deploy.py` probes the selected server (default .21) through the existing bastion, verifies the exact source checkpoint and selects a GPU with at least70GiB free, uploads only this small code bundle, then launches `run_teacher.py`. It never stops existing jobs and never connects the servers to each other. It records where it launched. The real-model smoke must pass before training starts.

Technical tests check transformation parity with torchvision/PIL, all attack gradients, image-gradient chain rule, PCGrad conflicts, weak-bit loss and the difference between exact FK and statistical acceptance. A passed local numerical test is not evidence that a GPU training run has started.

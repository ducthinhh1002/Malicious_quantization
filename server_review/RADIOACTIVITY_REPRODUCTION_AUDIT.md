# C10 radioactivity reproduction audit — 2026-10-10

> **Superseding finding, October 11:** The [deeper C10 audit](C10_CODE_AUDIT_2026-10-11.md) confirmed a gradient-clipping parameter-iterator defect in the pinned Diffusers trainer. It has been fixed for future runs. Its effect on radioactivity has not been measured; the original results are pre-fix. The historical finding below is not a current assertion that no implementation defect exists.

## Finding

No implementation defect explaining the gap has been found in the checks performed. The current experiment shows weak radioactive transfer, but does not reproduce the published 52.30% result. This is not evidence that the published number is unreliable. Neither the exact author artifact nor the exact adaptation implementation has been reproduced.

## Measured evidence

- Teacher: 4,999/5,000 watermarked images detected; 63/5,000 clean images detected.
- Unadapted SD1.4: 48/5,000 positive (0.96%).
- C00, adapted on clean teacher images: 46/5,000 positive (0.92%).
- C10, adapted on watermarked teacher images: 162/5,000 positive (3.24%).
- All official measurements use frozen threshold 0.04637297987937927.
- A separate post-hoc diagnostic threshold derived from C00 gives C10 3.36% at 1% empirical C00 FPR. This diagnostic reuses C00 for calibration and is not an independent evaluation. It does not replace the frozen threshold. The small change does not explain a jump to 52.30%.
- C10-versus-C00 score ROC AUC: approximately 0.610.

## Implementation checks

- Raw image decoding, counts, shard-to-merged hashes and four prompt split overlaps passed the existing audit.
- New full HF image/caption pairing audit: all 5,000 rows per C00/C10 match raw images and prompts, with zero mismatches.
- Rank 4, LR 1e-5, batch 8, alpha 0.5, 5,000 training images, 1,000 validation images, 5,000 evaluation outputs; text encoder frozen.
- LoRA targets attention q/k/v/out, alpha equals rank, optimizer and LR scheduling use the pinned Diffusers example's defaults. These are implementation choices; rank alone does not prove agreement with the paper's exact LoRA configuration.
- C10 reached 20,000 steps without logged nonfinite losses. Minimum unseen MSE selects checkpoint 20,000: 0.1460962672.
- Fixed-input UNet test: loaded C10 adapter changes predictions; disabling it returns exactly to the base prediction. Relative change was about 2.69% on two diagnostic inputs, not a quality or radioactivity measurement.
- SD1.4 tokenizer input IDs and float32 text embeddings are identical for nine tested prompts between Transformers 4.53.1 and 5.19.0. This excludes the tested text-conditioning discrepancy, not every possible dependency difference.
- Student outputs use 512x512, 50 steps, CFG 7.5 and PNDM, with paired prompt IDs and sampling seeds across controls.

## Remaining reproduction uncertainties

- Detector and watermark were trained locally. They are not the exact key/checkpoint used for the published radioactivity result. High teacher image TPR alone does not establish detector sensitivity to a weak, transferred signal.
- Prompt IDs and generated samples differ from the authors' exact adaptation dataset.
- The inspected public SERUM checkout (02f4929e30cd976c5cfe65a45ef1bc4ac79bdfed) has no dedicated radioactivity/LoRA fine-tuning script. Appendix I specifies main hyperparameters, but not a complete runnable adaptation recipe or all LoRA settings.
- Diffusers 0.41.0.dev0, PEFT 0.21.2 and Transformers 5.19.0 are the local student environment, not a verified author environment. The text-encoder comparison above checks one component only.
- SD2.1 Base is loaded from a pinned public mirror; its hashes are recorded, but the recorded manifest alone does not establish equality with the exact original weights used by the paper.
- Only one adaptation seed has been evaluated.

## Decision

Continue the user-authorized Stage 1–2 image pilot, retaining the weak-baseline status. Do not label the full radioactivity reproduction successful or interpret a future low C11 TPR as strong attack success solely from this run. Resolving the published gap requires the original radioactivity recipe/artifacts or controlled tests of remaining configuration differences, followed by more adaptation seeds. Do not tune checkpoint, threshold or LoRA settings on the evaluation TPR to force agreement.

Sources: [SERUM paper, Appendix I](https://adam-dziedzic.com/static/assets/papers/serum.pdf); [official SERUM repository](https://github.com/Hubizon/SERUM); [official SD2.1 configuration](https://raw.githubusercontent.com/Hubizon/SERUM/master/configs/config_sd21.yaml).

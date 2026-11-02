# SERUM author-artifact branch

The public SD2.1 checkpoint and original model cache linked by Hubizon in [SERUM issue #1](https://github.com/Hubizon/SERUM/issues/1#issuecomment-4963328577) are now the active experiment. The user cancelled the previous corrected core on 2026-10-11. Its logs and artifacts remain available for comparison.

## Assets and provenance

- Server asset directory: `/root/workspace/serum/assets/serum_author/`.
- Author checkpoint: Drive ID `12NTXeyIMFX1NWYzCJDeRX54T7BgEvXuQ`; epoch 50; checkpoint SHA-256 `e991a694342545fa4f59cc5b3c94b6e3a93bf831dd4d507207f52d1c167c05c4`.
- Original SD2.1 cache: Drive ID `17MFAZ2rsmaIlRNghzTbShLE73t6QCCrN`; approximately 9.58 GB compressed; verified ZIP SHA-256 `95877f6ab0305b92897c5f6fdba7fcd373b9031728f3e69599ffebfb6d2bc593`. Snapshot revision `5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06`. ZIP integrity passed; all 13 model/config/tokenizer file hashes recorded. This is author-supplied provenance; no independently published reference checksum was provided.
- Extract only the detector and watermark state dictionaries from a `weights_only=True`, CPU-loaded checkpoint. Strict architecture checks, finite tensors and `(1,4,64,64)` key shape must pass.
- Store original cache separately. Existing model caches, SERUM config, key and thresholds are retained for historical comparison. Previous GPU experiment jobs and pending old-key conditions are cancelled.

## Scheduled work

Session: `quant-author-queue`; status: `/root/workspace/serum/metrics/serum_author/queue_status.json`.

Immediate C00 utility evaluation is scheduled in `quant-c00-utility` at the user's request. After C00 generation and author-detector audit finish, the hook temporarily holds the raw queue coordinator, computes CLIPScore/FID/KID on GPU0, writes `metrics/C00_seed0_author_utility.json`, then releases the coordinator to start C10. It releases the coordinator on failure as well. Status: `metrics/serum_author/C00_utility_hook_status.json`.

1. Download and validate both assets.
2. Start immediately, with no dependency on the cancelled core. A full 50-step clean/watermarked compatibility smoke passed.
3. Reuse 11,000 clean teacher images: train 5,000, validation 1,000, calibration 5,000. Exact tensor and file-hash comparisons established that the previous mirror's UNet, VAE, text encoder, scheduler, tokenizer and configs match the author cache. Decode and hash audits are repeated before reuse. Generate 11,000 new watermarked images with the author key/cache on both GPUs. Preserve current disjoint prompt splits and prompt-ID pairing. New WM per-image seeds are 10,000 + index, 20,000 + index and 30,000 + index respectively. Use alpha 0.5, 512 px, 50 steps, CFG 7.5 and the author's scheduler. Reused clean images and new WM images share prompts; matching initial latent noise is not claimed because the legacy per-image noise was not preserved.
4. Validate image decoding/count/pairing, store image/metadata hashes, and calibrate a separate threshold at empirical FPR <=1% on the 5,000 clean calibration images. Freeze it before auditing watermarked data or student outputs. Report teacher image detection rates.
5. Resume `C00_seed0_author` from the previous corrected clean-control checkpoint at step 8,500, including optimizer/scheduler/RNG state, after verifying the trainer and identical training metadata. Copy all previous clean-control checkpoints with hard links to preserve all validation candidates. Train `C10_seed0_author` fresh on the new key's images. Both use SD1.4 LoRA, 20,000 total steps, batch 8, LR 1e-5, rank 4, seed 0 and corrected clipping. Validate all 40 saved checkpoints and final weights; choose by minimum denoising MSE on each condition's 1,000 validation images.
6. Generate 5,000 student outputs per condition using existing held-out prompts and generation seeds/settings. Audit with the author detector, author-cache VAE and frozen author threshold. Save `/root/workspace/serum/metrics/serum_author/raw_baseline_summary.json`.
7. The user subsequently authorized strong C01/C11 immediately, without waiting for baseline review. Session `quant-author-strong` waits for teacher detection/calibration readiness, then runs C01 and C11 sequentially on GPU1 while C00/C10 use GPU0. C01 reuses the audited 6,000 existing clean strong-regeneration train/validation images. C11 regenerates 6,000 new author-key WM images with the same CtrlRegen+ settings and per-index seeds (train 10,000+i, val 20,000+i). Each trains fresh for 20,000 steps, selects by validation MSE, generates 5,000 evaluation images and uses the common frozen author threshold. Compute CLIPScore/FID/KID for all four conditions and save `metrics/serum_author/core_strong_summary.json`. No mild experiments are scheduled. If the raw C10 baseline remains weak, the completed strong results remain exploratory; attack validity is not automatically approved. Extra seeds and defenses remain conditional.

## Interpretation limits

An author's detector identifies its own watermark key; it cannot replace the old detector for primary audits of the old-key datasets or students. Separate thresholds and result names prevent mixing these experiments. Author artifacts address key/detector/backbone provenance differences, but do not supply the missing exact SD1.4 LoRA adaptation recipe. This branch is therefore an artifact-controlled reproduction, not a claim of exact paper reproduction. The queue does not use downstream student detections to choose regeneration strength.

Scripts are mirrored in `server_review/scripts/`: `download_author_serum.sh`, `prepare_author_serum.py`, `compare_author_backbone.py`, `reuse_author_clean.py`, `author_serum_stage0.py`, `queue_author_serum.py`, `queue_author_strong.py`, `audit_author_student.py`. Download dependencies are isolated in `.venv-download`. Existing SD1.4, CLIP and CtrlRegen assets and the clean utility reference are reused. No old-key C10/C11 data or adapters are used as author-key primary results. Strong queue status is `metrics/serum_author/strong_queue_status.json`; its immutable source plan is `manifests/serum_author_strong_plan.json`.

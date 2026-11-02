# LoRA students trained on manipulated Structure32 teacher outputs

Teacher: the original 32-bit/rank32 Structure refine600 checkpoint, SHA256 `2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8`. The later manipulation-trained teacher is not used.

Seven independent students start from the same clean SD1.4 base, rank32 attention+convolution LoRA, batch4, seed0, 12500 optimizer steps. Each student sees all5000 training images transformed by exactly one family from the user's attached table. The unchanged-image12500 student already exists as the control; it is not retrained.

| Branch | Input image transformation |
|---|---|
| flip | Fixed horizontal flip; tensor |
| rotation15 | Uniform angle±15°, nearest, gray0.5 fill; RGB8 truncation |
| crop75 | Random square crop retaining75%area, bilinear antialias resize; tensor |
| drop96 | Independently keep96%pixels;4%replacegray0.5; RGB-shared mask; tensor |
| gaussian005 | σ0.05 noise on[0,1], clip; tensor |
| brightness07 | Fixed factor0.7; RGB8 truncation |
| blur15 | Kernel15, sigmaUniform[0.1,2]; RGB8 truncation |

There is no JPEG, salt–pepper, or SERUM-specific replacement setting. All transforms are applied to the **training target image**, after ToTensor and before Normalize, so ordinary denoising MSE trains on manipulated targets. The extractor, key and owner scores do not enter student training. The tensor channels are not silently quantized to PNG. A hash of source pixel content maps to its fixed image index/seed. Repeated epochs see the same manipulation; CPU RNG state is restored afterward, preserving the sampler/noise RNG.

The pinned original trainer, schedule, full-state resume driver and evaluation pipeline are cloned into each isolated branch by prepare.py. Only the transform insertion,40GiB process cap, immutable augmentation identity and additive statistical acceptance report change. Trainer source itself is not edited. LR remains1e-4 through7500, cosine to2e-5 by12500. Full optimizer/scaler/RNG state is saved every1250steps; no quality gate or validation checkpoint selection. Training loss is logged in updates.jsonl. A four-update actual-model smoke precedes each full run.

Four slots run on .21 (two per GPU); remaining branches queue. Slots include generation/evaluation after training. Existing services are not stopped. All source images, model assets and original checkpoint are reused locally on .21. No server-to-server transfer is needed.

Primary evaluation generates1000 native, unmanipulated student outputs from the fixed12500 checkpoint. This measures whether the original watermark transfers through **student learning from manipulated teacher data**. Save BA, exact32/32 FK, and acceptance≥24/32 separately for groups1/8/32/64/256; the conditional theoretical binomial FPR is0.350018%. This does not imply empirical FPR calibration. Image utility and paired LPIPS are also retained. A manipulation-trained student may reproduce altered orientation/brightness or corrupted textures; this is part of the measured effect, not an image-restoration objective.

Remote root: `/var/lib/structure32_student_manipulations`. Inspect study_status.json, per-branch queue status.json, logs/01_train.log, students/lora_r32_conv_wm/updates.jsonl and result/student_acceptance32.json. Keep the generated code_manifest.json and resume_identity.json with checkpoints for reproducibility.

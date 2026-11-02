# Structure refine600, 8-bit rank32: augmentation experiment

Server .23: `/var/lib/structure8bit_rank32_aug`. Process name `VLLM: Enginecore`.
Uses exactly the shared Structure refine600 adapter, original key/carrier and
5000 paired teacher images uploaded for the rank4 run. Fresh rank32 attention/conv
SD1.4 student; 12500 steps, batch4, seed0, same7500/cosine12500 LR schedule.
Full optimizer/RNG checkpoints every1250; fixed12500 selection.

Online training augmentation occurs on CPU tensors in[0,1], before normalization
and VAE encode. Probability50%clean; otherwise exactly one uniformly chosen
rotation/crop/drop/salt-pepper/Gaussian-noise/brightness/blur. Explicit parameters
are in `augmentations.CONFIG`. Drop retains64%pixels and fills the rest with gray.
Crop retains75%area. No JPEG. These are exploratory settings, not claimed to
reproduce the supplied manipulation table or the SERUM paper.

This changes ordinary denoising training targets. It is a robustness hypothesis;
fixed semantic carrier invariance is not guaranteed. Both clean quality and
held-out manipulations must be checked before claiming improvement.

Evaluation compares augmented student, unadapted control and original
Structure12500 rank32 checkpoint on the same prompts/seeds/attack randomness.
Bit accuracy and full-key recovery use groups1/8/32/64/256; no TPR selection.
`robustness_eval.py` retains coordinates and reports each attack separately.

Queue shares GPU1 with the rank4 control and runs first, with rank4 waiting for its completion,
avoiding two training jobs exhausting available memory. Gemma uses GPU0.
The source datasets/env are shared; no duplicate4GiB transfer and no teacher train.
See `deployment/status.json` and the campaign run status on the server.

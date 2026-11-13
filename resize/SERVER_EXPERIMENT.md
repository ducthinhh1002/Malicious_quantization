# Latest Structure teacher -> Janus-Pro-1B autoregressive student

Teacher is /var/lib/structure32_codec_refine/run/teacher_fp/refined/checkpoint-1200,
adapter SHA256 f389b224214be6e683e21f46e4dd937e45313c9d75a6161b3968e6238a89eb07.
Frozen verifier is the corresponding new semantic carrier. Original32 signs are
preserved. Fixed24/32 acceptance is reported along with BA and exact32; it is not
a calibrated empirical1%FPR statement. Clean base Janus outputs provide a matched
null reference, including aggregation.

The old /var/lib/wmq_ar_student experiment was deleted in an earlier cleanup.
Recovered history identifies Janus-Pro-1B, rank32, batch4,12500steps, lr1e-4 and
warmup100. This wrapper rebuilds teacher-forced image-token distillation using
the official Janus model, generation head and VQ tokenizer. It does not claim to
reproduce every historical trainer setting. LoRA targets language-model Q/K/V/O
and gate/up/down projections. CFG prompt dropout0.1. Final checkpoint12500 is
fixed, unlike the old historical run which selected7500 by validation CE.

Official source https://github.com/deepseek-ai/Janus
Model https://huggingface.co/deepseek-ai/Janus-Pro-1B
Official git revision and model revision are recorded in protocol.json.
Only server downloads are used. Existing environment is reused, with attrdict
installed into isolated deps/ and the Python3.12 collections compatibility shim.

Teacher training prompts remain separate from student distillation prompts.
Distillation uses existing shared new-COCO5000 train,128 validation,1000 test
prompts. Native FP32 teacher generation uses batch8 and the fixed merged adapter.
Teacher outputs are clean PNGs. VQ preprocessing resizes512->384 with LANCZOS.
Janus predicts576 image tokens; its native decoder outputs384x384. Verification
uses deterministic LANCZOS384->512 for both student and base Janus outputs.

Preflight uses128 pre-existing held-out original-teacher test images, not student
training data. Results from a single diagnostic sample set:

| Condition | BA/image | Acceptance24/image | BA agg32 | Acceptance24 agg32 |
|---|---:|---:|---:|---:|
| Original teacher |65.8447%|16.4063%|87.0156%|100%|
| Resize384 then reader resize512 |63.5498%|7.8125%|84.5781%|99.75%|
| VQ384 then reader resize512 |54.5898%|0%|58.2734%|0.25%|

Aggregate uses100 random partitions of128 images, yielding400 groups of32.
Images are reused across partitions. These are tokenizer diagnostics; full AR
student recovery is a separate final experiment.

smoke_ar.json confirms a real teacher-forcing update with CE7.3871, finite
parameter gradient norm0.4147,576 target tokens and30277632 trainable LoRA
parameters. The diagnostic weights are discarded. The smoke exposed inference
label tensors that needed cloning before autograd; the training loss wrapper
now performs that conversion explicitly.

Stages run automatically from run.sh: tokenizer probe, teacher generation,
image-token cache, clean base Janus generation,12500-step student training,
student generation and frozen-reader scoring. Checkpoints every1250 updates save
LoRA plus optimizer and CPU/CUDA RNG. Batch order is reconstructed from step and
epoch seed. Final scoring covers per-image, agg32 and agg256 on1000 outputs.
No image-quality or AR robustness claim follows from the tokenizer preflight.

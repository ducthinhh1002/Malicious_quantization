# Historical BitMark evidence

Archive SHA256: b4e46f9c228cb84a0eeb9675957135bf3294decf19b99c6967433b3a5b77fe5f

Teacher positive control: 1000/1000 detected (100.00%); proceeding with audit.
# BitMark → SD2.1: LoRA radioactivity

Primary metric: |z| at frozen 1% target teacher FPR.

| Branch | Detected | Rate | 95% Wilson CI |
|---|---:|---:|---:|
| teacher_clean_test | 11/1000 | 1.10% | 0.62–1.96% |
| teacher_wm_test | 1000/1000 | 100.00% | 99.62–100.00% |
| unadapted | 61/1000 | 6.10% | 4.78–7.76% |
| lora_r4_wm | 494/1000 | 49.40% | 46.31–52.50% |
| lora_r16_wm | 629/1000 | 62.90% | 59.86–65.84% |
| lora_r32_wm | 663/1000 | 66.30% | 63.31–69.16% |
| lora_r4_conv_wm | 745/1000 | 74.50% | 71.71–77.10% |
| lora_r16_conv_wm | 886/1000 | 88.60% | 86.48–90.42% |
| lora_r32_conv_wm | 921/1000 | 92.10% | 90.26–93.62% |
| full_wm | 970/1000 | 97.00% | 95.75–97.89% |

## Validation-selected checkpoints

See checkpoint_metrics.csv for all milestones; test images never select checkpoints.

- lora_r4_wm: step 5000; held-out MSE 0.16363541.
- lora_r16_wm: step 5000; held-out MSE 0.16294933.
- lora_r32_wm: step 5000; held-out MSE 0.16275707.
- lora_r4_conv_wm: step 5000; held-out MSE 0.16312040.
- lora_r16_conv_wm: step 5000; held-out MSE 0.16207609.
- lora_r32_conv_wm: step 5000; held-out MSE 0.16170645.
- full_wm: step 1250; held-out MSE 0.16890801.

Full-minus-LoRA is the paired detection-rate difference on watermarked-trained students; no clean-trained adjustment is available.

- full_wm_minus_unadapted: 90.90 pp; paired CI [89.10, 92.60] pp.
- lora_r4_wm_minus_unadapted: 43.30 pp; paired CI [40.10, 46.40] pp.
- full_minus_lora_r4: 47.60 pp; paired CI [44.50, 50.70] pp.
- lora_r16_wm_minus_unadapted: 56.80 pp; paired CI [53.60, 60.00] pp.
- full_minus_lora_r16: 34.10 pp; paired CI [30.80, 37.20] pp.
- lora_r32_wm_minus_unadapted: 60.20 pp; paired CI [57.00, 63.40] pp.
- full_minus_lora_r32: 30.70 pp; paired CI [27.50, 33.80] pp.
- lora_r4_conv_wm_minus_unadapted: 68.40 pp; paired CI [65.20, 71.30] pp.
- full_minus_lora_r4_conv: 22.50 pp; paired CI [19.80, 25.30] pp.
- lora_r16_conv_wm_minus_unadapted: 82.50 pp; paired CI [80.20, 84.70] pp.
- full_minus_lora_r16_conv: 8.40 pp; paired CI [6.30, 10.50] pp.
- lora_r32_conv_wm_minus_unadapted: 86.00 pp; paired CI [83.90, 88.10] pp.
- full_minus_lora_r32_conv: 4.90 pp; paired CI [3.00, 6.80] pp.
- conv_minus_attention_r4: 25.10 pp; paired CI [21.40, 28.80] pp.
- conv_minus_attention_r16: 25.70 pp; paired CI [22.40, 29.10] pp.
- conv_minus_attention_r32: 25.80 pp; paired CI [22.70, 28.90] pp.

## Interpretation limits

- Single training seed; image bootstrap does not capture training-seed variance.
- Matched full controls have no EMA, matching LoRA. Not an exact reproduction of the paper's full+EMA run.
- Clean calibration uses generated Infinity images, not natural COCO images; inspect clean student false-positive rates.
- Green fraction is NOT message bit accuracy: BitMark tests bit-pattern bias.
- No clean-trained controls: contrasts use the unadapted baseline and cannot isolate generic fine-tuning effects.
- unadapted: clean detection exceeds nominal FPR; raw TPR alone is insufficient evidence of transfer.
- Checkpoint selected by held-out denoising MSE; validation detector/utility curves are descriptive only.
- Ranks/scopes share step budgets, not equal trainable parameters or FLOPs; inspect adapter_audit.json.


## Asset/setup log

```text
Downloading annotations_trainval2014.zip: 4/253 MB
Downloading annotations_trainval2014.zip: 252/253 MB
Disjoint COCO splits: train=1000, calibration=5000, test=1000, validation=256
Checkpoint FoundationVision/Infinity: download/verify infinity_2b_reg.pth
Checkpoint FoundationVision/Infinity: download/verify infinity_vae_d32reg.pth
Checkpoint google/flan-t5-xl: download/verify config.json
Checkpoint google/flan-t5-xl: download/verify flax_model.msgpack.index.json
Checkpoint google/flan-t5-xl: download/verify generation_config.json
Checkpoint google/flan-t5-xl: download/verify model-00001-of-00002.safetensors
Checkpoint google/flan-t5-xl: download/verify model-00002-of-00002.safetensors
Checkpoint google/flan-t5-xl: download/verify model.safetensors.index.json
Checkpoint google/flan-t5-xl: download/verify pytorch_model.bin.index.json
Checkpoint google/flan-t5-xl: download/verify special_tokens_map.json
Checkpoint google/flan-t5-xl: download/verify spiece.model
Checkpoint google/flan-t5-xl: download/verify tf_model.h5.index.json
Checkpoint google/flan-t5-xl: download/verify tokenizer.json
Checkpoint google/flan-t5-xl: download/verify tokenizer_config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify feature_extractor/preprocessor_config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify model_index.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify scheduler/scheduler_config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify text_encoder/config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify text_encoder/model.safetensors
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify tokenizer/merges.txt
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify tokenizer/special_tokens_map.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify tokenizer/tokenizer_config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify tokenizer/vocab.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify unet/config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify unet/diffusion_pytorch_model.safetensors
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify vae/config.json
Checkpoint Manojb/stable-diffusion-2-1-base: download/verify vae/diffusion_pytorch_model.safetensors

```

Interpretation: materially stronger observed LoRA detection than the SERUM run, but a different teacher/student pair and training dataset/budget. The unadapted SD2.1 false-positive rate is 6.1%; no matched clean-trained controls. Do not present the raw student rates as calibrated TPR at 1% student-domain FPR.

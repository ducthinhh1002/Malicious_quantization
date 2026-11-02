# Experimental Report: Transfer90 Semantic Watermark Inheritance

**Updated:**27 October2026. **Scope:**seed-0 SD2.1 teacher to ordinary SD1.4 LoRA student. Transfer90 completed at 10: 33 ICT. This report follows the experimental structure of the supplied October11 English report; its SERUM/CtrlRegen numbers belong to a different experiment and are not pooled here.

## 1. Main findings

**Transfer90 reaches97.8% single-image detection (978/1000), Wilson95%CI[96.69,98.54]%.** Both the estimate and its lower confidence bound exceed90%. The score threshold,8-bit key and semantic verifier are unchanged from the comparable k=1 campaign. This is the highest detection among the comparable completed branches listed here.

**Against Structure 12500:**detection increases6.7 percentage points, FID improves52.89584 to52.56332, and CLIP increases0.786258 to0.786725. LPIPS worsens0.528817 to0.539145; KID is effectively unchanged at the displayed scale. Transfer90 therefore does not improve every quality metric.

**Against the matched-panel k=1 baseline:**detection increases22.8 percentage points, LPIPS falls0.550447 to0.539145, and FID/KID improve. CLIP is slightly lower. Longer training, a different teacher and a different LR schedule change together; the result is a system-level comparison, not an isolated effect of the three-view loss.

## 2. Experimental protocol

| Component | Completed setting |

| --- | --- |

| Teacher | Manojb/stable-diffusion-2-1-base; warm-start completed Projected native(k=1), then800additional refinement updates; teacher LoRA rank32. |

| Student | Pinned CompVis/stable-diffusion-v1-4, newly initialized LoRA rank32 attention+convolution; frozen VAE/text encoder; ordinary denoising MSE. |

| Data | 5000teacher training images;1000validation;1000test;5000clean calibration prompts. Same test prompt panel as Structure12500 and k=1baseline. |

| Teacher optimization | AdamW LR5e-6, batch4; auxiliary native objective every2updates; microbatch1; full50-step PNDM forward, derivatives through last10denoiser calls. |

| Student optimization | 15000updates, batch4,fp16,seed0;LR1e-4through10000,cosine to2e-5by15000. |

| Selection / generation | Minimum validation denoising MSE at7500/10000/12500/15000; selected15000. PNDM50,CFG7.5,512x512,batch16; student seed500000+index. |


---


## 3. Method: native preservation and transfer-aware teacher refinement

**Lineage.**The original k=1 teacher checkpoint3000 was refined for 600 updates to obtain Projected native. Transfer90 warm-starts that adapter and adds 800 updates. Structure 12500 uses a separate Structure/margin teacher refined 600 updates directly from the original k=1 checkpoint. Transfer90 is not a continuation of the Structure 12500 student.

**Inherited watermark objective.**The original threshold_cfg_v1 loss preserves clean conditional/unconditional teacher predictions and applies calibrated semantic shifts to predicted clean latents and their teacher-VAE cycle. The semantic term operates on timesteps up to500. The downstream student receives only images/captions and ordinary denoising supervision.

**Native image preservation.**For each training prompt/seed, generate the full PNDM trajectory and compare the final RGB image against its clean SD2.1 paired image. RGB quantization is represented differentiably. Only the last 10 denoiser calls carry gradients; the forward pass and PNDM history remain complete. This is inspired by DRaFT-K[1], not a shortened 10-step sampler.

```text
L_quality = LPIPS(x_marked,x_clean) +0.1*MSE_RGB(x_marked,x_clean)
            +0.25*(1-MS_SSIM(x_marked,x_clean))
```

**Three teacher-side views.**(1)Native marked RGB;(2)stochastic SD1.4 VAE encode/decode;(3)one noising/denoising reconstruction through a frozen historical SD1.4 proxy, cycling timesteps 50,150,300 and using CFG7.5. The proxy is the Projected native 10000 student trained on historical watermarked images. Its weights are frozen; gradients pass through it to the teacher image. The final student is separately initialized and is never used as this proxy.

```text
score_v = mean_j(key_j * standardized_coordinate_vj)
d_v = max(target_v-score_v,0)/sigma_clean
L_margin =0.5*mean_v(d_v^2) +0.5*max_v(d_v^2)
L_bits = mean_v,j[max(0,0.25-key_j*coordinate_vj)^2]
L_owner = L_margin +0.1*L_bits
```

**Score floors.**Let tau=max(frozen teacher threshold, frozen student threshold) and r=min(1,(update+1)/200). Native/cycle floor=tau+(1+r)*sigma_clean; denoised floor=tau+(0.5+0.5r)*sigma_clean. The native target also preserves the previous Projected native image score when that score exceeds the floor. Deficits are penalized per image before averaging; the worst of the three views receives extra weight. This finite-view objective is not a population CVaR guarantee.

**Gradient balancing and projection.**Compute base, quality and ownership gradients separately. Scale auxiliary gradients to target1.0and1.5 times the base-gradient norm, with coefficient cap100. AdamW proposes an update; on auxiliary steps, project its actual parameter displacement onto the intersection of the three local halfspaces g_i dot delta&lt;=0. This is inspired by gradient surgery[2], but operates on the realized AdamW displacement. It provides local truncated-gradient constraints, not a monotonic detection/quality guarantee.


---


## 4. Final student results and paired comparisons

| Branch | Steps | Detected | TPR /95%CI | LPIPS | FID | KID x1000 | CLIP |

| --- | --- | --- | --- | --- | --- | --- | --- |

| Baseline k=1 | 7500 | 750/1000 | 75.0% | 0.550447 | 54.444 | 1.6665 | 0.787681 |

| Structure12500 | 12500 | 911/1000 | 91.1% /89.17-92.71 | 0.528817 | 52.896 | 0.6842 | 0.786258 |

| Transfer90 | 15000 | 978/1000 | 97.8% /96.69-98.54 | 0.539145 | 52.563 | 0.6847 | 0.786725 |

**Metric definitions.**LPIPS(AlexNet0.1)[3]compares each student image with unadapted SD1.4 at the same prompt/seed; lower means closer, not necessarily more attractive. FID[4]/KID[5]compare 1000 student outputs with clean SD2.1 teacher outputs, not real COCO. CLIPScore=2.5*max(cosine(image, text),0), CLIPViT-B/32. Teacher LPIPS uses a different reference(clean SD2.1), so its magnitude must not be ranked against student LPIPS.

| Transfer90 minus | TPR delta pp /95%CI | LPIPS delta /95%CI | Gained/lost detection | Images LPIPS improved |

| --- | --- | --- | --- | --- |

| Baseline k=1 | +22.8 [+20.0,+25.6] | -0.01130 [-0.01666,-0.00595] | 240/12 | 56.7% |

| Structure12500 | +6.7 [+4.9,+8.6] | +0.01033 [+0.00471,+0.01601] | 78/11 | 45.9% |

| Structure7500 | +10.4 [+8.4,+12.4] | +0.02258 [+0.01706,+0.02804] | 112/8 | 35.0% |

| Projected native 7500 | +18.6 [+16.1,+21.1] | +0.01693 [+0.01151,+0.02249] | 193/7 | 39.0% |

Paired percentile bootstrap: 5000 prompt-index resamples, seed20261027. Generation settings and clean-reference hashes match for the main three-way comparison. These intervals describe variation across sampled prompts for fixed trained models; they do not measure retraining-seed variation or correct adaptive test reuse.

## 5. Frozen detector, clean controls and bit recovery

Student threshold tau=0.8541307449340821, calibration target FPR1% on 5000 clean prompts. The empirical held-out FPR is0.8%(8/1000)for unadapted SD1.4 and0.2%(2/1000)for the historical clean-trained 7500 student. Their Wilson95%CIs are[0.406,1.571]% and[0.055,0.726]%. A matched clean-trained15000control is missing.

Transfer90 per-image bit accuracy is91.9125%; exactly all8bits are recovered in51.6%of images. Signs of the mean coordinates recover8/8bits(100%aggregate key accuracy). **These three quantities differ from97.8%single-image threshold detection.** A100%aggregate bit figure must not be presented as100% image-level detection.


---


## 6. Comparison with completed historical branches

Comparable-panel k=1 experiments: 1000 test images, the same student score threshold and metric reference conventions. Budgets and teacher recipes differ; rows are descriptive operating points. No attribution to one component follows from ranking this table.

| Student branch | Updates | TPR% | LPIPS | FID | KID x1000 | CLIP |

| --- | --- | --- | --- | --- | --- | --- |

| Baseline k=1 | 7500 | 75.0 | 0.550447 | 54.444 | 1.6665 | 0.787681 |

| Quality | 7500 | 80.8 | 0.546656 | 55.281 | 1.7606 | 0.785113 |

| Continuation | 7500 | 82.3 | 0.550218 | 54.887 | 1.4148 | 0.786429 |

| Native full | 7500 | 80.5 | 0.544096 | 54.605 | 1.2649 | 0.786889 |

| Projected native 7500 | 7500 | 79.2 | 0.522219 | 53.358 | 0.7950 | 0.785335 |

| Projected native 10000 | 10000 | 82.1 | 0.533119 | 53.453 | 0.7732 | 0.787516 |

| Structure7500 | 7500 | 87.4 | 0.516560 | 52.735 | 0.5454 | 0.788896 |

| Structure10000 warm restart | 10000 | 86.4 | 0.547930 | 53.851 | 0.9707 | 0.785565 |

| Structure12500 | 12500 | 91.1 | 0.528817 | 52.896 | 0.6842 | 0.786258 |

| scale085_7500 | 7500 | 42.7 | 0.533659 | 54.563 | 1.4637 | 0.787436 |

| scale085_long22500 | 22500 | 52.3 | 0.573231 | 54.858 | 2.0206 | 0.790023 |

| scale085_conv_restore | 7500 | 57.5 | 0.535404 | 54.230 | 1.3024 | 0.788586 |

| quality_gradient | 7500 | 80.7 | 0.543704 | 55.377 | 1.6724 | 0.786408 |

| quality_cka | 7500 | 80.5 | 0.548790 | 54.939 | 1.6421 | 0.784089 |

| quality_transfer | 7500 | 80.5 | 0.545900 | 55.087 | 1.4964 | 0.786084 |

| Transfer90 | 15000 | 97.8 | 0.539145 | 52.563 | 0.6847 | 0.786725 |

**Earlier99.70%result.**The historical lora_r32_conv_wm row in result_vast used a selected kappa=2 teacher and a different artifact/panel configuration: TPR99.70%, FID62.676, KID0.0069348, CLIP0.79334. It is preserved as historical context, not the matched k=1baseline. Student LPIPS is not supplied in that row. Therefore we cannot claim Transfer90 improves its LPIPS or reproduces its setup.

**Lessons supported by the runs.**Uniform scale0.85reduced detection sharply; longer student training did not consistently repair it. Native image objectives improved paired fidelity more reliably than the earlier proxy-only refinements. Structure 7500 retains better LPIPS than Transfer90; Transfer90 prioritizes inheritance and reaches a much stronger detection operating point. Neither outcome proves that all visual artifacts have been removed.


---


## 7. Teacher evidence and student learning curve

| Measure | Transfer90 teacher |

| --- | --- |

| Validation detection | 998/1000 (99.8%) |

| Test detection | 999/1000 (99.9%); teacher threshold0.8509314060 |

| Validation LPIPS / CLIP | 0.367454 / 0.781779; clean teacher CLIP 0.782060 |

| Interpretation | Teacher detection is strong, but is not a substitute for downstream student detection; teacher FID/KID not included in this exported completed summary. |

| Student checkpoint | Validation MSE | Validation TPR | Selected |

| --- | --- | --- | --- |

| 7500 | 0.146889691 | 94.7% | No |

| 10000 | 0.146826940 | 95.7% | No |

| 12500 | 0.146716820 | 96.3% | No |

| 15000 | 0.146566953 | 97.0% | Yes |

![Figure](../../output_artifacts/previews/transfer90_report_20261027/validation.png)

At 12500 updates, Transfer90 validation detection is96.3%, versus91.3% for Structure 12500 on validation. This reduces the concern that the entire difference appears only after the extra2500 updates, but remains a teacher-and-training-recipe comparison. A fixed 12500 Transfer90 test endpoint was not evaluated by this report; do not substitute validation96.3% for its test TPR.


---


## 8. Quality/detection trade-offs and limitations

![Figure](../../output_artifacts/previews/transfer90_report_20261027/pareto.png)

Transfer90 is strongest in detection among the displayed comparable branches; Structure 7500 lies farther left in LPIPS. FID values are close at this 1000-image sample size. Small KID/FID gaps are point estimates, not evidence of a statistically established ranking. KID subset standard deviation is not a confidence interval across training seeds.

**Visual audit scope.**The following20pages show the first 20 test prompts with no selection by attractiveness, score or detection. Each page includes clean/marked SD2.1teachers and unadapted SD1.4, then baseline k=1, Structure 12500 and Transfer90students. Teacher seeds use image IDs; student seeds are500000+index. Pairing holds within teacher and student families; teacher-vs-student noise is not identical. The gallery retains all120original512x512PNGs; PDF images use high-quality JPEG previews for portability.

**Limits.**One training seed and repeatedly inspected test prompts; no fresh confirmatory holdout. Teacher source, refinement duration, student budget and LR schedule change together. The frozen proxy is a surrogate rather than differentiated full student training. No matched 15000 clean control, robustness-to-distortion sweep or systematic human preference/anatomy evaluation is completed. Per-image LPIPS is reference distance; altered composition can raise it even if an image is plausible.

**Supported statement.**Under the current seed-0 protocol, Transfer90 preserves strong image-level semantic detection through ordinary LoRA adaptation while improving LPIPS/FID/KID relative to the matched k=1baseline. It raises detection above Structure12500at the cost of worse paired LPIPS. The evidence supports a stronger inheritance operating point, not a universal image-quality improvement or a reproduction of the historical kappa=2result.

## 9. Evidence, reproducibility and next checks

The evidence bundle includes completed student/teacher JSONs, verifier report, baseline summary, source code, all generation manifests,20 matched prompts and SHA256checks for120 images. Student adapter SHA256: 960564f91f2d132a139ff5beb7c3b202f8955fe18f0f488d57dc3473c8b7adb9. The snapshot was selected only by validation MSE0.1465669530. Current server location:/var/lib/wmq/logs/project_native_transfer90/runs/run_20261026_114758_661241/transfer90. Archived manifests retain the old/root/logs paths; relocation requires path repair before rerunning.

Priority validation: repeat seeds and a fresh holdout; add matched 15000 clean adaptation; compare fixed budgets and leave-one-component-out teacher ablations; perform blinded human comparisons of the same prompt/seed panel. Checkpoint averaging was cancelled and is not part of these results.

Primary methodological references:
[1]Clark et al., Directly Fine-Tuning Diffusion Models on Differentiable Rewards(ICLR2024), arxiv.org/abs/2309.17400.
[2]Yu et al., Gradient Surgery for Multi-Task Learning(NeurIPS2020), arxiv.org/abs/2001.06782.
[3]Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric(CVPR2018), arxiv.org/abs/1801.03924.
[4]Heusel et al., GANs Trained by a Two Time-Scale Update Rule...(NeurIPS2017), arxiv.org/abs/1706.08500.
[5]Binkowski et al., Demystifying MMD GANs(ICLR2018), arxiv.org/abs/1801.01401.
Channel motivation: HMARK(arxiv.org/abs/2512.00094)and Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models(arxiv.org/abs/2506.23731). Their reported metrics are not directly comparable to this experiment.


---


## 10.1. Paired visual comparison: sample00

**Prompt:**A battered refrigerator stands open in a dirty room.

Teacher seed(image ID): 45663; student seed: 500000. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000000.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000000.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000000.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000000.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000000.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000000.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.2. Paired visual comparison: sample01

**Prompt:**a person cutting a pink cake with pink roses

Teacher seed(image ID): 209548; student seed: 500001. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000001.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000001.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000001.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000001.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000001.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000001.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.3. Paired visual comparison: sample02

**Prompt:**A man in a green shirt stands by a girl holding a piece of cake on a plate.

Teacher seed(image ID): 313034; student seed: 500002. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000002.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000002.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000002.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000002.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000002.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000002.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.4. Paired visual comparison: sample03

**Prompt:**Two children under a &quot; Sesame Street&quot; sign on a light post.

Teacher seed(image ID): 217137; student seed: 500003. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000003.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000003.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000003.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000003.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000003.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000003.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.5. Paired visual comparison: sample04

**Prompt:**The red plate holds rice, tofu, broccoli and water chestnuts.

Teacher seed(image ID): 205435; student seed: 500004. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000004.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000004.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000004.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000004.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000004.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000004.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.6. Paired visual comparison: sample05

**Prompt:**A man holding a Nintendo Wii controller next to a pretty young girl holding another Nintendo Wii controller.

Teacher seed(image ID): 335785; student seed: 500005. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000005.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000005.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000005.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000005.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000005.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000005.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.7. Paired visual comparison: sample06

**Prompt:**A sad, young girl sits on her bed, moping.

Teacher seed(image ID): 118542; student seed: 500006. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000006.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000006.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000006.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000006.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000006.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000006.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.8. Paired visual comparison: sample07

**Prompt:**Two tall teddy bears standing next to each other with a person kneeling between them.

Teacher seed(image ID): 227801; student seed: 500007. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000007.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000007.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000007.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000007.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000007.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000007.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.9. Paired visual comparison: sample08

**Prompt:**a couple of cows stand on some grass

Teacher seed(image ID): 576564; student seed: 500008. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000008.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000008.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000008.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000008.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000008.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000008.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.10. Paired visual comparison: sample09

**Prompt:**Several stuffed animals handing from a metal rail.

Teacher seed(image ID): 208844; student seed: 500009. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000009.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000009.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000009.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000009.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000009.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000009.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.11. Paired visual comparison: sample10

**Prompt:**A baby girl standing in a shopping cart holding an umbrella.

Teacher seed(image ID): 443969; student seed: 500010. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000010.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000010.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000010.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000010.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000010.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000010.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.12. Paired visual comparison: sample11

**Prompt:**A bus travels on a road near buildings and parked vehicles

Teacher seed(image ID): 334405; student seed: 500011. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000011.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000011.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000011.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000011.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000011.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000011.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.13. Paired visual comparison: sample12

**Prompt:**a small child is standing on a skateboard

Teacher seed(image ID): 507935; student seed: 500012. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000012.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000012.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000012.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000012.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000012.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000012.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.14. Paired visual comparison: sample13

**Prompt:**A dog is standing in floodwater beside a house.

Teacher seed(image ID): 184606; student seed: 500013. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000013.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000013.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000013.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000013.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000013.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000013.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.15. Paired visual comparison: sample14

**Prompt:**A room filled with furniture and hardwood floors.

Teacher seed(image ID): 15947; student seed: 500014. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000014.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000014.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000014.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000014.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000014.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000014.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.16. Paired visual comparison: sample15

**Prompt:**A street containing various cars and lampposts at night.

Teacher seed(image ID): 372316; student seed: 500015. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000015.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000015.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000015.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000015.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000015.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000015.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.17. Paired visual comparison: sample16

**Prompt:**Two elephants are near a watering hole at the same time

Teacher seed(image ID): 309883; student seed: 500016. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000016.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000016.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000016.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000016.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000016.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000016.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.18. Paired visual comparison: sample17

**Prompt:**A wooden bench and a fence with some flowers.

Teacher seed(image ID): 540782; student seed: 500017. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000017.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000017.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000017.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000017.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000017.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000017.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.19. Paired visual comparison: sample18

**Prompt:**A crowd of people standing near a fire hydrant.

Teacher seed(image ID): 551315; student seed: 500018. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000018.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000018.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000018.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000018.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000018.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000018.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.


---


## 10.20. Paired visual comparison: sample19

**Prompt:**A toddler with a fork in her mouth and a picnic table and people behind her.

Teacher seed(image ID): 1667; student seed: 500019. Teacher-vs-student seeds differ. First 20 test samples; no visual filtering.

**Clean SD2.1 teacher**

![Clean SD2.1 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/clean_teacher/sample_000019.png)

**Transfer90 teacher**

![Transfer90 teacher](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_teacher/sample_000019.png)

**Unadapted SD1.4**

![Unadapted SD1.4](../../output_artifacts/previews/transfer90_report_20261027/images/unadapted/sample_000019.png)

**Baseline k=1 student**

![Baseline k=1 student](../../output_artifacts/previews/transfer90_report_20261027/images/baseline_k1/sample_000019.png)

**Structure12500 student**

![Structure12500 student](../../output_artifacts/previews/transfer90_report_20261027/images/structure12500/sample_000019.png)

**Transfer90 student**

![Transfer90 student](../../output_artifacts/previews/transfer90_report_20261027/images/transfer90_student/sample_000019.png)

Score measures watermark evidence, not image quality. Inspect composition, faces, hands, textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.
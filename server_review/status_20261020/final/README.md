# Experimental UNet watermark results

One fixed detector threshold. Clean rows measure false positives; marked rows measure detection.

| Model | Detected | Rate | Bit accuracy | 95% Wilson CI | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| teacher_clean_test | 3/1000 | 0.3% | 32.04% | 0.1%–0.9% | -2.7941538405684696e-05 | -0.0007755977555110371 |
| teacher_wm_test | 72/1000 | 7.2% | 52.50% | 5.8%–9.0% | 59.62865976946165 | 0.007582258502004042 |
| unadapted | 3/1000 | 0.3% | 32.04% | 0.1%–0.9% | -2.7941538405684696e-05 | -0.0007755977555110371 |
| lora_r16_wm | 19/1000 | 1.9% | 40.23% | 1.2%–2.9% | 49.92580598597607 | 0.0018449461472946015 |
| lora_r32_wm | 19/1000 | 1.9% | 41.48% | 1.2%–2.9% | 50.852356474058126 | 0.002293458366733483 |

Differences versus unadapted (not clean-trained adjusted): see report.json. Passing gates is descriptive, not proof of causal inheritance.

- Frozen 8-bit LALT reader reused; known key221 is a development choice, not an unseen-key generalization result.
- One seed; pilot search cost is additional compute. Repeat seeds and independent keys before paper claims.
- Teacher uses on-policy DDIM with truncated tail gradients; transfer uses a short predicted-x0 proxy, not full student training.
- Semantic shifts can alter content; inspect images and quality metrics, not just TPR.
- Wrong keys are a finite negative-control audit, not a security guarantee.
- All final students start from the CLEAN base, not the marked teacher; no key or detector loss in student training.
- No feasible positive transfer on pilot; using the strongest teacher positive control for diagnosis only.
- No final clean-trained or fixed-teacher students: differences versus unadapted cannot isolate watermark inheritance from generic fine-tuning effects or establish a selection benefit.
- Teacher watermark failed held-out positive control; low student detection cannot isolate a LoRA transmission failure.
- TEACHER_INVALID: teacher detection/CI/quality/FPR guard failed; no successful inheritance claim.

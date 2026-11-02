# C10/C11 implementation and reproduction review

Reviewed: 12 October 2026. Scope: current author-key LoRA and full-UNet branches, including the running fixed-step-3000 evaluation. Server: `/root/workspace/serum`.

**Verdict:** the inspected training/data/scoring paths are internally consistent, and the previous exhausted-iterator clipping defect is absent. Exact reproduction is still unestablished. The main actionable issues are the full-training budget/selection definition, residual input-latent signal in CtrlRegen+, and incomplete prompt-disjointness checks. None of these observations alone establishes the cause of the LoRA TPR discrepancy.

## Findings, ordered by scientific impact

### 1. The full-training reproduction budget and selection policy need separate labels

The [SERUM paper, Appendix I and Figure 4](https://adam-dziedzic.com/static/assets/papers/serum.pdf) specifies SD1.4 adaptation with 5,000 watermarked training images, 1,000 validation images, batch 8, full LR `5e-6`, and LoRA rank 4/LR `1e-5`. Its full-training curve ends at 5,000 steps; minimum validation loss occurs at 3,000 (77.12% TPR). The LoRA minimum is at 19,500 (52.30%). Checkpoint selection uses unseen denoising MSE.

Our full queue searches through 20,000 nominal steps. This is an extended-budget experiment. The current step-3000 queue uses a user-requested fixed checkpoint, not the local validation optimum.

| Full checkpoint | C10 validation MSE | C11 validation MSE |
|---|---:|---:|
| 1,000 | 0.1511757320 | 0.1467298529 |
| 2,000 | 0.1506252545 | 0.1465141289 |
| 3,000 | 0.1492988908 | 0.1464058998 |
| 4,000 | 0.1379797386 | 0.1463264702 |
| 5,000 | **0.1289769928** | **0.1462460377** |
| 20,000 | 0.1211553759 | 0.1458756525 |

Among saved checkpoints up to 5,000, both local minima are at 5,000. A budget-matched, selection-rule-matched comparison therefore needs evaluation of those checkpoints as well. No retraining is necessary. The 1,000-step cadence is coarser than the published figure; it cannot identify an unrecorded intermediate optimum.

C10 and C11 use different, condition-specific validation images, as requested by proposal §17. Their MSE values rank checkpoints **within** a condition; they are not directly comparable measures of cross-condition utility. Declining C10 validation MSE does not by itself prove or disprove memorization. Assess nearest-neighbor similarity and outputs on unseen prompts before making that claim.

Code: `serum/scripts/queue_full_author.py:20`, `:80`; `select_full_checkpoint.py:21`, `:60`; `evaluate_full_step3000.py:26`.

### 2. CtrlRegen+ strength 1.0 retains a direct input-latent path

The current wrapper matches the public CtrlRegen+ notebook: same base model, DINOv2 encoder, controls, fixed positive/negative prompts, 50 steps, CFG 2, UniPC scheduler, and color matching. This is consistent with proposal §§8.1/8.3; it is not an accidental departure from that chosen operator.

However, the installed Diffusers 0.30.2 `StableDiffusionControlNetImg2ImgPipeline.prepare_latents` always encodes the input and calls `scheduler.add_noise`, including at strength 1.0. On the actual cached scheduler:

```text
t_start = 999
alpha_bar = 0.00466009508818388
z_start = 0.0682648891 * z_image + 0.9976672316 * epsilon
```

Thus strength 1.0 is not exactly pure Gaussian initialization. The coefficient is a latent signal amplitude, **not** a measured watermark-retention percentage. SERUM's Appendix C describes its CtrlRegen evaluation as initialization from clean Gaussian noise. Equating our CtrlRegen+ endpoint with that configuration is unsupported.

**Recommended experimental improvement:** add a separately named pure-noise initialization variant with the same semantic/spatial controls. Verify that its initial latent is independent of the image, hold its noise fixed across paired clean/watermarked inputs, then run a preservation-only pilot before any downstream training. Keep current results as the CtrlRegen+ baseline. Pure-noise initialization may help but is not guaranteed to remove watermark information conveyed by the control branches.

Color matching also accesses the source image after regeneration. A separately labeled on/off ablation can test its contribution; there is currently no evidence that this operation causes the observed radioactivity.

Code: `serum/scripts/launder_ctrlregen.py:166`, `:184`; `serum/third_party/CtrlRegen/custom_i2i_pipeline.py:6`; installed source and numerical coefficients are archived in `report_seed0_evidence/c10_c11_regen_numerics_20261012.json`.

### 3. Exact-string split separation misses equivalent conditioning

All four splits have the expected sizes and no exact string intersections. Case/whitespace normalization finds three overlapping evaluation prompts. Checking the complete padded/truncated SD1.4 token sequence identifies **10 distinct evaluation indices** overlapping another split:

```text
48, 145, 309, 406, 1042, 2550, 2832, 2897, 3885, 4246
```

Four overlap training, two validation, and four calibration. One additional validation/train pair has identical SD1.4 tokens. These are conditioning collisions, not proof of identical image leakage: the audit found zero exact raw train/validation image-hash overlap. Near-duplicate visual content has not been measured.

The following sensitivity analysis excludes all ten evaluation indices, retaining the original frozen threshold and score arrays:

| Student | Original n=5,000 | Restricted n=4,990 |
|---|---:|---:|
| LoRA C00 | 2.200% | 2.204% |
| LoRA C01 | 1.900% | 1.904% |
| LoRA C10 | 4.420% | 4.429% |
| LoRA C11 | 3.600% | 3.607% |
| Full C10, 20k | 97.960% | 97.976% |
| Full C11, 20k | 93.140% | 93.146% |

This issue does not explain the large detection discrepancy. The restricted table is supplementary; it does not repair the existing validation/calibration intersections or redefine the original experiment. Future datasets should be deduplicated before allocation using normalized text and the actual model tokenizers, with semantic near-duplicates audited separately. Existing frozen prompts/checkpoints/thresholds were not overwritten.

Evidence: `report_seed0_evidence/c10_c11_prompt_review_20261012.json`.

### 4. Selection and provenance can be strengthened without detector tuning

The selectors use one latent-posterior draw and one noise/timestep draw per validation image, fixed across checkpoints. This is reproducible but does not quantify ranking uncertainty. For close candidates, use a predeclared set of additional draws or timestep strata and save per-image losses/paired confidence intervals in a separate validation version. Do not choose that protocol based on which checkpoint has favorable TPR.

The launch manifests hash `metadata.jsonl`; that file contains filenames/captions, so equal C10/C11 metadata hashes are expected and do not prove image identity. The independent inventory hashes do verify image provenance. Future preflight should enforce those image hashes, source IDs, tokenizer-disjoint splits, and checkpoint hashes at stage boundaries. Resume checks should also pin model/control weights and operator configuration, beyond seed/path/strength.

Log gradient norms before/after clipping, successful optimizer updates, and mixed-precision skipped updates. This would make future clipping/overflow diagnosis direct rather than retrospective.

## Checks that passed

- SHA-256 comparison: all 11 inspected local runtime files match the server, including both trainers, selectors, generation, scorer and laundering wrapper.
- All 6,000 C10 source images and 6,000 paired C11 regenerated images match their archived source/output inventory hashes. None of the paired raw/regenerated PNGs are identical.
- All 10,000 training image references and captions match their intended sources/prompts. Raw train and validation contain 5,000 and 1,000 unique hashes, respectively, with zero shared hashes.
- Full C10/C11 launch arguments match except dataset/output path. All UNet parameters train; VAE and text encoder are frozen. Sampled VAE latents, scaling, uniform timesteps and the scheduler-dependent denoising target are used. No random horizontal flip is enabled; 512-to-512 cropping does not change these inputs.
- Full clipping obtains a fresh `unet.parameters()` iterator on each call. LoRA uses the corrected reusable parameter list. Two successive CPU `Accelerator` backward/clip/update tests for each pattern clipped norms to 1.0. This verifies the code pattern, not historical per-step norm values, which were not recorded.
- Saved optimizer state confirms LR `5e-6` for full and `1e-5` for LoRA, AdamW betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`. Full C10/C11 each contain 19,997 successful updates at nominal step 20,000; LoRA C10/C11 contain 19,997/19,998. A few AMP-skipped updates are consistent with these counts and cannot plausibly explain the very large detection difference; the exact overflow events were not logged.
- Teacher normalization/injection and detector VAE encoding/scaling agree with the inspected official SERUM implementation. The detector uses the author checkpoint and SD2.1 VAE, and the common frozen threshold is `0.06189660727977753`.
- The student generator loads the requested full UNet; generation manifests and the step-3000 audit check checkpoint identity and prompt hashes. Student generation contains no explicit watermark injection.

## Reproduction limits and next priorities

The available official SERUM release does not establish the exact downstream training recipe for optimizer defaults, LoRA target modules/alpha, precision, sampler, checkpoint cadence, or original image/prompt instances. Our choices are documented, but should not be described as verified author settings. Altering these to obtain a desired TPR would be a new experiment, not evidence that the present code was wrong.

FID/KID currently use 5,000 clean teacher outputs as the reference. They measure teacher-distribution similarity and should not be compared numerically with the paper's COCO-reference FID. Full clean C00/C01 controls, author-key regenerated-teacher detector/utility measurements, and independent adaptation seeds are still missing for a complete causal/robustness claim.

Recommended order:

1. Finish the already running fixed-3k comparison; include the restricted-prompt sensitivity result.
2. Evaluate existing 5k full checkpoints as the local minimum-MSE candidates within the saved 1k–5k budget. Report fixed-step and selected-within-budget results separately.
3. Complete author-key regenerated-image preservation/detection measurements. Do not reuse old-key pilot TPR as author-key evidence.
4. Pilot the explicitly pure-noise operator and a color-matching ablation. Freeze utility criteria before examining downstream detector results.
5. If a promising effect remains, add clean full controls and independent seeds under the chosen protocol.

Only read-only probes and supplementary audit artifacts were added during this review. Running generation, model weights, frozen evaluation sources, prompts and thresholds were not modified. No additional training was launched.

Evidence bundles: `c10_c11_code_review_evidence_20261012.json`, `c10_c11_optimizer_review_20261012.json`, `c10_c11_regen_numerics_20261012.json`, and `c10_c11_prompt_review_20261012.json`, under `report_seed0_evidence/`. Probe scripts are preserved under `server_review/scripts/review_c10_c11_*.py` and on the server.

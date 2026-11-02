# Corrected core rerun: raw versus strong only

Updated October 11, 2026, following the user's instruction to remove mild runs.

## Scientific scope

The target remains pre-adaptation output laundering: teacher outputs are regenerated **before** SD1.4 LoRA training, and SERUM audits the resulting student's newly generated outputs. The selected strong setting is the original pilot choice (strength 1.00), not a strength reselected from downstream detector scores.

The corrected experiment retains the full 2×2 design: C00 clean/raw, C10 watermarked/raw, C01 clean/regenerated, C11 watermarked/regenerated. Removing mild narrows the strength ablation/Pareto sampling but does not remove a causal control. All runs use seed 0, the same base revision, rank 4, LR 1e-5, batch 8, and 20,000 nominal training steps. Checkpoints are chosen by minimum unseen denoising MSE on the corresponding 1,000-image validation set. Evaluation uses the same 5,000 prompts/seeds and frozen threshold.

## Execution schedule

| GPU | Order |
|---|---|
| 0 | Existing `C10_seed0_clipfix` completes training, validation, 5,000-image generation, SERUM and utility; then `C11_step1.00_seed0_clipfix` runs the same stages. |
| 1 | `C00_seed0_clipfix`, then `C01_step1.00_seed0_clipfix`, each through training and complete evaluation. |

No C01/C11 mild jobs are scheduled. Old artifacts are preserved and excluded from the corrected 2×2 summary. Once the four corrected runs complete, the queue writes `metrics/clipfix_strong_summary.json`, paired confidence intervals, score interactions, ROC/AUC and histograms, raw/strong frontier, and fixed-prompt qualitative grids. KID/FID use the existing 5,000-image clean teacher reference, not COCO FID.

This narrowing was requested after the pre-fix results had been observed. The existing mild findings remain documented in the original reports; the strong-only rerun should be described as an exploratory scope change, not a preregistered full strength sweep.

## Checks performed before launch

- Re-read the proposal, including data split rules, fixed CtrlRegen prompt, 2×2 controls, validation selection, fixed detector protocol and claim gates.
- Full current audit decoded and hashed 36,000 raw/regenerated train/validation files (including already existing mild datasets); checked every HF training row against source bytes and prompt, regeneration provenance/seed/strength, and four unique/disjoint prompt splits. The active raw/strong subset contains 24,000 image files across train/validation conditions.
- Checked the current trainer hash equals the tested correction and the running C10 manifest. Recorded inference/selection/scoring script hashes, prompt hashes, detector/key hashes, threshold, pilot policy, and SD1.4 cache revision `133a221b8aa7292a167afc5127cb63fb5005638b`.
- All three environments passed dependency consistency checks.
- Actual four-step fp16 training exercised clipping on all 256 optimizer parameters, saved checkpoints, verified nonzero learned LoRA B tensors and finite weights, and loaded the checkpoint for inference.
- Actual gradient norms in this short test were approximately 0.011–0.017, below the clipping limit 1.0. Thus the iterator defect is real, but fixing it need not materially change the TPR; its impact later in training remains to be measured.
- Visually inspected 50 deterministic random prompt/source rows with regenerated counterparts. No sampled row mispairing found; some teacher images incompletely follow complex prompts, documented separately. This is not a utility-equivalence test.

## Execution protections

The queue runs in tmux, uses exclusive worker locks, waits for full C10 completion before using GPU 0, and writes atomic heartbeat/status files. Hash mismatches, failed stages, missing/nonfinite checkpoints, wrong validation source or count, incorrect selected checkpoint, or invalid evaluation outputs stop execution with a recorded failure. All 40 checkpoints are retained for selection. A partial training directory requires explicit recovery; the queue does not silently start a different/resumed experiment or change hyperparameters. GPU 1's pre-existing service is retained.

## Boundaries of this assurance

These checks substantially improve implementation assurance but cannot prove that no defect remains. The original author SERUM key/artifacts and complete radioactivity recipe have not been reproduced exactly. A weak corrected C10 would still prevent a strong attack claim. The user has explicitly authorized the corrected exploratory core rerun despite that pending baseline gate; the queue never creates a scientific baseline pass automatically.

Three-seed confirmation, additional operators/students, mechanism analysis and defense remain conditional. Mild ablations are excluded by the latest user instruction. The new summary must not mix pre-fix and corrected students.

## Records

Server: `manifests/clipfix_core_protocol.json`, `manifests/clipfix_core_inventory.json`, `metrics/clipfix_training_preflight.json`, `manifests/clipfix_visual_audit/review.json`, and per-run manifests/status files.

Queue: `scripts/queue_clipfix_strong.py`; tmux session `quant-clipfix-strong`; status `metrics/clipfix_strong_queue_status.json`; worker logs `logs/clipfix_strong_worker0.log` and `logs/clipfix_strong_worker1.log`.

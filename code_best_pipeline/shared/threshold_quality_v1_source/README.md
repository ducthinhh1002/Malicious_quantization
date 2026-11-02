# Threshold quality v1 — frozen detector, adapter strength 0.85

**Policy revision 2026-10-23, explicitly requested by the user:** LPIPS and CLIP
are report-only. Their old acceptance checks below are retained as historical
diagnostics, not stopping conditions. Selection now minimizes LPIPS among
scale-0.85 candidates retaining baseline detections. Final evaluation writes
`status: evaluated` and `quality_gates_enforced: false`, including all metrics
even when the original quality targets fail. Detection and finite/integrity
checks remain. See `run/policy_revisions/` on VM4 for old/new source hashes,
archived code, tests and checkpoint-preserving restart evidence.

Additional 0.75/0.65 strength diagnostics are implemented in `strength_sweep.py`;
these report results separately and do not silently alter the selected student.
Research findings: [quality_research_20261023.md](docs/quality_research_20261023.md).

## Queued extension, 2026-10-23

### Superseding kappa-1 campaign

The user stopped the kappa-2/Pareto queue on 2026-10-23 and requested improvements
from kappa=1. `kappa1_campaign.py` now runs the existing verified kappa-1
checkpoint-3000 at scale 1, then 0.95/0.85 in parallel, then the group vector
`[0.85,0.85,1,1,0.85,1]` in down-attention/down-conv/mid-attention/mid-conv/
up-attention/up-conv order. This conservative selective proposal is informed by
the previous attribution experiment, not by this campaign's final test results.
Each improved student is compared with the newly completed kappa-1 scale-1
student. The first baseline report additionally compares against the old VM4
kappa-2 control; its baseline lineage explicitly records this distinction.

Artifacts are under `kappa1_campaign/`; launch with
`/root/logs/gate_env/bin/python kappa1_campaign.py --launch` from the VM4 root.
Old checkpoint/results remain. Only reconstructible merged exports in the
stopped queue were released after preserving their merge proofs and recipes.
Existing kappa-1 scale-1/0.85 validation images and native models are verified
and reused. Clean images, prompts, carrier and frozen thresholds are shared.
New teachers require new training images; kappa-2 training images are not
substituted for kappa-1 data. Keep ordinary sampling, seed, optimizer and
7,500-step rank-32 attention+conv student protocol unchanged.

History supports modest attenuation: kappa-2 scale .65 gave student TPR .869,
.85 gave .997, and selective KernelSHAP gave .968. Therefore the new fixed
sweep adds .95 and omits .65. The historical result remains the success target:
TPR .997, FID 62.676, KID .0069348, CLIP .79334. The current kappa-1 teacher's
LPIPS .676267 is not a reproduction of historical .488; missing historical
checkpoint/provenance remains unresolved and must not be hidden by comparisons
against a weaker recreated baseline.

Research review: [AquaLoRA](https://arxiv.org/abs/2405.11135) supports investigating
prior preservation, while [SIREN](https://arxiv.org/abs/2410.10437) explicitly
addresses watermark learnability in downstream personalization. Neither proves
our fixed carrier will retain transfer after attenuation. These findings favor
matched student evidence and retaining carrier structure before introducing a
new loss. This campaign implements controlled attenuation, not PPFT or SIREN.

`research_queue.py` adds an isolated, persistent queue after the current main run:

1. Two parallel SD1.4 students trained from the **same kappa=2, step-3000 teacher**
   at scales 0.65 and 0.85. Both retain rank 32 attention+conv, 1,000 training
   images, 7,500 steps and snapshots 2,500/5,000/7,500. Their checkpoint selection
   remains held-out denoising MSE. Reuse the verified main scale-1 student as the
   paired baseline, with the same frozen student calibration threshold.
2. KernelSHAP-inspired group attribution: six groups (down/mid/up x attention/conv),
   with input/output boundary convolutions assigned to down/up. Measure all 64
   coalitions between strengths 0.65 and 0.85 on 32 paired validation prompts.
   Fit endpoint-constrained KernelSHAP WLS for LPIPS, CLIP drop and p10 detector
   margin, with 200 paired-image bootstrap replicates. Validate the proposed
   group vector on 256 images and train its student if it retains detections.
3. MPaGE-inspired Pareto-grid crossover/mutation: common initial vectors, 16 new
   evaluations for search and 16 for a seeded random control, at equal oracle
   budgets. Search LPIPS, CLIP drop and negative p10 margin. Fully validate up to
   three Pareto anchors and train the detection-preserving minimum-LPIPS choice.
   These are adaptations of the ideas, not full NAS or LLM-method reproductions.

Every scheduled quality evaluation finishes and is saved. Failed detection in
a research shortlist is recorded, does not run its student, and does not cancel
the following independent queue job. Exact repeated student vectors reuse the
completed result. Test scores never enter proposal generation or selection.
Final student reports include TPR, paired lost positives, LPIPS with paired-image
bootstrap intervals, CLIP, FID/KID, source hashes and the MSE-selected checkpoint.

Queue artifacts: `queued_research_20261023/{plan,status,code_manifest}.json`,
`supervisor.log`, per-stage logs, attribution, Pareto archive, frozen selections,
and `students/*/result/student_comparison.json`.

Initialize once on VM4, then launch; the launcher refuses a duplicate live queue:

```bash
/root/logs/gate_env/bin/python research_queue.py --initialize /root/logs/threshold_quality_20261022/run
/root/logs/gate_env/bin/python launch_research_queue.py
```

The supervisor waits for `run/complete.json` before using shared baseline results.
It polls every 30 seconds, and waits for disk space below 12 GiB rather than
discarding checkpoints. Completed byte-identical model exports are hardlinked,
with an audit log. New temporary merged models are released after evaluation;
the original checkpoint, recipes, merge-equivalence proofs, images, measurements,
selected student adapters and periodic student snapshots are preserved. Pilot
adapters are reproducible from the immutable source and saved group vector.

Validation: `python -m unittest test_group_search -v`; VM4 smoke evaluation uses
two images in a separate `queue_preflight_20261023` directory and is excluded
from research selection. This smoke result is not evidence of image quality.

Isolated fork of `t2ifing_radioactive`. Original source is untouched. The actual
`threshold_cfg_v1` training objective, 8-bit carrier, CFG 7.5, native scheduler,
50 inference steps, key seed, calibration rule and sampler are unchanged.
Execution batching is recorded separately from the effective optimizer batch.

The historical successful checkpoint/carrier are unavailable in this workspace
and on the inspected VM4 paths. This campaign recreates a matched baseline;
it does **not** claim identical historical carrier bytes or threshold values.
The recreated carrier/key and family-specific thresholds are frozen and shared
by every candidate and baseline in this run. RSSF artifacts are never reused.

## Predeclared campaign

1. Train the original kappa 1, 1.5 and 2 teachers, rank 32 attention+convolution,
   3,000 steps, effective batch 4, learning rate 1e-4, bf16.
2. Fixed reference: kappa 2 at adapter scale 1. It must reach validation TPR >=99%.
3. Three candidates: each original adapter at scale **0.85**. Scale only B;
   scaling both A and B incorrectly gives 0.7225.
4. Two exploratory candidates: equal-weight mixtures of kappa 2 with kappa 1.5
   or 1. Each mixture is also scaled by 0.85. Concatenate A along rank and
   weighted B along rank: BA equals the exact weighted sum, without cross terms.
   The temporary mixed teacher adapter has rank 64, but is merged into the
   ordinary base UNet before inference. Student rank stays 32.
5. Evaluate all candidates on the same validation prompts and Gaussian seeds.
   Retain for downstream training only if LPIPS improves at least 2%, a paired
   bootstrap lower bound on improvement is positive (one-sided alpha .01 for
   five fixed comparisons), CLIP drop <=.01 and no worse than baseline, and
   **no image detected by baseline becomes undetected**. Select lowest LPIPS
   among eligible candidates. Never change the frozen threshold to pass a gate.
6. Train matched SD1.4 rank32 attention+conv students from reference and selected
   teacher images, same seed/budget/hyperparameters, snapshots 2500/5000/7500.
   Original held-out denoising-MSE checkpoint selection is retained.
7. Final fixed test: detection, FID/KID/CLIP and paired LPIPS to unadapted SD1.4.
   `result/acceptance.json` applies the same conservative quality+detection rule
   to students. No test-guided reselection or automatic retry. A rejected result
   remains an audit artifact; it is not published as an improved model.

All adapters must pass native merge/reload output equivalence before generation.
Gate failures stop expensive downstream work. A completed pipeline alone is not
an accepted improvement: check `result/acceptance.json`. Teacher-only success
does not establish that the watermark transfers to students.

LPIPS here measures paired perceptual distance to the clean model, not absolute
human preference. CLIP is an additional semantic check; FID/KID and image review
remain necessary. Bootstrap bounds concern these images, not training-seed
uncertainty. Candidate reuse of validation makes final independent test essential.

## Why this additional experiment

[PEFT's exact concatenation merge](https://huggingface.co/blog/peft_merging)
provides an algebraically exact interpolation of LoRA updates. We use it to test
whether intermediate updates retain sufficient watermark margin with less
perceptual drift. This is an experimental hypothesis, **not** a published
watermark-preservation guarantee. Two fixed mixtures limit validation search.
[Diffusers adapter documentation](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters)
notes that some per-block scaling APIs only cover attention. Direct B scaling
also covers this experiment's convolution adapters, avoiding that omission.

[DIAGNOSIS](https://arxiv.org/abs/2307.03108) and
[SIREN](https://arxiv.org/abs/2410.10437) motivate checking learnability through
downstream adaptation. [DRaFT](https://arxiv.org/abs/2309.17400) can optimize
differentiable sampler objectives, but provides no preservation guarantee for
this carrier. This fork therefore keeps the original coherent-shift objective;
it does not reintroduce the failed trajectory/pixel-dominated training loss.

## Execution

Use `launch_vm4.sh` in the deployed campaign directory. Original optimizer batch
is 4; teacher microbatch 4, generation batch 16, scoring batch 8, student
inference batch 16. These increase GPU work per forward without changing
optimizer batch or enabling TF32. Workers remain sequential on the single GPU.
Existing unrelated GPU processes are left running. At least 12 GiB disk free
is required before starting each stage.

Tests: `python -m unittest test_threshold_quality test_semantic_threshold
test_semantic_numerics test_semantic_execution test_generation test_validation`.

## Parallel scale comparison requested on 2026-10-23

`parallel_scale_eval.py --source /root/logs/threshold_quality_20261022/run`
evaluates scale 1 and 0.85 concurrently while the main campaign continues.
It uses immutable checkpoints 2500 and 3000 from the same kappa=2 teacher.
Checkpoint 2500 gives an early diagnostic; checkpoint 3000 is queued until its
completed manifest appears. All comparisons use validation only.

Artifacts are isolated under `parallel_scale085/step_2500` and `step_3000`.
The original key/carrier, threshold, prompts, clean reference images and
checkpoint hashes are verified and reused. Both parallel workers use generation
batch 4 and scoring batch 4 to fit alongside active training. This execution
change is recorded in `lineage.json`; the main run config/source are untouched.
No separate scale-0.85 training is needed: it is exactly the same adapter with
its B factors multiplied by 0.85 before native merge.

Each step produces `result/baseline.json`, `result/scale085.json` and
`result/comparison.json` with detection, CLIP, LPIPS, paired detection losses,
and relative LPIPS improvement. These diagnostics do not change the frozen
teacher selection or final independent student acceptance in the main run.

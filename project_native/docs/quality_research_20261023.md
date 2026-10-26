# Quality recovery: evidence and next experiments, 2026-10-23

The user removed quality stopping gates. Complete the existing baseline/candidate
student comparison and save all detector and quality measurements. Numerical
integrity and detection preservation remain required. LPIPS/CLIP targets are
diagnostic, including failed targets, rather than reasons to stop training.

## What the measurements establish

At step 3000, on the same 256 validation prompts and seeds, kappa=2 scale 1
has LPIPS 0.68549795 and CLIP drop 0.06907608. Scale 0.85 has LPIPS 0.64158383,
CLIP drop 0.04202964, and retains all 256 baseline detections. Improvement is
6.406% in paired perceptual distance, not a 6.406% human-quality score.
Both still differ materially from the clean reference. Teacher success does
not prove inheritance by a separately trained SD1.4 student.

Kappa=1.5 scale 1 produces LPIPS 0.75961486 and CLIP drop 0.05677857, despite
256/256 detections. Retraining with a lower target margin has not yielded a
monotonic quality benefit. Post-training scale on the same weights is a more
controlled immediate experiment than assuming smaller kappa means better images.

The main run's generation batch 16 gives LPIPS 0.68547484 at kappa=2; the parallel
batch-4 evaluation gives 0.68549795. This small difference on identical weights
does not explain the gap to historical LPIPS 0.54189. It does not rule out effects
of different clean training batches/PCA, teacher microbatch or model assets.

## Original-code audit and uncertainty

The fddd62e-to-current git diff preserves the threshold teacher training formula;
changes to that training file are RSSF dispatch. The fork's original AST audit
also confirms unchanged `train_one`, threshold backward, predicted-x0, adapter
targets and carrier files against the copied original. The historical frozen
checkpoint/config/carrier remain unavailable, so exact reconstruction cannot
be asserted. Current teacher snapshot is sd2-community SD2.1, whereas the source
defaults name a Manojb SD2.1 snapshot. Architecture alone does not prove weight
identity. Current teacher microbatch is 4; the threshold protocol document names
1. Accumulation tests preserve objective math, but BF16 training trajectories
need not be bit-identical. These are open confounders, not established causes.

## Immediate experiment

### Completed strength sweep, 2026-10-23 07:24 ICT

The sweep has finished. Source artifact:
`output_artifacts/threshold_quality_20261022/parallel_step3000/strength_sweep.json`.
All rows use the same kappa=2 checkpoint at step 3000, the same 256 validation
prompt/seed pairs, and the frozen detector threshold 0.7413828969001771.

| Adapter scale | Paired LPIPS, lower is better | Relative LPIPS reduction | CLIP drop | Teacher detections | Mean detector score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.685498 | reference | 0.069076 | 256/256 | 1.684600 |
| 0.85 | 0.641584 | 6.41% | 0.042030 | 256/256 | 1.581095 |
| 0.75 | 0.606591 | 11.51% | 0.024847 | 256/256 | 1.475572 |
| 0.65 | 0.566414 | 17.37% | 0.014315 | 256/256 | 1.329006 |

This is evidence of reduced paired distortion and reduced CLIP degradation,
not a human quality percentage or proof of student transfer. Saturated TPR
hides a decreasing score margin: mean score minus threshold falls from 0.943217
to 0.587623. Consequently, a Pareto search using teacher TPR alone would miss
an important tradeoff. Save score quantiles and paired margins as well as TPR.
The old CLIP-drop target of 0.01 is still missed at 0.65; it remains diagnostic
and does not stop the run. The main campaign still completes its predeclared
0.85 candidate/student comparison; these exploratory scales do not replace it.

### Original sweep protocol

Measure scale 1, .85, .75, .65 from the same completed kappa=2 checkpoint using
the original frozen key/carrier/threshold, sampler and paired prompt/seed list.
Only B factors are scaled. The additional .75/.65 workers run concurrently and
record all results, even when quality or detection worsens. No test-set search.
Summaries include lost positives, score distribution, CLIP, paired LPIPS and
relative distance reduction. Bootstrap diagnostics use seven comparisons to
acknowledge the five original candidates plus two additional scales. Because
the new experiment was motivated by validation results, this is exploratory;
an independent fixed test and downstream student remain necessary.

Do not silently substitute a new scale into the running .85 campaign. Complete
its paired student baseline first. If lower scales preserve teacher detections,
evaluate downstream inheritance at the same rank, data size, optimizer budget
and independent test before calling them effective improvements.

## Mechanistic research direction

The original preservation term is half the conditional plus half the
unconditional noise-prediction MSE. At CFG 7.5 the perturbation actually driving
generation is `7.5 * delta_cond - 6.5 * delta_uncond`. By Cauchy-Schwarz its MSE
is bounded above by **197 times** that preservation term. This is a loose
mathematical bound, not a measured amplification factor. A small preservation
loss therefore does not itself establish small CFG drift or paired image drift.
Measure conditional, unconditional and CFG residuals on the same timesteps
before adding a CFG preservation term. Such a term also penalizes useful
watermark displacement and must not be increased without tracking detection.

A possible subsequent controlled ablation is modest CFG-aware preservation,
with the original coherent carrier shift retained and native-image evaluation
after a short fixed budget. No LPIPS/pixel-dominated objective is justified by
the present evidence. Another candidate is quality-constrained adjustment of
adapter groups; it requires measured sensitivity, not an assumption that all
convolution layers damage quality or all attention layers carry the watermark.
Neither new training objective has been deployed in this revision.

## Primary-source findings

[AquaLoRA, section 4.3.2](https://arxiv.org/html/2405.11135v1) uses frozen-model
noise predictions as targets for prior-preserving training with a latent offset.
Its sampling strength parameter provides an explicit fidelity/detection tradeoff.
This supports measuring a strength curve; its proof does not cover our semantic
PCA carrier or prove transfer to a fresh model trained on generated images.
Section 5.2 also distinguishes perceptual distance changes from distributional
quality: denoising may amplify paired differences while FID changes little.
Therefore finish FID/KID/CLIP and visual assessment rather than treating LPIPS
alone as absolute quality.

[SIREN](https://arxiv.org/abs/2410.10437) explicitly optimizes coating learnability
for personalization, alongside perceptual constraints. Its key implication here
is that downstream inheritance is a separate empirical requirement. Its coating
generator and verifier differ from this fixed semantic carrier, so importing
its losses or published numbers directly would not establish correctness here.

The next decision should be based on the completed strength curve and matched
student results. Quality targets are recorded objectives, not production gates.

## Huynh Thi Thanh Binh (HUST): applicable research

Implementation update after the user's follow-up: the isolated extension in
`research_queue.py` and `group_search.py` now implements fixed 0.65/0.85 students,
complete-coalition KernelSHAP-style group attribution, and Pareto-grid search
with a matched random-search oracle budget. It is queued after the current main
campaign, with full validation and student evaluation for selected research
candidates. See the README's queued-extension section for the frozen operational
protocol. The text below records the original research rationale and proposal.

Reviewed on 2026-10-23. Attribution was checked against the
[author's HUST publication list](https://users.soict.hust.edu.vn/binhht/publication.htm)
and each paper's author list. The adaptations below are our proposals; none of
these papers establishes improved radioactive watermark transfer for this model.
The deployed run does not implement these proposed research methods yet.

### 1. KernelSHAP-NAS: most directly relevant attribution idea

[KernelSHAP-NAS, Neural Computing and Applications, 2025](https://link.springer.com/article/10.1007/s00521-025-11071-2)
uses KernelSHAP to estimate operation contributions in a NAS supernet. The
publisher's abstract reports experiments on DARTS/CIFAR-10 and NAS-Bench-201.
Access was limited to the abstract and publisher preview; the full method was
not available. This supports the attribution idea, not a claim that we have
reproduced the algorithm or verified its detailed settings.

Proposed adaptation: treat populated adapter groups as explanatory variables.
Begin with down/mid/up regions crossed with attention/convolution targets,
giving at most six groups. For each group, use a low and high strength around
the measured operating range, for example 0.65 and 0.85. Estimate contributions
separately to paired LPIPS, CLIP drop and detector score margin. Keep A fixed
and multiply each group's B factors by its assigned coefficient. This preserves
the original adapter architecture and allows ordinary merged-model sampling.

This is attribution of changes between two scale settings, not attribution of
presence versus absence of a NAS operation. Interactions can reverse a ranking;
single-group ablations alone are not Shapley values. Do not assume convolution
damages quality or attention exclusively carries the watermark. Assess sampled
coalitions with fixed prompt/seed pairs and report uncertainty before using an
importance ranking to propose selective attenuation.

### 2. MPaGE: preserve useful tradeoffs during search

[MPaGE, sections 4.1-4.5, arXiv v3, January 2026](https://arxiv.org/html/2507.20923v3)
designs combinatorial optimization heuristics with LLMs. It uses a normalized
Pareto Front Grid, regional elites and semantic clustering to balance heuristic
quality, runtime and diversity. Its grid mechanism builds on prior work; neither
Pareto optimization nor NSGA-II originated with this paper's authors.

Proposed adaptation: archive every evaluated scale vector with LPIPS, CLIP drop,
detector margin and measured student transfer when available. Use nondominated
ranking to choose follow-up experiments, keeping both high-margin and low-drift
solutions. A plain Pareto archive is sufficient for the present four global
scales; grid partitioning becomes useful only with many block-scale candidates.
Borrowing this archive idea is not an MPaGE implementation and needs no LLM
code-generation loop. Quality ranking must not reinstate early-stopping gates:
finish scheduled evaluations and retain all measurements.

### 3. MF-LTGA: later, when block interactions and multiple tasks are measured

[A Multifactorial Optimization Paradigm for Linkage Tree Genetic Algorithm,
sections 3.2 and 4](https://arxiv.org/html/2005.03090v1)
combines multifactorial optimization with linkage-tree crossover. It constructs
task-specific trees so that related variables can be exchanged together rather
than independently.

Potential adaptation: linked groups of adapter coefficients could be crossed
together across related optimization tasks, such as separately evaluated teacher
checkpoints. LPIPS and detection are objectives of one candidate, not automatically
two multitasking problems. Four global-scale measurements cannot establish a
linkage model. Negative transfer and the cost of collecting a sufficiently large
population make this lower priority than controlled group ablations.

### 4. SCOPE: an expensive-search direction, not a new watermark loss

[SCOPE, sections 3.2 and 4, July 2026 preprint](https://arxiv.org/html/2607.27630v1)
constructs executable synthetic objectives from evaluation history. Their value
is assessed through the true quality of candidates produced by induced search
policies, rather than pointwise approximation of the real objective. A portfolio
of policies guides a limited black-box evaluation budget.

Potential adaptation: after collecting enough block-scale evaluations, compare
several proposal strategies and allocate expensive native generation/student
evaluations using observed utility. Such a proposal score must never replace
LPIPS, detector or student measurements. Current data are too small to justify
this machinery. The original method targets combinatorial optimization; applying
it here would require a new experiment and appropriate simple-search baselines.

### Concrete next experiment design (proposed, not launched)

1. Finish the running 0.85 versus baseline campaign and keep its student results.
   Test 0.65 and 0.75 for transfer under the same data size, rank, training budget,
   and independent test protocol before choosing a replacement.
2. Freeze the same kappa=2 teacher checkpoint, key, carrier, threshold, sampler,
   CFG and seeds for a separate group-scale experiment. Verify module coverage,
   exact B scaling and merge/reload equivalence for each adapter export.
3. Start with the two uniform endpoints and each group changed in isolation in
   both directions: at most 14 distinct candidates for six groups. This is a
   sensitivity pilot, not KernelSHAP. Complete a predeclared small prompt panel
   for every candidate and save results even if quality or detection worsens.
4. If selective attenuation shows useful effects, evaluate sampled mixed
   coalitions and fit separate KernelSHAP-style contribution estimates. Compare
   with an equal-generation-budget random/coordinate search. Account for every
   generated image; shared clean images and cached unchanged endpoints save work.
5. Use Pareto ranking to nominate a small fixed shortlist for full validation
   and matched student training. Record detector score quantiles, lost positives,
   paired LPIPS intervals, CLIP, FID/KID at a suitable sample size, and visual
   comparisons. Freeze the shortlist before independent test evaluation.

Priority: controlled group scaling plus measured attribution, then Pareto-guided
selection. MF-LTGA and SCOPE remain longer-term options. The central hypothesis
is that some adapter groups contribute more to distortion than to transferable
watermark signal; the experiment must establish this, not assume it.

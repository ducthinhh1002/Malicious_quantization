# JPEG failure diagnosis and Structure carrier revision

The DCT consensus run was explicitly stopped by the user. Its last50 training
updates had JPEG bit accuracy49.69%, versus49.81% in the first50. Loss decreased
only3.9%. A prior8-update same-example loss reduction did not establish learning
across prompts. Treating that probe as sufficient for a600-step experiment was
an evaluation-design error.

Two different limitations were conflated. The old semantic carrier selected PCA
coordinates for clean/VAE-cycle stability, whereas JPEG substantially changes
those coordinates. Adding a fixed spatial-frequency carrier also introduces a
new generator-learning problem. No experiment established that a low-rank UNet
could realize all those fixed-frequency signs across prompts within the proposed
quality constraints and budget. Phase-locking/spatial capacity and truncated
sampler gradients remain hypotheses, not proven causes. The measured projection
retained essentially all update norm, which does not support simple projection
suppression as the explanation.

## Literature considered

Stable Signature learns an extractor through transformations before optimizing
model weights against that extractor. This supports checking the representation
before attempting model optimization.
https://arxiv.org/html/2303.15435v2

DistSeal studies latent watermarkers and distillation into model or decoder
parameters. It provides motivation for separating representation suitability
from parameter distillation. No DistSeal component/checkpoint is used here.
https://ai.meta.com/research/publications/learning-to-watermark-in-the-latent-space-of-generative-models/
https://github.com/facebookresearch/distseal

Slow Feature Analysis learns invariant representations using signal variation
and perturbation variation. Our paired-codec covariance construction borrows
this statistical principle with JPEG differences instead of temporal derivatives.
https://www.ini.rub.de/PEOPLE/wiskott/Reprints/WiskottSejnowski-2002-NeurComp-LearningInvariances.pdf

## Implemented experiment

The frozen SD2.1 VAE/UNet semantic extractor, GAP features, original32 message
signs, original refine600 rank32 warm start, Structure objective and quality
projection are reused. The carrier directions change. No learned CNN detector
or inference-time watermark guidance is introduced.

Using1024 clean ORIGINAL teacher-training images and paired JPEG25 versions:
1. Restrict to the original PC8..64 semantic subspace.
2. Estimate centered clean signal covariance S and noncentered JPEG difference
   covariance N, so systematic codec bias contributes to the noise estimate.
3. Solve the generalized eigenproblem for N relative to S, with a1e-4 relative
   covariance floor. Retain32 low-perturbation directions.
4. QR-orthogonalize the resulting raw-feature directions and diagonalize clean
   signal covariance within the selected subspace. This preserves the existing
   orthonormal-carrier implementation and decorrelates clean coordinates.
5. Freeze mean, directions, scales and original message signs. Carrier fitting
   uses no watermark-positive images or key-dependent fitting objective.

Preliminary ranking of individual PCs gave only61.55% sign retention on fitting
pairs versus58.98% for the old carrier. It was superseded before a full training
run by the covariance-based subspace approach. Its cached features are reused.

Independent1000 clean-validation pairs gave:
- old sign retention59.078%, normalized squared coordinate discrepancy2.07844
- new sign retention76.666%, discrepancy0.46174.
These measure clean-coordinate stability, NOT watermark recovery or detection.
They justify a model-training experiment but do not guarantee its success.

Scalar training calibration is recomputed on the1000 clean-validation originals
AFTER carrier fitting. These images are excluded from model optimization.
The existing fixed24/32 acceptance metric is reported as such, without assuming
its nominal binomial FPR equals the empirical null.

Training budget is1200 extra updates, selected because carrier directions have
changed. Native image/JPEG per-bit margin updates occur every2 updates on one
training example, with20 differentiable tail calls and a full50-step PNDM forward.
Native training and production inference useFP32. Protected AdamW preserves
first-order descent constraints for Structure, native margin and quality.

The first16 validation prompts are monitored before training and every100
updates. Monitoring does not select the final checkpoint, which is fixed1200.
Final evaluation uses1000 held-out test prompts and the new carrier. The legacy
output directory name teacher_jpeg600_test is retained by the reused generator,
but checkpoint identity and verified training step are1200.

Known remaining limitations include truncated parameter derivatives, native
training batch1, JPEG-specific fitting, a single key/seed and unverified image
quality/generalization. No claim of unseen-tokenizer or AR-student robustness is
supported by this experiment.

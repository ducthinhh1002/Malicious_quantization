# Image cleanup on vm4

Completed: 2026-10-18T09:31:42.532895+00:00

- Removed 22,618 individual generated image files (9.41 GiB) from 125 pruned image sets.
- Retained 7,582 original images in total, including 1,550 deterministic examples in pruned sets.
- Kept all LALT images, strong BitGreen datasets, selected useful clean/robust controls, centered step500 triplet, native fit/gate images, result figures and all JSON/CSV reports.
- Verified every preserved original image is present at its original size; every planned removal is absent.
- No model weights, tensor caches, environments, source code or active-run assets were deleted in this pass.
- Each partial set contains IMAGE_RETENTION.json. Original full-dataset metrics remain in reports; the retained examples are for qualitative comparison and do not replace the original full metric dataset.

[Plan](image_cleanup_20261018_093011Z.plan.json)

[Per-image deletion log](image_cleanup_20261018_093011Z.plan.deleted.jsonl)

[Summary](image_cleanup_20261018_093011Z.plan.summary.json)

[Verification](verification.json)

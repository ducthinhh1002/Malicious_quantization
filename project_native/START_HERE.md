# Projected native — source snapshot from VM4

Downloaded on 2026-10-26 from `/root/logs/threshold_quality_20261022`.
This folder is a source-and-metadata snapshot of the project described in
`../Report_Projected_native_20261026.pdf`. The source was subsequently reorganized locally: the shared harness is now `src/`.
`RELOCATION_MANIFEST.json` records original and current paths and hashes.

## Read these files first

- `pareto_quality_refine.py`: Projected native teacher refinement, paired native sampler, gradient balancing and score floor.
- `pareto_step.py`: constrained projection of the actual AdamW displacement.
- `pareto_quality_campaign.py`: teacher refinement and ordinary SD1.4 student campaign.
- `student10000_campaign.py`, `student10000_replay_driver.py`: seeded student replay and fixed 10000-step follow-up.
- `transfer_refine.py`, `transfer_proxy.py`, `native_quality_refine.py`: shared training, image and sampler components.
- `src/`: complete shared Python harness, requirements and pinned vendor source resources.
- `pareto_quality_tests.py`, `native_quality_tests.py`: original method tests.

## Provenance and results

`DOWNLOAD_MANIFEST.json` preserves the original paths/SHA256 for all 227 files transferred from VM4; it is historical provenance, not the current layout.
The files named in all three historical campaign code manifests were checked
against the server source and matched at download time. All 110 downloaded Python
files parsed locally; no local model training or GPU runtime verification was run.

Lightweight configs, plans, comparisons, thresholds and available summary reports
are preserved under:

- `pareto_quality_20261025/`: Projected native and balanced-native campaign metadata.
- `native_quality_20261025/`: baseline/continuation/native-full comparison metadata.
- `projected_student10000_20261025/`: 10000-step follow-up metadata.

`SERVER_ENVIRONMENT.json` records package versions in the prepared VM4 environment
at download time. The original requirements files are also retained.

## Execution limits of this download

Large teacher/student checkpoints, PCA tensors, image datasets, optimizer states,
and latent caches are not included. The original campaign scripts also refer to
previous VM4 runs (for example `kappa1_manojb_5k/source` and research queue plans).
Those dependencies and absolute paths in saved configs must be restored or
adapted before launching on a different machine. This is NOT a claim that simply
running a campaign on this downloaded snapshot will reproduce the experiment.

Report summary: Projected native teacher validation detection 99.4%, LPIPS
0.371757; student detection 79.2% at 7500 steps, 82.1% in the 10000-step seeded
follow-up. See the report for controls, reference definitions and limitations.

## Local layout update

Shared code now lives in `src/`; root entrypoints and semantic result/source-manifest paths were updated. `python main.py --plan-only` and pinned vendor integrity verification pass. Four missing original vendor resource files were restored by matching their pinned hashes. Historical run manifests retain their original hashes; changed source files are audited separately in `RELOCATION_MANIFEST.json`. Old VM4 runs cannot be resumed blindly against relocated sources. Model/data dependencies described above still apply.

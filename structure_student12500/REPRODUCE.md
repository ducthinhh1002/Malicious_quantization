# Reproduce Structure12500 (91.1% exploratory seed-0 result)

## Code included

- Root scripts:12500-step student campaign, absolute LR schedule, deterministic
  sampler and full-state checkpoint hooks.
- `historical/`:complete downloaded historical source package, including
  Structure/margin teacher refinement and semantic fingerprint implementation.
- `historical/Malicious_quantization/bitmark_lora/`:student/validation/scoring/
  utility harness plus pinned Diffusers trainers under `vendor/diffusers/`.
- `bundled_source_manifest.json`:SHA256 for all149 imported historical files.
- `reference/`:original configuration,asset revisions,code/training identity,
  test record and compact final result. These are evidence,not launch configs.

The imported code and vendored trainers retain their source bytes. The packaging
changes only root code/data lookup in `bootstrap.py` and the optional source-run
override in `campaign.py`. Loss,rank,batch,LR schedule,sampling and selection are
unchanged. The changed launch files have different hashes from the archived run;
use a NEW output run for reproduction. Do not replace its original frozen
manifests or resume it with this packaged launcher.

## External artifacts intentionally excluded

No checkpoints,model weights,image datasets,virtual environment or cancelled
checkpoint-averaging experiment is committed.

To reproduce the STUDENT run,provide the completed Structure teacher source
run,including its5000training images,1000validation/test images,clean references,
prompts,carrier/key,thresholds,validation cache and metadata. The downloaded
`output_artifacts/Structure7500_20261026/run/` is that source. The original
12500full-state checkpoint is in the separate Structure12500 download.

Provide these base model revisions:

- Teacher:`Manojb/stable-diffusion-2-1-base`,revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`.
- Student:`CompVis/stable-diffusion-v1-4`,revision
  `133a221b8aa7292a167afc5127cb63fb5005638b`.

The full historical teacher code is included,but regenerating the TEACHER from
scratch also requires its original k=1initial checkpoint,calibration artifacts
and historical campaign inputs. Bundling code does not regenerate those assets.

## Environment and checks

The recorded run used Python3.12,PyTorch2.7.1+cu126,Diffusers0.35.1,
Transformers4.51.3,PEFT0.17.1and Accelerate1.10.1. Other pinned dependencies
are listed in `historical/Malicious_quantization/bitmark_lora/requirements-semantic.txt`.
That historical file specifies cu128;choose matching CUDA wheels explicitly
when recreating the recorded cu126 environment. It is not an exact environment
lock for the completed VM4 run.

```bash
cd structure_student12500
python verify_bundle.py
python -m unittest tests
```

The resume tests use small stochastic models. CUDA FP16 tests require a CUDA
device;CPU tests are also provided. Passing them does not establish an identical
full-model metric on another GPU/runtime.

## Configure data,then launch a NEW student run

On Linux,set the source to a WORKING COPY of the downloaded source run:

```bash
export STRUCTURE12500_SOURCE_RUN=/path/to/working-copy-of-Structure7500/run
python campaign.py --initialize
```

Before initialization,update model locations in that working copy's
`assets.json` and `config.json` to real paths on the new machine. Preserve
the archived originals. Model content must match `reference/original_assets.json`.
The working source's prompts,key,carrier,images and thresholds must remain the
same. `STRUCTURE12500_DATA_ROOT` can instead point to a historical artifact tree
with `structure_margin_20261026/structure_margin_native` underneath it.

Run validation and issue a receipt tied to the newly frozen code manifest:

```bash
python validate.py
python campaign.py --launch
```

The receipt prevents treating old tests as covering changed launch code. Check
the test log for CUDA availability/skips. The actual four-update SD1.4 smoke is
then a mandatory stage before12500-step training. The supervisor persists after
the launching terminal exits.

Do not run the archived cancelled averaging campaign. No full12500-step
retraining was launched or claimed during this packaging task.

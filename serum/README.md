# SERUM author-artifact pipeline (server snapshot)

Source copied from `/root/quant` on gpu-node-2, 11 October 2026. This is the
current raw/strong experiment, not the retired locally trained watermark branch.
No old broken LoRA trainer, obsolete training queues, credentials, images or
model weights are included. Vendored projects retain their upstream licenses.

## Entry points and scientific settings

- `scripts/download_author_serum.sh` downloads the public author checkpoint and
  SD2.1 cache; `prepare_author_serum.py` validates/extracts the artifacts.
- `scripts/author_serum_stage0.py` generates clean/watermarked teacher images
  (5,000 train + 1,000 validation + 5,000 calibration per branch), audits pairing,
  calibrates 1% FPR on held-out clean images and freezes the author threshold.
  Generation uses alpha 0.5, 512 pixels, 50 steps, CFG 7.5 and per-image seeds.
- `scripts/queue_author_serum.py`: teacher readiness -> C00/C10 rank-4 SD1.4
  LoRA, LR 1e-5, batch 8, 20,000 steps -> unseen validation MSE -> 5,000
  evaluation outputs -> fixed-threshold author audit.
- `scripts/queue_author_strong.py`: C01/C11 strength 1.00 datasets -> identical
  adaptation/selection/generation/audit -> CLIPScore/FID/KID -> four-condition
  summary. No lower-strength student jobs are queued.
- `scripts/c11_parallel_validation.py`: GPU0 precomputes C11 validation while
  GPU1 processes C11 training images, then GPU0 computes C10 utility. Use only
  during `C11_regen_train` after the raw queue completes. The regular queue
  validates and reuses completed validation images.
- `scripts/pilot_metrics.py` and `select_pilot_strengths.py` preserve the original
  preservation-based selection implementation, not an active student queue.

The frozen detector threshold must never be tuned on student evaluation data.
Updated C10 detection is 4.42% (221/5,000); the paper's 52.30% radioactivity
baseline remains unreproduced. Completion of execution does not imply baseline
reproduction or attack success.

## Corrected training source

`third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py`
materializes `lora_layers` as a list before optimizer construction and gradient
clipping. Its SHA-256 is:

`18c712a5b4aa1f47f3221ee0ea5a98758f02f35ac76f753bd5ad2bdf07e98534`

`manifests/lora_gradient_clipping_fix.json` records the fix. Other Diffusers
training examples are excluded. `SERVER_SOURCE_SHA256.json` records the copied
server source; verify with `python verify_snapshot.py` from this directory.
Vendored Diffusers includes its runtime package; SERUM/CtrlRegen include their
tracked source/configuration files. The `.git` directories and checkpoints are
excluded. Pinned upstream commits and environment freezes are in `manifests/`.

## Deployment and required external state

The unchanged server scripts intentionally use absolute root `/root/quant`.
Deploy this directory's **contents** there on a separate Linux GPU machine;
do not overwrite the live server while its queue is running. Python 3.12,
CUDA-compatible GPUs and tmux are required. Restore environment packages from
the three recorded `*_env.txt` files; replace their editable Diffusers line with
`pip install -e /root/quant/third_party/diffusers`. These are captured environment
inventories, not portable dependency guarantees.

The running queue reuses existing clean images and a corrected C00 checkpoint.
To resume that exact queue, additionally restore the external state referenced
by `reuse_author_clean.py`: `data/raw/{clean_train,clean_val,calib_clean}`,
`data/laundered/clean_step_1.00_{train,val}`, `data/hf/C00_train/metadata.jsonl`,
`students/C00_seed0_clipfix`, `metrics/serum_author/backbone_comparison.json`
and the utility reference `data/utility_teacher_clean` (5,000 clean SD2.1 images
on `prompts/student_eval.txt`). These images/checkpoints are intentionally not
in Git. The supplied inventory is provenance, not a replacement for the data.
The manifest under `assets/serum_author/` records expected external weights;
re-run artifact preparation after download rather than assuming the files exist.

Teacher generation itself can run from scratch without legacy reuse:

```bash
cd /root/quant
mkdir -p logs metrics/serum_author manifests
bash scripts/download_author_serum.sh
.venv/bin/python scripts/prepare_author_serum.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/author_serum_stage0.py --worker 0 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/author_serum_stage0.py --worker 1 &
wait
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/author_serum_stage0.py --audit-only
```

This produces fresh clean and watermarked images if no `clean_reuse.json` exists.
The exact running queue entry points below additionally require the reused state
listed above; they are not a one-command fresh reproduction launcher:

```bash
tmux new-session -d -s quant-author-queue '.venv/bin/python scripts/queue_author_serum.py > logs/serum_author_queue_active.log 2>&1'
tmux new-session -d -s quant-author-strong '.venv/bin/python scripts/queue_author_strong.py > logs/serum_author_strong_queue.log 2>&1'
```

Status files are `metrics/serum_author/queue_status.json`,
`strong_queue_status.json`, and `C11_parallel_validation_status.json`.
No SSH key or Hugging Face token should be committed; authenticate externally.

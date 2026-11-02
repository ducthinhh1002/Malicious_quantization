#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
KIND=$1
COND=$2
STEP=$3
export CUDA_VISIBLE_DEVICES=$4
PY=/root/workspace/serum/.venv-student/bin/python
for split in train val; do
  seed=10000; test "$split" != val || seed=20000
  .venv-ctrlregen/bin/python scripts/launder_ctrlregen.py --ctrlregen-repo third_party/CtrlRegen --input-dir "data/raw/${KIND}_${split}" --output-dir "data/laundered/${KIND}_step_${STEP}_${split}" --step "$STEP" --base-seed "$seed" --resume --spatial-control third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000 --semantic-control third_party/CtrlRegen/public_checkpoints/semanticnet_ckp
done
"$PY" scripts/audit_laundered_pairing.py --kind "$KIND" --step "$STEP"
DATA="data/hf/${COND}_step${STEP}_train"
DEST="students/${COND}_step${STEP}_seed0"
"$PY" scripts/make_hf_imagefolder.py --images "data/laundered/${KIND}_step_${STEP}_train" --prompts prompts/adapt_train.txt --out "$DATA"
if ! test -f "$DEST/validation_selection.json"; then
  if ! test -f "$DEST/pytorch_lora_weights.safetensors"; then
  RESUME=$("$PY" scripts/find_resume_checkpoint.py "$DEST")
  resume_args=(); test -z "$RESUME" || resume_args=(--resume_from_checkpoint="$RESUME")
  "$PY" -m accelerate.commands.launch --num_processes 1 third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --train_data_dir="$DATA" --image_column=image --caption_column=text --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=1 --learning_rate=1e-5 --rank=4 --max_train_steps=20000 --checkpointing_steps=500 --mixed_precision=fp16 --seed=0 "${resume_args[@]}" --output_dir="$DEST"
  fi
  "$PY" scripts/select_lora_checkpoint.py --student "$DEST" --images "data/laundered/${KIND}_step_${STEP}_val"
fi
BEST=$(cat "$DEST/BEST_LORA_PATH.txt")
"$PY" scripts/generate_student.py --lora "$BEST" --prompts prompts/student_eval.txt --out "student_outputs/${COND}_step${STEP}_seed0" --resume
.venv/bin/python scripts/serum_scores.py --images "student_outputs/${COND}_step${STEP}_seed0" --out "metrics/${COND}_step${STEP}_seed0_scores.npy"

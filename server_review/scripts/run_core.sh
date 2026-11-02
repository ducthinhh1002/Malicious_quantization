#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
: "${GPU:=0}"; : "${SEED:=0}"
: "${STEP:?Set STEP to a pilot-selected mild/strong strength}"
SERUM_PY=/root/workspace/serum/.venv/bin/python
CTRL_PY=/root/workspace/serum/.venv-ctrlregen/bin/python
STUDENT_PY=/root/workspace/serum/.venv-student/bin/python
test -x "$CTRL_PY"; test -x "$STUDENT_PY"
"$SERUM_PY" scripts/check_stage_gates.py --strength "$STEP"
S="$STEP"
for split in train val; do
  seed=10000; if [ "$split" = val ]; then seed=20000; fi
  for kind in clean wm; do
    CUDA_VISIBLE_DEVICES="$GPU" "$CTRL_PY" scripts/launder_ctrlregen.py --ctrlregen-repo third_party/CtrlRegen --input-dir "data/raw/${kind}_${split}" --output-dir "data/laundered/${kind}_step_${S}_${split}" --step "$STEP" --base-seed "$seed" --spatial-control third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000 --semantic-control third_party/CtrlRegen/public_checkpoints/semanticnet_ckp
  done
done
for cond in C00 C01 C10 C11; do
  case "$cond" in
    C00) train=data/raw/clean_train; val=data/raw/clean_val;;
    C01) train="data/laundered/clean_step_${S}_train"; val="data/laundered/clean_step_${S}_val";;
    C10) train=data/raw/wm_train; val=data/raw/wm_val;;
    C11) train="data/laundered/wm_step_${S}_train"; val="data/laundered/wm_step_${S}_val";;
  esac
  dest="students/${cond}_step${S}_seed${SEED}"
  "$STUDENT_PY" scripts/make_hf_imagefolder.py --images "$train" --prompts prompts/adapt_train.txt --out "data/hf/${cond}_step${S}_train"
  CUDA_VISIBLE_DEVICES="$GPU" "$STUDENT_PY" -m accelerate.commands.launch --num_processes 1 third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --train_data_dir="data/hf/${cond}_step${S}_train" --image_column=image --caption_column=text --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=1 --learning_rate=1e-5 --rank=4 --max_train_steps=20000 --checkpointing_steps=500 --mixed_precision=fp16 --seed="$SEED" --output_dir="$dest"
  CUDA_VISIBLE_DEVICES="$GPU" "$STUDENT_PY" scripts/select_lora_checkpoint.py --student "$dest" --images "$val"
  best=$(cat "$dest/BEST_LORA_PATH.txt")
  outputs="student_outputs/${cond}_step${S}_seed${SEED}"
  CUDA_VISIBLE_DEVICES="$GPU" "$STUDENT_PY" scripts/generate_student.py --lora "$best" --prompts prompts/student_eval.txt --out "$outputs"
  CUDA_VISIBLE_DEVICES="$GPU" "$SERUM_PY" scripts/serum_scores.py --images "$outputs" --out "metrics/${cond}_step${S}_seed${SEED}_scores.npy"
done


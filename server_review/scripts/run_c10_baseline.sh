#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
: "${GPU:=0}"; : "${SEED:=0}"
PY=/root/workspace/serum/.venv-student/bin/python
test -x "$PY"
.venv/bin/python scripts/check_data_ready.py
"$PY" scripts/make_hf_imagefolder.py --images data/raw/wm_train --prompts prompts/adapt_train.txt --out data/hf/C10_train
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m accelerate.commands.launch --num_processes 1 third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --train_data_dir=data/hf/C10_train --image_column=image --caption_column=text --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=1 --learning_rate=1e-5 --rank=4 --max_train_steps=20000 --checkpointing_steps=500 --mixed_precision=fp16 --seed="$SEED" --output_dir="students/C10_seed${SEED}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/select_lora_checkpoint.py --student "students/C10_seed${SEED}" --images data/raw/wm_val
BEST=$(cat "students/C10_seed${SEED}/BEST_LORA_PATH.txt")
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/generate_student.py --lora "$BEST" --prompts prompts/student_eval.txt --out "student_outputs/C10_seed${SEED}"
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/serum_scores.py --images "student_outputs/C10_seed${SEED}" --out "metrics/C10_seed${SEED}_scores.npy"
echo 'C10 scored. Review baseline significance and utility before recording a pass gate.'



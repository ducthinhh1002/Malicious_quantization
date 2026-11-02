#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/workspace/serum
SERUM=$ROOT/third_party/SERUM
cd "$ROOT"
exec > >(tee -a logs/fast_2gpu.log) 2>&1
while tmux has-session -t quant-assets 2>/dev/null; do sleep 15; done
[[ -f manifests/sd21_mirror.json ]] || { echo 'SD2.1 mirror not ready'; exit 2; }
echo "[$(date -Is)] stopping vllm service for GPU0"
docker stop -t 30 vllm-gemma4-26b-a4b
source .venv/bin/activate
cd "$SERUM"
BASE=configs/config_sd21.yaml
for shard in 0 1; do
  cfg="$ROOT/manifests/config_sd21_shard${shard}.yaml"
  cp "$BASE" "$cfg"
  sed -i "s/generated_data_file: \"data_sd21_gen.pt\"/generated_data_file: \"data_sd21_gen_shard${shard}.pt\"/" "$cfg"
  sed -i "s/augmented_data_file: \"data_sd21_aug.pt\"/augmented_data_file: \"data_sd21_aug_shard${shard}.pt\"/" "$cfg"
done
CUDA_VISIBLE_DEVICES=0 python -m src.data.dataset --config "$ROOT/manifests/config_sd21_shard0.yaml" --mode both --batch-size 32 --start-idx 0 --end-idx 20000 > "$ROOT/logs/serum_data_gpu0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -m src.data.dataset --config "$ROOT/manifests/config_sd21_shard1.yaml" --mode both --batch-size 32 --start-idx 20000 --end-idx 40000 > "$ROOT/logs/serum_data_gpu1.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
python - <<'PY'
from pathlib import Path
import torch
p=Path('datasets')
for kind in ('gen','aug'):
 out=[]
 for s in (0,1):
  files=sorted(p.glob(f'data_sd21_{kind}_shard{s}*.pt'))
  if not files: raise FileNotFoundError((kind,s))
  out.extend(torch.load(files[-1],map_location='cpu',weights_only=False))
 torch.save(out,p/f'data_sd21_{kind}.pt')
 print(kind,len(out))
PY
CUDA_VISIBLE_DEVICES=0 python -m src.training.train --config "$BASE" --dont-load-checkpoint
split_prompts() { local src=$1 prefix=$2 n=$3; head -n "$n" "$src" > "${prefix}0.txt"; tail -n +$((n+1)) "$src" > "${prefix}1.txt"; }
mkdir -p "$ROOT/prompts/shards" "$ROOT/data/shards"
split_prompts "$ROOT/prompts/adapt_train.txt" "$ROOT/prompts/shards/adapt_train_" 2500
split_prompts "$ROOT/prompts/adapt_val.txt" "$ROOT/prompts/shards/adapt_val_" 500
split_prompts "$ROOT/prompts/detector_calib.txt" "$ROOT/prompts/shards/detector_calib_" 2500
gen_pair() { local name=$1 total=$2 half=$3 clean=$4 wm=$5;
 CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.generate --config "$BASE" --generate --prompts-path "$ROOT/prompts/shards/${name}_0.txt" --clean-path "$ROOT/data/shards/${name}_clean_0" --watermarked-path "$ROOT/data/shards/${name}_wm_0" --num-samples "$half" --batch-size 32 > "$ROOT/logs/${name}_gpu0.log" 2>&1 & local a=$!
 CUDA_VISIBLE_DEVICES=1 python -m src.evaluation.generate --config "$BASE" --generate --prompts-path "$ROOT/prompts/shards/${name}_1.txt" --clean-path "$ROOT/data/shards/${name}_clean_1" --watermarked-path "$ROOT/data/shards/${name}_wm_1" --num-samples $((total-half)) --batch-size 32 > "$ROOT/logs/${name}_gpu1.log" 2>&1 & local b=$!
 wait "$a"; wait "$b"
 python "$ROOT/scripts/merge_image_shards.py" --inputs "$ROOT/data/shards/${name}_clean_0" "$ROOT/data/shards/${name}_clean_1" --out "$clean" --prefix clean
 python "$ROOT/scripts/merge_image_shards.py" --inputs "$ROOT/data/shards/${name}_wm_0" "$ROOT/data/shards/${name}_wm_1" --out "$wm" --prefix watermarked
}
gen_pair adapt_train 5000 2500 "$ROOT/data/raw/clean_train" "$ROOT/data/raw/wm_train"
gen_pair adapt_val 1000 500 "$ROOT/data/raw/clean_val" "$ROOT/data/raw/wm_val"
gen_pair detector_calib 5000 2500 "$ROOT/data/raw/calib_clean" "$ROOT/data/raw/calib_wm_unused"
echo "[$(date -Is)] fast two-GPU SERUM stage complete"
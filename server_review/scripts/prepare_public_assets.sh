#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/workspace/serum
cd "$ROOT"
exec > >(tee -a logs/assets.log) 2>&1
while tmux has-session -t quant-setup 2>/dev/null; do sleep 10; done
source .venv/bin/activate
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
root=Path('/root/workspace/serum/prompts'); root.mkdir(parents=True, exist_ok=True)
ds=load_dataset('Gustavosta/Stable-Diffusion-Prompts', split='train').shuffle(seed=42)
col='Prompt' if 'Prompt' in ds.column_names else ('prompt' if 'prompt' in ds.column_names else ds.column_names[0])
vals=[]; seen=set()
for value in ds[col]:
    value=' '.join(str(value).split())
    if value and value not in seen:
        seen.add(value); vals.append(value)
    if len(vals)>=16000: break
assert len(vals)==16000, len(vals)
splits={'adapt_train.txt':(0,5000),'adapt_val.txt':(5000,6000),'student_eval.txt':(6000,11000),'detector_calib.txt':(11000,16000)}
for name,(a,b) in splits.items():
    (root/name).write_text('\n'.join(vals[a:b])+'\n',encoding='utf-8')
print({k:b-a for k,(a,b) in splits.items()})
PY
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path('/root/workspace/serum')
snapshot_download('yepengliu/ctrlregen', local_dir=root/'third_party/CtrlRegen/public_checkpoints')
for model in ('stabilityai/stable-diffusion-2-1-base','CompVis/stable-diffusion-v1-4','SG161222/Realistic_Vision_V4.0_noVAE','facebook/dinov2-giant','stabilityai/sd-vae-ft-mse'):
    try:
        snapshot_download(model, local_dir_use_symlinks=False)
        print('cached',model)
    except Exception as e:
        print('MODEL_DOWNLOAD_FAILED',model,repr(e),flush=True)
PY
python - <<'PY'
from pathlib import Path
import json,hashlib
root=Path('/root/workspace/serum')
files={p.name:sum(1 for _ in p.open(encoding='utf-8')) for p in (root/'prompts').glob('*.txt')}
sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'prompts').glob('*.txt')}
(root/'manifests/public_assets.json').write_text(json.dumps({'prompt_source':'Gustavosta/Stable-Diffusion-Prompts','shuffle_seed':42,'files':files,'sha256':sha,'ctrlregen_source':'yepengliu/ctrlregen'},indent=2)+'\n')
PY
echo "[$(date -Is)] public assets complete"
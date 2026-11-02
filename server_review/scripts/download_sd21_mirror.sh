#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/workspace/serum
cd "$ROOT"
exec > >(tee -a logs/sd21_download.log) 2>&1
if [ ! -x .venv-hf/bin/python ]; then
 python3 -m venv .venv-hf
 .venv-hf/bin/python -m pip install -q --upgrade pip
 .venv-hf/bin/python -m pip install -q 'huggingface_hub[hf_xet]'
fi
.venv-hf/bin/python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import hashlib,json
repo='sd-research/stable-diffusion-2-1-base'
info=HfApi().model_info(repo, files_metadata=True)
revision=info.sha
out=Path('/root/workspace/serum/models/sd21-base')
out.mkdir(parents=True,exist_ok=True)
snapshot_download(repo_id=repo, revision=revision, local_dir=out,
 ignore_patterns=['*.ckpt','v2-1_512-*.safetensors','*.bin'])
required=['model_index.json','scheduler/scheduler_config.json','tokenizer/tokenizer_config.json','text_encoder/config.json','unet/config.json','vae/config.json']
missing=[x for x in required if not (out/x).exists()]
if missing: raise RuntimeError(f'missing components: {missing}')
weights=sorted(p for p in out.rglob('*.safetensors') if p.is_file())
if not weights: raise RuntimeError('no safetensors downloaded')
manifest={'repo':repo,'revision':revision,'weights':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}}
Path('/root/workspace/serum/manifests/sd21_mirror.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'repo':repo,'revision':revision,'weight_files':len(weights)},indent=2))
PY
echo "[$(date -Is)] SD2.1 mirror ready"
#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
.venv-student/bin/python - <<'PY'
import lpips
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
AutoModel.from_pretrained('facebook/dinov2-base')
AutoImageProcessor.from_pretrained('facebook/dinov2-base')
lpips.LPIPS(net='alex')
PY
mkdir -p data/pilot_preflight/input
.venv/bin/python - <<'PY'
from pathlib import Path
import os
src=sorted(Path('data/raw/clean_train').glob('*.png'))[0]
dst=Path('data/pilot_preflight/input')/src.name
if not dst.exists(): os.link(src,dst)
PY
CUDA_VISIBLE_DEVICES=0 .venv-ctrlregen/bin/python scripts/launder_ctrlregen.py --ctrlregen-repo third_party/CtrlRegen --input-dir data/pilot_preflight/input --output-dir data/pilot_preflight/output --step 0.25 --base-seed 10000 --spatial-control third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000 --semantic-control third_party/CtrlRegen/public_checkpoints/semanticnet_ckp
.venv/bin/python - <<'PY'
from PIL import Image
from pathlib import Path
import json
files=list(Path('data/pilot_preflight/output').glob('*.png'))
assert len(files)==1
with Image.open(files[0]) as im:
    im.load()
    assert im.size==(512,512)
Path('metrics/stage12_preflight.json').write_text(json.dumps({'ok':True,'image':str(files[0])},indent=2))
PY

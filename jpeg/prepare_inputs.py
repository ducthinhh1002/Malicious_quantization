"""Portable clean-input generation for the fixed Structure warm start."""
import os
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('JPEG_SOURCE_RUN',str(ROOT/'source'))).expanduser().resolve()
WARM=Path(os.environ.get('JPEG_WARM_ADAPTER',str(ROOT/'inputs/refine600'))).expanduser().resolve()
EXPECTED='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'
assert hashlib.sha256((WARM/'adapter_model.safetensors').read_bytes()).hexdigest()==EXPECTED,'Incorrect original refine600 adapter'
if os.environ.get('JPEG_SOURCE_RUN'):
    for name in ['config.json','assets.json','prompts.json','carrier/complete.json','images/teacher_clean_train/image_hashes.json','images/teacher_clean_validation/image_hashes.json']:
        if not (SOURCE/name).exists():raise FileNotFoundError(SOURCE/name)
    print('Reusing complete supplied original source run',SOURCE)
    sys.exit(0)
# Resolve pinned base model before importing the offline experiment bootstrap.
from huggingface_hub import snapshot_download
assets=json.loads((ROOT/'inputs/original/assets.json').read_text())
model=snapshot_download(assets['teacher_source']['repository'],revision=assets['teacher_source']['revision'],cache_dir=str(ROOT/'cache'))
assets['teacher_clean']=assets['teacher']=model
assets['student']=str(ROOT/'cache/unused_sd14')
import bootstrap
from bootstrap import freeze,write,read,sha
import shutil,torch
from semantic_teacher import load_native,generate_set
SOURCE.mkdir(exist_ok=True,parents=True)
for name in ['config.json','prompts.json']:
    freeze(SOURCE/name,read(ROOT/'inputs/original'/name))
freeze(SOURCE/'assets.json',assets)
if not (SOURCE/'carrier').exists():shutil.copytree(ROOT/'inputs/original/carrier',SOURCE/'carrier')
pipe=load_native(model)
for split in ['train','validation']:
    generate_set(SOURCE,model,split,'teacher_clean_'+split,pipe)
write(SOURCE/'portable_input_provenance.json',dict(original_adapter_sha256=EXPECTED,base_repository=assets['teacher_source']['repository'],base_revision=assets['teacher_source']['revision'],clean_images='regenerated from original prompt/seed/config; GPU numerical differences possible'))

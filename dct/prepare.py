"""Resolve pinned public models and freeze the portable DCT experiment inputs."""
from pathlib import Path
import json,os,hashlib
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
pins={
 'teacher':dict(repository='Manojb/stable-diffusion-2-1-base',revision='0094d483a120f3f33dafbd187ea4aa60d10de75c'),
 'student':dict(repository='CompVis/stable-diffusion-v1-4',revision='133a221b8aa7292a167afc5127cb63fb5005638b')}
assets={}
for name,pin in pins.items():
 supplied=os.environ.get('DCT_'+name.upper()+'_MODEL')
 path=Path(supplied).expanduser().resolve() if supplied else Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(ROOT/'.cache/hf')))
 if not (path/'model_index.json').is_file():raise FileNotFoundError(path/'model_index.json')
 assets[name+'_path']=str(path)
assets_path=ROOT/'assets.json'
if assets_path.exists() and json.loads(assets_path.read_text())!=assets:raise RuntimeError('Model paths changed; use a fresh run directory')
assets_path.write_text(json.dumps(assets,indent=2))
prompt=ROOT/'benchmark/shared/data/prompts.json'
assert hashlib.sha256(prompt.read_bytes()).hexdigest()=='5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2'
protocol=dict(models=pins,teacher_dtype='float32',teacher_sampler='DDIM50 CFG7.5',message='NW32',key='latent32-dct-pilot-20261111',carrier='orthonormal DCT-II latent band[12,32),4 channels,50 copies/bit',guidance_scale=200,amplitude=.5,guidance_start_step=25,student='clean SD1.4 LoRA r4 attention_conv',student_steps=12500,prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),workers=os.environ.get('DCT_WORKERS',os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]+':2'))
p=ROOT/'protocol.json'
if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Experiment protocol changed')
p.write_text(json.dumps(protocol,indent=2))

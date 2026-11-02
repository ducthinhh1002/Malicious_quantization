"""Corrected seed-0 2x2 core queue: raw plus strong only; no mild jobs."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

R=Path('/root/workspace/serum')
PY=str(R/'.venv-student/bin/python')
SERUM=str(R/'.venv/bin/python')
TRAIN='third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
TASKS={1:['C00','C01_step1.00'],0:['C11_step1.00']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def status(path,**kw):write(path,{'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})
def guard():
    protocol=json.loads((R/'manifests/clipfix_core_protocol.json').read_text())
    assert protocol['status']=='passed'
    assert json.loads((R/'metrics/clipfix_training_preflight.json').read_text())['status']=='passed'
    assert json.loads((R/'manifests/clipfix_visual_audit/review.json').read_text())['reviewed_source_prompt_rows']==50
    for path,digest in protocol['frozen_files'].items():assert sha(R/path)==digest,f'Frozen artifact changed: {path}'
    from huggingface_hub import try_to_load_from_cache
    cache=Path(try_to_load_from_cache('CompVis/stable-diffusion-v1-4','model_index.json'))
    assert cache.parent.name==protocol['base_model_revision'] and sha(cache)==protocol['base_model_index_sha256']
    return protocol
def train_command(base,dest):
    return [PY,'-m','accelerate.commands.launch','--num_processes','1',TRAIN,'--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4',f'--train_data_dir=data/hf/{base}_train','--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8','--gradient_accumulation_steps=1','--learning_rate=1e-5','--rank=4','--max_train_steps=20000','--checkpointing_steps=500','--mixed_precision=fp16','--seed=0',f'--output_dir={dest}']
def run_stage(name,stage,cmd,gpu):
    guard()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    st=R/f'metrics/{name}_status.json'
    with (R/f'logs/{name}_{stage}.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            status(st,run=name,stage=stage,state='running',gpu=gpu,pid=p.pid)
            time.sleep(15)
        if p.returncode:
            status(st,run=name,stage=stage,state='failed',gpu=gpu,exit_code=p.returncode)
            raise RuntimeError(f'{name} {stage} failed: {p.returncode}; inspect its stage log')
    status(st,run=name,stage=stage,state='completed',gpu=gpu)
def condition(base,gpu):
    import numpy as np
    from safetensors.torch import load_file
    import torch
    from PIL import Image
    name=base+'_seed0_clipfix';dest=R/'students'/name
    protocol=guard();data=protocol['conditions'][base]
    existing=R/f'metrics/{name}_status.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old.get('stage')=='all' and old.get('state')=='completed':return
    assert not dest.exists(),f'{name}: previous partial run exists; explicit recovery required rather than silent restart'
    inventory=json.loads((R/'manifests/clipfix_core_inventory.json').read_text())
    for split in ['train','val']:
        for i,entry in enumerate(inventory[f'{base}_{split}']):
            path=R/entry['file']
            if split=='train':path=R/f'data/hf/{base}_train/{i:06d}.png'
            assert sha(path)==entry['sha256'],f'Image changed: {path}'
    assert sha(R/f'data/hf/{base}_train/metadata.jsonl')==data['metadata_sha256']
    command=train_command(base,dest)
    manifest={'name':name,'base_condition':base,'gpu':gpu,'scope':'corrected 2x2; strong only; user authorized; baseline gate remains pending review','training':protocol['training'],'trainer_sha256':protocol['frozen_files'][TRAIN],'source_protocol_sha256':sha(R/'manifests/clipfix_core_protocol.json'),'base_revision':protocol['base_model_revision'],'train_command':command,'validation_images':data['val']['directory'],'threshold':protocol['threshold'],'fresh_initialization':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    manifest['queue_script_sha256']=sha(__file__)
    write(R/f'manifests/{name}.json',manifest)
    run_stage(name,'train',command,gpu)
    expected={f'checkpoint-{s}' for s in range(500,20001,500)}
    assert {p.name for p in dest.glob('checkpoint-*')}==expected,'Missing checkpoint candidates'
    for p in [*dest.glob('checkpoint-*'),dest]:
        weights=load_file(str(p/'pytorch_lora_weights.safetensors'))
        assert weights and all(bool(torch.isfinite(t).all()) for t in weights.values()),str(p)
    run_stage(name,'validation',[PY,'scripts/select_lora_checkpoint.py','--student',str(dest),'--images',data['val']['directory']],gpu)
    selection=json.loads((dest/'validation_selection.json').read_text())
    assert selection['num_images']==1000 and Path(selection['images']).resolve()==Path(data['val']['directory'])
    assert len(selection['results'])==41 and all(np.isfinite(x['mse']) for x in selection['results'])
    assert selection['best']==min(selection['results'],key=lambda x:x['mse'])
    best=Path((dest/'BEST_LORA_PATH.txt').read_text().strip())
    assert best.parent==dest and str(best)==selection['best']['checkpoint']
    manifest['selected_checkpoint']=str(best);manifest['adapter_sha256']=sha(best/'pytorch_lora_weights.safetensors')
    write(R/f'manifests/{name}.json',manifest)
    run_stage(name,'generation',[PY,'scripts/generate_student.py','--lora',str(best),'--prompts','prompts/student_eval.txt','--out',f'student_outputs/{name}'],gpu)
    files=sorted((R/'student_outputs'/name).glob('*.png'))
    assert [p.name for p in files]==[f'sample_{i:06d}.png' for i in range(5000)]
    for p in files:
        with Image.open(p) as im:im.load();assert im.mode=='RGB' and im.size==(512,512)
    run_stage(name,'serum',[SERUM,'scripts/serum_scores.py','--images',f'student_outputs/{name}','--out',f'metrics/{name}_scores.npy'],gpu)
    scores=np.load(R/f'metrics/{name}_scores.npy').reshape(-1)
    assert scores.shape==(5000,) and np.isfinite(scores).all()
    run_stage(name,'utility',[PY,'scripts/student_utility.py','--student',name],gpu)
    utility=json.loads((R/f'metrics/{name}_utility.json').read_text());assert utility['num_images']==5000
    assert all(np.isfinite(x) for x in utility['distribution_metrics'].values())
    status(existing,run=name,stage='all',state='completed',gpu=gpu)
def worker(gpu):
    lock=(R/f'manifests/clipfix_queue_gpu{gpu}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if gpu==0:
        st=R/'metrics/clipfix_strong_worker0.json'
        while True:
            dep=json.loads((R/'metrics/C10_seed0_clipfix_status.json').read_text())
            if dep['state']=='failed':raise RuntimeError('C10 failed; GPU0 branch withheld until review')
            if dep['state']=='completed' and dep['stage']=='all':break
            status(st,state='waiting_for_C10',dependency=dep)
            time.sleep(20)
    for base in TASKS[gpu]:condition(base,gpu)
    status(R/f'metrics/clipfix_strong_worker{gpu}.json',state='completed')
def main():
    os.chdir(R)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1]);parser.add_argument('--dry-run',action='store_true');a=parser.parse_args()
    guard()
    if a.dry_run:
        print(json.dumps({'tasks':TASKS,'commands':{base:train_command(base,R/'students'/(base+'_seed0_clipfix')) for bases in TASKS.values() for base in bases},'mild_jobs':0},indent=2));return
    if a.worker is not None:
        try:worker(a.worker)
        except BaseException as e:
            status(R/f'metrics/clipfix_strong_worker{a.worker}.json',state='failed',error=str(e));raise
        return
    lock=(R/'manifests/clipfix_strong_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=[];logfiles=[]
    try:
        for gpu in [1,0]:
            f=(R/f'logs/clipfix_strong_worker{gpu}.log').open('a');logfiles.append(f)
            processes.append(subprocess.Popen([PY,__file__,'--worker',str(gpu)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Worker failed; see worker status and logs')
            status(R/'metrics/clipfix_strong_queue_status.json',state='running',tasks=TASKS,mild_jobs=0,worker_pids=[p.pid for p in processes if p.poll() is None])
            time.sleep(20)
        assert all(p.returncode==0 for p in processes)
        run_stage('clipfix_strong_summary','summary',[PY,'scripts/summarize_clipfix_strong.py'],0)
        status(R/'metrics/clipfix_strong_queue_status.json',state='completed',summary='metrics/clipfix_strong_summary.json',mild_jobs=0)
    except BaseException as e:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        status(R/'metrics/clipfix_strong_queue_status.json',state='failed',error=str(e));raise
    finally:
        for f in logfiles:f.close()
if __name__=='__main__':main()

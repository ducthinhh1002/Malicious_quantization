"""Fresh C10 seed-0 rerun with corrected gradient clipping and frozen evaluation."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT=Path('/root/workspace/serum')
os.chdir(ROOT)
NAME='C10_seed0_clipfix'
DEST=ROOT/'students'/NAME
STATUS=ROOT/'metrics'/f'{NAME}_status.json'
PY=str(ROOT/'.venv-student/bin/python')
SERUM=str(ROOT/'.venv/bin/python')
TRAIN=ROOT/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
fix=json.loads((ROOT/'manifests/lora_gradient_clipping_fix.json').read_text())
assert sha(TRAIN)==fix['after_sha256'], 'Trainer differs from tested correction'
assert not DEST.exists(), 'Fresh comparison requires unused output directory'
assert len(list((ROOT/'data/hf/C10_train').glob('*.png')))==5000
manifest={'name':NAME,'created_at':datetime.now().isoformat(),'comparison':'C10_seed0','initialization':'fresh SD1.4; no resume from original adapter','only_intended_training_change':'materialized LoRA parameter list restores gradient clipping','trainer_sha256':sha(TRAIN),'data_metadata_sha256':sha(ROOT/'data/hf/C10_train/metadata.jsonl'),'prompt_hashes':{p:sha(ROOT/'prompts'/p) for p in ['adapt_train.txt','adapt_val.txt','student_eval.txt','detector_calib.txt']},'threshold_sha256':sha(ROOT/'metrics/serum_threshold_1pct.json'),'seed':0,'max_train_steps':20000,'rank':4,'learning_rate':1e-5,'batch_size':8,'gpu':0}
write_json(ROOT/'manifests'/f'{NAME}.json',manifest)
def status(stage,state,**kw):
    write_json(STATUS,{'run':NAME,'stage':stage,'state':state,'updated_at':datetime.now().isoformat(),**kw})
def run(stage,cmd):
    print(stage,cmd,flush=True)
    with (ROOT/'logs'/f'{NAME}_{stage}.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            status(stage,'running',pid=proc.pid)
            time.sleep(20)
        if proc.returncode:
            status(stage,'failed',exit_code=proc.returncode)
            raise RuntimeError(f'{stage} exited {proc.returncode}')
    status(stage,'completed')
run('train',[PY,'-m','accelerate.commands.launch','--num_processes','1',str(TRAIN),'--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4','--train_data_dir=data/hf/C10_train','--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8','--gradient_accumulation_steps=1','--learning_rate=1e-5','--rank=4','--max_train_steps=20000','--checkpointing_steps=500','--mixed_precision=fp16','--seed=0',f'--output_dir={DEST}'])
run('validation',[PY,'scripts/select_lora_checkpoint.py','--student',str(DEST),'--images','data/raw/wm_val'])
best=(DEST/'BEST_LORA_PATH.txt').read_text().strip()
manifest['selected_adapter']=best
manifest['selected_adapter_sha256']=sha(Path(best)/'pytorch_lora_weights.safetensors')
write_json(ROOT/'manifests'/f'{NAME}.json',manifest)
run('generation',[PY,'scripts/generate_student.py','--lora',best,'--prompts','prompts/student_eval.txt','--out',f'student_outputs/{NAME}'])
assert sha(ROOT/'metrics/serum_threshold_1pct.json')==manifest['threshold_sha256']
run('serum',[SERUM,'scripts/serum_scores.py','--images',f'student_outputs/{NAME}','--out',f'metrics/{NAME}_scores.npy'])
run('utility',[PY,'scripts/student_utility.py','--student',NAME])
import numpy as np
threshold=json.loads((ROOT/'metrics/serum_threshold_1pct.json').read_text())['threshold']
old=np.load(ROOT/'metrics/C10_seed0_scores.npy').reshape(-1)>=threshold
new=np.load(ROOT/f'metrics/{NAME}_scores.npy').reshape(-1)>=threshold
assert len(old)==len(new)==5000
rng=np.random.default_rng(20261011)
diff=new.astype(float)-old.astype(float)
boots=[float(diff[rng.integers(0,5000,5000)].mean()) for _ in range(1000)]
result={'old_tpr':float(old.mean()),'clipfix_tpr':float(new.mean()),'clipfix_minus_old_tpr':float(diff.mean()),'paired_difference_ci95':np.quantile(boots,[.025,.975]).tolist(),'threshold':threshold,'checkpoint_selection':json.loads((DEST/'validation_selection.json').read_text())['best'],'old_utility':json.loads((ROOT/'metrics/C10_seed0_utility.json').read_text()),'clipfix_utility':json.loads((ROOT/f'metrics/{NAME}_utility.json').read_text()),'scope':'Single-seed controlled rerun; no automatic baseline approval or C11 retraining'}
write_json(ROOT/'metrics'/f'{NAME}_comparison.json',result)
status('all','completed',comparison=f'metrics/{NAME}_comparison.json')
print(json.dumps(result,indent=2),flush=True)

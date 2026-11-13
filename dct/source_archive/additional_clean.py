"""Two native training-free teachers -> reused SD1.4 r4 clean distillation."""
from pathlib import Path
import os,sys,json,gc,fcntl,subprocess,hashlib,math,datetime,importlib.metadata
from benchmark.shared.additional_protocol import PROJECT,BASE,METHODS,read,write,freeze,sha,validate_data,verify_sources,threshold_record,detection
PINS=PROJECT/'benchmark/configs/additional_upstream_pins.json'
PROMPTS=PROJECT/'benchmark/shared/data/prompts.json'
CALIBRATION=PROJECT/'calibration.json'
EXPECTED_PROMPTS='5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2'

def prepare():
 pins=read(PINS);prompts=read(PROMPTS);calibration=read(CALIBRATION)
 if sha(PROMPTS)!=EXPECTED_PROMPTS:raise ValueError('Shared new_prompt_sd14 manifest changed')
 validate_data(prompts,calibration)
 assets=read(PROJECT/'assets.json')
 for role in ['teacher_path','student_path']:
  model=Path(assets[role]);assert (model/'model_index.json').is_file() and (model/'unet/config.json').is_file()
  c=read(model/'unet/config.json');assert c['in_channels']==4
  assert c['cross_attention_dim']==(1024 if role=='teacher_path' else 768)
 for method in METHODS:verify_sources(method,pins)
 BASE.mkdir(parents=True,exist_ok=True)
 versions={p:importlib.metadata.version(p) for p in ['torch','diffusers','transformers','peft','accelerate']}
 if versions['diffusers']!='0.35.1' or versions['peft']!='0.17.1':raise RuntimeError('Re-audit compatibility for changed runtime: '+str(versions))
 loader=PROJECT/'benchmark/shared/student_training/reference/harness/student.py'
 if 'Checkpoint LoRA tensors differ after loading' not in loader.read_text():raise RuntimeError('Strict full checkpoint adapter loader is required')
 identity={'prompt_sha256':sha(PROMPTS),'calibration_sha256':sha(CALIBRATION),'teacher':assets['teacher_path'],'student':assets['student_path'],
   'teacher_config_sha256':sha(Path(assets['teacher_path'])/'unet/config.json'),'student_config_sha256':sha(Path(assets['student_path'])/'unet/config.json'),
   'versions':versions,'methods':{m:pins[m]['commit'] for m in METHODS},'student_rank':4,'student_steps':20000,'evaluation_steps':[12500,20000],
   'train_images':5000,'validation_images':1000,'test_images':1000,'calibration_images':5000,'train_channel':'none','metric':'native per-image; frozen clean teacher thresholds; no aggregation',
   'strict_loader_sha256':sha(loader),'student_integration_sha256':sha(PROJECT/'benchmark/shared/student_training/integration.py')}
 freeze(BASE/'protocol.json',identity)
 freeze(BASE/'prompts.json',prompts);freeze(BASE/'calibration.json',calibration)
 print('CPU_PROTOCOL_VERIFIED',json.dumps(identity),flush=True)
 return assets,pins,prompts,calibration

def folder_manifest(folder,rows):
 from common import manifest
 manifest(folder,rows)
def verify_images(folder,rows):
 if not (folder/'image_hashes.json').exists():return False
 entries=read(folder/'image_hashes.json')
 if len(entries)!=len(rows):raise ValueError('Incomplete image manifest: '+str(folder))
 for i,(entry,row) in enumerate(zip(entries,rows)):
  if entry['file_name']!=f'sample_{i:05d}.png' or entry['text']!=row['text'] or sha(folder/entry['file_name'])!=entry['sha256']:
   raise ValueError('Image/caption hash mismatch: '+str(folder))
 return True
def generate_teacher(backend,folder,rows,marked):
 from common import save_images
 folder.mkdir(parents=True,exist_ok=True)
 if verify_images(folder,rows):return
 for i,row in enumerate(rows):
  path=folder/f'sample_{i:05d}.png'
  if not path.exists():save_images([backend.generate(row['text'],int(row['image_id']),marked)],folder,[i])
  if (i+1)%100==0:print('TEACHER_IMAGES',backend.method,folder.name,i+1,'/',len(rows),flush=True)
 folder_manifest(folder,rows)

def scores(backend,folder,output):
 from PIL import Image
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 metadata={'images_sha256':sha(folder/'image_hashes.json'),'key_sha256':backend.key_sha,'method':backend.method,'source_commit':backend.pins[backend.method]['commit']}
 freeze(output.with_suffix('.identity.json'),metadata)
 cache=output.with_suffix('.jsonl')
 entries=[]
 if cache.exists():
  with cache.open('rb+') as f:
   while True:
    start=f.tell();line=f.readline()
    if not line:break
    if not line.endswith(b'\n'):f.truncate(start);break
    entries.append(json.loads(line))
  if any(row['index']!=i for i,row in enumerate(entries)):raise RuntimeError('Noncontiguous score cache')
 manifest=read(folder/'image_hashes.json')
 for start in range(len(entries),len(manifest),4):
  names=[r['file_name'] for r in manifest[start:start+4]]
  images=[Image.open(folder/name).convert('RGB') for name in names]
  values=backend.score(images)
  if len(values)!=len(names) or not all(math.isfinite(v) for v in values):raise RuntimeError('Invalid native verifier scores')
  with cache.open('a') as f:
   for index,value in enumerate(values,start):
    row={'index':index,'score':value};f.write(json.dumps(row)+'\n');entries.append(row)
  if len(entries)%100==0:print('SCORED',backend.method,folder,len(entries),'/',len(manifest),flush=True)
 values=[row['score'] for row in entries]
 if len(values)!=len(manifest):raise ValueError('Score length mismatch')
 write(output,values);return values

def preflight(backend,folder):
 from PIL import Image
 from common import save_images
 folder.mkdir(parents=True,exist_ok=True)
 identity={'key_sha256':backend.key_sha,'upstream_commit':backend.pins[backend.method]['commit'],'protocol_sha256':sha(BASE/'protocol.json')}
 if (folder/'passed.json').exists():
  if read(folder/'passed.json')['identity']!=identity:raise RuntimeError('Preflight identity changed')
  return
 clean=[];marked=[]
 for i in range(2):
  prompt=['A photograph of a red bicycle beside a brick wall.','A photograph of a bowl of fruit on a wooden table.'][i]
  clean.append(backend.generate(prompt,9000000+i,False));marked.append(backend.generate(prompt,9000000+i,True))
 save_images(clean,folder/'clean',[0,1]);save_images(marked,folder/'marked',[0,1])
 negative=backend.score(clean);positive=backend.score(marked);repeat=backend.score(marked)[0]
 if not all(math.isfinite(x) for x in negative+positive+[repeat]):raise RuntimeError('Nonfinite native preflight')
 if abs(repeat-positive[0])>1e-5*max(abs(positive[0]),1):raise RuntimeError('Native verifier is not repeatable')
 if sum(positive)<=sum(negative):raise RuntimeError('Native paired watermark preflight separation failed; no fallback verifier is allowed')
 write(folder/'passed.json',{'identity':identity,'clean_scores':negative,'marked_scores':positive,'repeat_score':repeat,'execution_check_only':True})

def teacher(method,assets,pins,prompts,calibration):
 import torch
 from benchmark.shared.additional_native import Native
 folder=BASE/method/'teacher';folder.mkdir(parents=True,exist_ok=True)
 # Validate reusable generated images on every invocation, including a resume.
 if (folder/'complete.json').exists():
  for split in ['train','validation','test']:assert verify_images(folder/'images'/('wm_'+split),prompts[split])
  assert verify_images(folder/'images/clean_calibration',calibration)
  return
 backend=Native(method,folder,assets['teacher_path'],pins)
 preflight(backend,folder/'preflight')
 negatives=folder/'images/clean_calibration'
 generate_teacher(backend,negatives,calibration,False)
 values=scores(backend,negatives,folder/'scores/clean_calibration.json')
 threshold=threshold_record(values,method,backend.key_sha,sha(negatives/'image_hashes.json'))
 freeze(folder/'threshold.json',threshold)
 # Threshold is frozen before any marked teacher or student test scoring.
 for split in ['train','validation','test']:generate_teacher(backend,folder/'images'/('wm_'+split),prompts[split],True)
 values=scores(backend,folder/'images/wm_test',folder/'scores/wm_test.json')
 result=detection(values,threshold);write(folder/'results.json',{'method':method,'model':'teacher_SD2.1',**result})
 write(folder/'complete.json',{'protocol_sha256':sha(BASE/'protocol.json'),'key_sha256':backend.key_sha,'threshold_sha256':sha(folder/'threshold.json')})
 print('DONE_TEACHER',method,json.dumps(result),flush=True)
 del backend;gc.collect();torch.cuda.empty_cache()

def verify_checkpoint(path,identity):
 # Extract the existing CPU integrity functions without importing helper/common.py.
 from benchmark.shared.student_training.integration import function,REF
 ns={'Path':Path,'hashlib':hashlib,'json':json}
 exec(function(REF/'pipeline/student/resume_state.py','digest'),ns)
 exec(function(REF/'pipeline/student/resume_state.py','verify_checkpoint'),ns)
 return ns['verify_checkpoint'](path,identity)

def train_student(method,assets):
 from benchmark.shared.student_training.integration import sd14
 job=BASE/method/'student';source=BASE/method/'teacher/images/wm_train'
 v=sd14(job,assets['student_path'],source,4);v['config']['checkpoint_steps']=[]
 target=job/'images/teacher_wm_train';target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.symlink_to(source,target_is_directory=True)
 if target.resolve()!=source.resolve():raise RuntimeError('Wrong distillation target directory')
 driver=v['helper']/'driver.py';text=driver.read_text()
 marker="    source=patched_source(a.trainer.read_text())"
 addition="""    source=patched_source(a.trainer.read_text())
    marker='if global_step % args.checkpointing_steps == 0:'
    if source.count(marker)!=1:raise RuntimeError('Checkpoint marker changed')
    source=source.replace(marker,'if global_step in (12500, 20000):')"""
 if text.count(marker)!=1:raise RuntimeError('Student driver changed')
 text=text.replace(marker,addition);assert 'training_channel' not in text;driver.write_text(text)
 branch=job/'students/sd14_r4';identity=read(v['identity'])
 if (branch/'checkpoint-20000').exists():
  verify_checkpoint(branch/'checkpoint-20000',identity)
  write(job/'trained.json',{'steps':20000,'rank':4,'checkpoints':[12500,20000]});return
 cmd=v['builder'](v['config'],v['assets'],job,v['branch']);cmd[cmd.index('--checkpointing_steps')+1]='1250'
 cps=sorted(branch.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
 if cps:
  verify_checkpoint(cps[-1],identity);cmd+=['--resume_from_checkpoint',str(cps[-1])]
 freeze(job/'training_protocol.json',{'identity':identity,'teacher_data':str(source),'steps':20000,'rank':4,'batch':4,'seed':0,'attack':'none','endpoint_policy':'fixed12500 and20000; no validation selection','base_command':cmd[:cmd.index('--resume_from_checkpoint')] if '--resume_from_checkpoint' in cmd else cmd})
 env=os.environ.copy()
 for key in ['BASELINE_TRAIN_ATTACK','BASELINE_PIXEL_INDEX']:env.pop(key,None)
 subprocess.run([sys.executable,str(driver),'--trainer',cmd[1],'--identity-file',str(v['identity']),*cmd[2:]],cwd=PROJECT,env=env,check=True)
 for step in [12500,20000]:verify_checkpoint(branch/f'checkpoint-{step}',identity)
 write(job/'trained.json',{'steps':20000,'rank':4,'checkpoints':[12500,20000]})

def evaluate_student(method,step,assets,pins,prompts):
 import torch
 from benchmark.shared.additional_native import Native
 from benchmark.shared.evaluation.student_images import generate
 folder=BASE/method;job=folder/'student';out=job/'evaluation'/str(step);out.mkdir(parents=True,exist_ok=True)
 with (out/'lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (out/'results.json').exists():return read(out/'results.json')
  weights=out/'students/sd14_r4';weights.parent.mkdir(exist_ok=True)
  cp=job/'students/sd14_r4'/f'checkpoint-{step}'
  verify_checkpoint(cp,read(job/'resume_identity.json'))
  if not weights.exists():weights.symlink_to(cp,target_is_directory=True)
  if weights.resolve()!=cp.resolve():raise RuntimeError('Wrong checkpoint endpoint')
  images=out/'test_images'
  if not verify_images(images,prompts['test']):generate('sd14','r4',assets['student_path'],out,prompts['test'],images)
  assert verify_images(images,prompts['test'])
  gc.collect();torch.cuda.empty_cache()
  backend=Native(method,folder/'teacher',assets['teacher_path'],pins)
  values=scores(backend,images,out/'scores.json')
  threshold=read(folder/'teacher/threshold.json')
  if threshold['key_sha256']!=backend.key_sha:raise RuntimeError('Key/threshold mismatch')
  result={'method':method,'model':'SD1.4_LoRA_r4','steps':step,'training_attack':'none','test_attack':'clean',**detection(values,threshold)}
  write(out/'results.json',result);print('DONE_STUDENT',method,step,json.dumps(result),flush=True)
  del backend;gc.collect();torch.cuda.empty_cache();return result

def run(method):
 if method not in METHODS:raise ValueError('Only Tree-Ring/RingID are authorized to run; GenPTW is deferred')
 assets,pins,prompts,calibration=prepare();folder=BASE/method;folder.mkdir(parents=True,exist_ok=True)
 from benchmark.shared.process_title import configure
 configure()
 with (folder/'job.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   write(folder/'status.json',{'stage':'teacher','pid':os.getpid()})
   teacher(method,assets,pins,prompts,calibration)
   write(folder/'status.json',{'stage':'student_training','pid':os.getpid()})
   train_student(method,assets)
   results=[]
   for step in [12500,20000]:
    write(folder/'status.json',{'stage':'student_evaluation','step':step,'pid':os.getpid()})
    results.append(evaluate_student(method,step,assets,pins,prompts))
   summary={'method':method,'teacher':read(folder/'teacher/results.json'),'student_endpoints':results,'protocol_sha256':sha(BASE/'protocol.json')}
   write(folder/'results.json',summary)
   # Existing recheck workers read this compatibility summary after marking done.
   write(PROJECT/'run/recheck_stable_grow_20261105/students'/method/'clean_two_endpoints/results.json',summary)
   write(folder/'status.json',{'stage':'completed','pid':os.getpid()})
  except BaseException:
   import traceback
   write(folder/'status.json',{'stage':'failed','error':traceback.format_exc()});raise

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 if a.prepare_only:prepare()
 elif a.method:run(a.method)
 else:p.error('Use --prepare-only or --method')

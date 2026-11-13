"""Create isolated rank4/rank16 DiT trials from the verified rank32 protocol."""
from pathlib import Path
import json,os,shutil,subprocess,time,hashlib
OLD=Path('/var/lib/structure32_newprompts_rank32')
ROOT=Path('/var/lib/structure32_dit_sweep_20261101')
ROOT.mkdir(exist_ok=True)
code=(OLD/'cross_arch_jobs.py').read_text()
jobs=[('sana600m',4,0),('sana600m',16,0),('pixart512',4,1),('pixart512',16,1)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,rank,gpu in jobs:
    p=ROOT/f'{name}_rank{rank}';p.mkdir(exist_ok=True)
    assert not (p/'launch.json').exists(),'Already created: '+str(p)
    src=OLD/'cross_arch'/name
    s=code.replace("SOURCE=ROOT/'coco_disjoint_source'","SOURCE=Path('/var/lib/structure32_newprompts_rank32/coco_disjoint_source')")
    s=s.replace("p=ROOT/'cross_arch'/name;p.mkdir(parents=True,exist_ok=True);return p","return ROOT")
    s=s.replace("'rank':32","'rank':"+str(rank)).replace('LoraConfig(r=32,lora_alpha=32',f'LoraConfig(r={rank},lora_alpha={rank}')
    # Each run starts from clean base weights. Alpha/r remains1 for all ranks.
    (p/'run.py').write_text(s)
    for n in ['model_identity.json','feature_cache','adapter_targets.json']:
        if n=='adapter_targets.json':continue
        (p/n).symlink_to((src/n).resolve(),target_is_directory=(src/n).is_dir())
    for n in ['sitecustomize.py','acceptance_metrics.py']:shutil.copy2(OLD/n,p/n)
    # Import only to construct the frozen identity; this imports no GPU modules.
    import importlib.util
    spec=importlib.util.spec_from_file_location('sweep_'+name+str(rank),p/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.freeze(p/'experiment_identity.json',mod.identity(name))
    proof=json.loads((src/'feature_cache_complete.json').read_text());proof['identity_sha256']=sha(p/'experiment_identity.json')
    mod.freeze(p/'feature_cache_complete.json',proof)
    mod.freeze(p/'cache.complete.json',{'reused_exact_cache':str(src/'feature_cache'),'files':len(proof['files']),'source_identity_sha256':sha(src/'experiment_identity.json')})
    for n in ['status.json','coco_data_complete.json']:(p/n).symlink_to(OLD/n)
    # status.json must be independent: paths() points at this directory.
    (p/'status.json').unlink()
    # Existing worker waits for ROOT/status.json before training; use immutable
    # per-run marker rather than colliding with its own mutable status.
    s=s.replace("read_json(ROOT/'status.json')","read_json(ROOT/'upstream_complete.json')")
    (p/'upstream_complete.json').write_text(json.dumps({'stage':'completed'}))
    (p/'run.py').write_text(s)
    # Freeze code identity only after the final source transformation.
    identity=mod.identity(name);identity['code_sha256']=sha(p/'run.py')
    (p/'experiment_identity.json').write_text(json.dumps(identity,indent=2))
    proof['identity_sha256']=sha(p/'experiment_identity.json');(p/'feature_cache_complete.json').write_text(json.dumps(proof,indent=2))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(p),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    with (p/'supervisor.log').open('ab') as log:
        c=subprocess.Popen(['/var/lib/vllm/env/bin/python',str(p/'run.py'),name],cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    mod.write_json(p/'launch.json',{'pid':c.pid,'gpu':gpu,'rank':rank,'created':time.time(),'source':str(mod.SOURCE),'prior_student_reused':False})
    print(name,rank,'gpu',gpu,'pid',c.pid,flush=True)

"""Audited user-requested policy revision at a completed teacher checkpoint."""
from pathlib import Path
import os,json,hashlib,shutil,signal,subprocess,time,datetime
import psutil
root=Path(__file__).resolve().parent;run=root/'run';h=root/'Malicious_quantization/bitmark_lora'
checkpoint=run/'teacher_fp/candidates/kappa_1.0/checkpoint-1500/checkpoint.json'
while not checkpoint.exists():
    print('Waiting for checkpoint 1500',flush=True);time.sleep(10)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
audit=run/'policy_revisions'/stamp;audit.mkdir(parents=True)
proc=psutil.Process(3805623)
if str(root/'main.py') not in proc.cmdline():raise RuntimeError('Coordinator identity changed')
family=proc.children(recursive=True)+[proc]
for p in family:
    try:p.send_signal(signal.SIGSTOP)
    except psutil.NoSuchProcess:pass
for p in family:
    try:p.terminate();p.send_signal(signal.SIGCONT)
    except psutil.NoSuchProcess:pass
gone,alive=psutil.wait_procs(family,timeout=15)
if alive:raise RuntimeError('Old workers did not exit; refusing source replacement')
manifest=run/'code_manifest.json';old=json.loads(manifest.read_text());new=dict(old)
shutil.copy2(manifest,audit/'code_manifest.before.json')
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changes={}
for name in ['threshold_quality.py','semantic_teacher_train.py','test_threshold_quality.py']:
    source=root/'policy_patch'/name;target=h/name
    if digest(target)!=old[name]:raise RuntimeError('Unexpected old source hash')
    shutil.copy2(target,audit/name);shutil.copy2(source,target)
    new[name]=digest(target);changes[name]={'before':old[name],'after':new[name]}
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',BITMARK_RUNTIME_ROOT='/root/logs/runtime',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (audit/'tests.log').open('w') as log:
    subprocess.run(['/root/logs/gate_env/bin/python','-m','unittest','test_threshold_quality','test_semantic_threshold'],cwd=h,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
temporary=manifest.with_suffix('.partial');temporary.write_text(json.dumps(new,indent=2)+'\n');temporary.replace(manifest)
record={'user_instruction':'Remove quality stopping gates, finish and save metrics; preserve detection',
        'quality_gates_enforced':False,'detection_gate_retained':True,'changes':changes,
        'config_sha256':digest(run/'config.json'),'checkpoint':str(checkpoint),'stopped_pids':[p.pid for p in family]}
(audit/'revision.json').write_text(json.dumps(record,indent=2)+'\n')
with (root/'resume_report_only.log').open('a') as log:
    p=subprocess.Popen(['/root/logs/gate_env/bin/python',str(root/'main.py'),'--resume',str(run),'--min-free-gb','12'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'report_only_launcher.json').write_text(json.dumps({'pid':p.pid,'audit':str(audit)},indent=2)+'\n')
print('RESUMED',p.pid,'audit',audit,flush=True)

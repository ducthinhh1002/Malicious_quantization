"""Package exact runtime, PixArt native weights and frozen training cache for .23."""
from pathlib import Path
import os,json,hashlib,shutil,tarfile,time,secrets,subprocess
OLD=Path('/var/lib/structure32_newprompts_rank32')
ROOT=Path('/var/lib/structure32_dit_sweep_20261101')
P=ROOT/'pixart512_fullft';P.mkdir(exist_ok=True)
src=OLD/'cross_arch/pixart512'
model=Path(json.loads((src/'model_identity.json').read_text())['path'])
source=OLD/'coco_disjoint_source'
base=(OLD/'cross_arch_jobs.py').read_text()
train=base[base.index('def lr_factor'):base.index('\ndef smoke_infer')]
a=train.index('    model.requires_grad_(False)');b=train.index('    scheduler=')
train=train[:a]+'''    model.requires_grad_(True);model.to(torch.float32)
    model.train();model.enable_gradient_checkpointing()
    named=list(model.named_parameters());params=[v for n,v in named]
    freeze(p/'trainable_scope.json',{'scope':'all PixArt transformer parameters; VAE/text frozen',
        'trainable_count':sum(v.numel() for v in params),'precision':'FP32 parameters/AdamW; BF16 autocast'})
'''+train[b:]
a=train.index('    if smoke:\n        assert applied>0')
train=train[:a]+'''    if smoke:
        assert applied==4 and any(not torch.equal(a,b) for a,b in zip(before,params))
        dest=p/'smoke_transformer'
    else:dest=p/'transformer'
    model.save_pretrained(dest,safe_serialization=True)
    del optimizer,before,params,named,model
    gc.collect();torch.cuda.empty_cache()
    # Real native full-model save/reload and forward check on the same cached batch.
    model=PixArtTransformer2DModel.from_pretrained(dest,torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    assert all(torch.isfinite(v).all() for v in model.parameters())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(hidden_states=noisy,encoder_hidden_states=emb,encoder_attention_mask=mask,
                   timestep=times,added_cond_kwargs={'resolution':None,'aspect_ratio':None},return_dict=False)[0].chunk(2,dim=1)[0]
        loss=torch.nn.functional.mse_loss(pred.float(),target.float())
    assert torch.isfinite(loss)
    freeze(p/('smoke.json' if smoke else 'trained.json'),{'passed':True,'attempted_steps':total,'applied':applied,
        'skipped':skipped,'native_full_model_reload_forward_finite':True,
        'files':{f.name:sha256(f) for f in dest.iterdir() if f.is_file()}})
'''
header='''from pathlib import Path
import os,sys,json,time,random,shutil,fcntl,gc,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'helpers'))
from common import freeze,sha256,read_json,write_json
def paths(name):return ROOT
def status(p,stage,**kw):write_json(p/'status.json',{'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw})
'''
footer='''
if __name__=='__main__':
    try:
        with (ROOT/'lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not (ROOT/'smoke.json').exists():train('pixart512',True)
            if not (ROOT/'trained.json').exists():train('pixart512',False)
            status(ROOT,'trained_waiting_for_evaluation_on_21')
    except Exception as exc:status(ROOT,'failed',error=repr(exc));raise
'''
(P/'ft_worker.py').write_text(header+train+footer)
(P/'helpers').mkdir(exist_ok=True)
for n in ['common.py','semantic_teacher.py']:shutil.copy2(Path('/var/lib/vllm/harness')/n,P/'helpers'/n)
shutil.copy2(OLD/'sitecustomize.py',P/'sitecustomize.py')
shutil.copy2(src/'model_identity.json',P/'model_identity.json')
# Actual files go to the same pinned snapshot directory on .23; only transformer needed for train.
shutil.copy2(src/'feature_cache_complete.json',P/'feature_cache_source.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity={'model':json.loads((src/'experiment_identity.json').read_text())['model'],
    'mode':'full finetune all transformer weights; VAE/text frozen','batch_size':4,'seed':0,'steps':20000,
    'lr':'same rank32 protocol:1e-4 through7500,cosine2e-5 at12500,constant to20000',
    'precision':'FP32 trainable model and AdamW; BF16 autocast; no AMP scaling',
    'prompt_sha256':sha(source/'prompts.json'),'training_image_manifest_sha256':sha(source/'images/teacher_wm_train/image_hashes.json'),
    'teacher_adapter_sha256':'2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8',
    'source_cache_identity_sha256':sha(src/'experiment_identity.json'),'source_cache_manifest_sha256':sha(src/'feature_cache_complete.json'),
    'worker_sha256':sha(P/'ft_worker.py'),'full_state_every':1250,'selection':'fixed20000; no early stopping/test selection',
    'evaluation_server':'.21; identical model pipeline/reference images/decoder/metrics to rank32'}
(P/'experiment_identity.json').write_text(json.dumps(identity,indent=2))
proof=json.loads((src/'feature_cache_complete.json').read_text());proof['identity_sha256']=sha(P/'experiment_identity.json')
(P/'feature_cache_complete.json').write_text(json.dumps(proof,indent=2))
bundle=ROOT/'pixart_fullft_transport.tar'
print('BUILDING_BUNDLE',flush=True)
with tarfile.open(bundle,'w',dereference=True) as t:
    for f in P.rglob('*'):
        if f.is_file():t.add(f,arcname=str(f).lstrip('/'))
    for folder in [Path('/var/lib/vllm/env'),model/'transformer',model/'scheduler',src/'feature_cache']:
        # feature cache is installed into the independent fullFT job directory.
        arc=str(P/'feature_cache').lstrip('/') if folder.name=='feature_cache' else str(folder).lstrip('/')
        t.add(folder,arcname=arc)
digest=sha(bundle)
manifest={'time':time.time(),'bytes':bundle.stat().st_size,'sha256':digest,'token':secrets.token_urlsafe(24),'port':18761,
    'bundle':str(bundle),'result_dir':str(P),'source_host':'10.1.21.21','target_host':'10.1.21.23'}
(ROOT/'transport.json').write_text(json.dumps(manifest,indent=2))
print('BUNDLE_READY',bundle.stat().st_size,flush=True)

"""Queue two architecture-aware LoRA students on the frozen newprompt data."""
from pathlib import Path
import os,sys,json,time,hashlib,random,shutil,subprocess,fcntl,gc
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'coco_disjoint_source'
HARNESS=Path('/var/lib/structure32_student12500/harness')
sys.path.insert(0,str(HARNESS))
from common import read_json,write_json,freeze,sha256,image_name
MODELS={
 'sana600m':{'repo':'Efficient-Large-Model/Sana_600M_512px_diffusers','revision':'83d7a190bfd1fd070570a793d2dab5c7a3231b9d'},
 'pixart512':{'repo':'PixArt-alpha/PixArt-XL-2-512x512','revision':'50f702106901db6d0f8b67eb88e814c56ded2692'}}

def paths(name):
    p=ROOT/'cross_arch'/name;p.mkdir(parents=True,exist_ok=True);return p

def status(p,stage,**kw):write_json(p/'status.json',{'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw})

def snapshot(name):
    from huggingface_hub import snapshot_download
    p=paths(name);status(p,'download_model')
    m=MODELS[name]
    target=snapshot_download(m['repo'],revision=m['revision'],cache_dir=str(ROOT/'cross_arch/model_cache'),
        allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.tiktoken'],
        ignore_patterns=['*.fp16*','*.bin','*.onnx'],max_workers=4)
    files={str(f.relative_to(target)):sha256(f) for f in Path(target).rglob('*') if f.is_file()}
    freeze(p/'model_identity.json',{'path':target,**m,'files':files})
    return target

def pipeline(name):
    import torch
    from diffusers import SanaPipeline,PixArtAlphaPipeline
    target=read_json(paths(name)/'model_identity.json')['path']
    cls=SanaPipeline if name=='sana600m' else PixArtAlphaPipeline
    pipe=cls.from_pretrained(target,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    pipe.vae.to(torch.float32).eval().requires_grad_(False)
    pipe.text_encoder.to(torch.bfloat16).eval().requires_grad_(False)
    pipe.transformer.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_adapter(model,folder):
    """PixArt in Diffusers0.35.1 lacks the PeftAdapterMixin convenience API."""
    if hasattr(model,'load_adapter'):
        model.load_adapter(str(folder))
    else:
        from peft import LoraConfig,inject_adapter_in_model,set_peft_model_state_dict
        from safetensors.torch import load_file
        inject_adapter_in_model(LoraConfig.from_pretrained(str(folder)),model)
        result=set_peft_model_state_dict(model,load_file(str(Path(folder)/'adapter_model.safetensors')))
        assert not result.unexpected_keys,result.unexpected_keys
    model.eval().requires_grad_(False)

def infer(pipe,name,output_type='pil',**kwargs):
    # PixArt0.35.1 does not cast denoised latents to the VAE dtype on decode.
    # Keep the text/transformer/latents in BF16, explicitly decode in FP32.
    if name=='sana600m':return pipe(output_type=output_type,**kwargs)
    from types import SimpleNamespace
    latents=pipe(output_type='latent',**kwargs).images
    decoded=pipe.vae.decode(latents.to(pipe.vae.dtype)/pipe.vae.config.scaling_factor,return_dict=False)[0]
    return SimpleNamespace(images=pipe.image_processor.postprocess(decoded,output_type=output_type))

def prompt_kwargs(name):
    return {'clean_caption':False,'max_sequence_length':77,**({'complex_human_instruction':None} if name=='sana600m' else {})}

def identity(name):
    p=paths(name)
    return {'model':MODELS[name],'model_identity_sha256':sha256(p/'model_identity.json'),
        'source_run':str(SOURCE),'prompt_source':'COCO 2014; image IDs, normalized text and effective tokens disjoint from all original teacher splits',
        'prompt_sha256':sha256(SOURCE/'prompts.json'),
        'training_image_manifest_sha256':sha256(SOURCE/'images/teacher_wm_train/image_hashes.json'),
        'carrier_sha256':sha256(SOURCE/'carrier/complete.json'),
        'teacher_adapter_sha256':'2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8',
        'code_sha256':sha256(Path(__file__)),'rank':32,'batch_size':4,'seed':0,'steps':20000,
        'precision':'bf16 transformer/text/latents;fp32 LoRA/AdamW/VAE; explicit FP32 PixArt VAE decode',
        'adapter_api':'native add_adapter/load_adapter when available; otherwise PEFT inject_adapter_in_model/set_peft_model_state_dict',
        'prompt_processing':prompt_kwargs(name),'lr':'1e-4 through7500;cosine to2e-5 at12500;constant to20000',
        'objective':'flow matching uniform density shift1;target=noise-latent' if name=='sana600m' else 'uniform DDPM epsilon MSE; discard predicted variance half as official PixArt LoRA trainer',
        'inference':'native model scheduler;50steps;CFG7.5;512square;seed500000+i;no resolution binning',
        'selection':'fixed20000; no test/quality gate; report actual AMP applied/skipped counts'}

def cache(name):
    import torch
    from semantic_teacher import pixels
    p=paths(name);status(p,'cache_frozen_features');pipe=pipeline(name)
    freeze(p/'experiment_identity.json',identity(name))
    rows=read_json(SOURCE/'prompts.json')['train'];assert len(rows)==5000
    hashes=read_json(SOURCE/'images/teacher_wm_train/image_hashes.json')
    assert all(sha256(SOURCE/'images/teacher_wm_train'/r['file_name'])==r['sha256'] for r in hashes)
    folder=p/'feature_cache';folder.mkdir(exist_ok=True)
    for start in range(0,5000,32):
        f=folder/f'{start:05d}.pt'
        if f.exists():continue
        items=[]
        for i in range(start,min(start+32,5000),4):
            ids=list(range(i,min(i+4,start+32,5000)))
            with torch.inference_mode():
                emb,mask,_,_=pipe.encode_prompt([rows[j]['text'] for j in ids],do_classifier_free_guidance=False,device='cuda',**prompt_kwargs(name))
                rgb=pixels([SOURCE/'images/teacher_wm_train'/image_name(j) for j in ids])
                encoded=pipe.vae.encode(rgb)
                if name=='sana600m':z=encoded.latent*pipe.vae.config.scaling_factor;mean=z;std=None
                else:
                    dist=encoded.latent_dist;mean=dist.mean*pipe.vae.config.scaling_factor;std=dist.std*pipe.vae.config.scaling_factor
            for k,j in enumerate(ids):
                items.append({'index':j,'embedding':emb[k].cpu(),'mask':mask[k].cpu(),
                    'mean':mean[k].float().cpu(),'std':std[k].float().cpu() if std is not None else None})
        temp=f.with_suffix('.tmp');torch.save(items,temp);temp.replace(f)
        status(p,'cache_frozen_features',images=min(start+32,5000))
    freeze(p/'feature_cache_complete.json',{'identity_sha256':sha256(p/'experiment_identity.json'),
        'files':{f.name:sha256(f) for f in folder.glob('*.pt')},'n':5000})

def lr_factor(step):
    import math
    return 1. if step<=7500 else .2+.8*.5*(1+math.cos(math.pi*min(1.,(step-7500)/5000)))

def train(name,smoke=False):
    import torch,numpy as np
    from diffusers import SanaTransformer2DModel,PixArtTransformer2DModel,DDPMScheduler,FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig,get_peft_model_state_dict,inject_adapter_in_model
    from safetensors.torch import save_file,load_file
    from semantic_teacher import numerical_policy
    p=paths(name);status(p,'smoke' if smoke else 'train')
    numerical_policy();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    modelpath=read_json(p/'model_identity.json')['path']
    cls=SanaTransformer2DModel if name=='sana600m' else PixArtTransformer2DModel
    dtype=torch.bfloat16
    model=cls.from_pretrained(modelpath,subfolder='transformer',torch_dtype=dtype,local_files_only=True).cuda()
    model.requires_grad_(False)
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) or (isinstance(m,torch.nn.Conv2d) and m.groups==1)]
    config=LoraConfig(r=32,lora_alpha=32,init_lora_weights='gaussian',target_modules=targets)
    if hasattr(model,'add_adapter'):model.add_adapter(config)
    else:model=inject_adapter_in_model(config,model)
    model.train();model.enable_gradient_checkpointing()
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    assert named and all('lora_' in n for n,v in named)
    for n,v in named:v.data=v.data.float()
    params=[v for n,v in named]
    freeze(p/'adapter_targets.json',{'rank':32,'modules':targets,'trainable_count':sum(v.numel() for v in params),
        'scope':'all transformer Linear and ungrouped Conv2d; architecture-specific; base/VAE/text frozen'})
    scheduler=(FlowMatchEulerDiscreteScheduler.from_pretrained(modelpath,subfolder='scheduler',local_files_only=True)
               if name=='sana600m' else DDPMScheduler.from_pretrained(modelpath,subfolder='scheduler',local_files_only=True))
    proof=read_json(p/'feature_cache_complete.json');data=[]
    assert proof['identity_sha256']==sha256(p/'experiment_identity.json')
    for n,digest in sorted(proof['files'].items()):
        f=p/'feature_cache'/n;assert sha256(f)==digest
        data.extend(torch.load(f,map_location='cpu',weights_only=True))
    assert [r['index'] for r in data]==list(range(5000))
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),weight_decay=.01,eps=1e-8)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16);rng=torch.Generator('cpu').manual_seed(0)
    start=0;applied=0;skipped=0;order=[];offset=0
    cpdir=p/'checkpoints';cpdir.mkdir(exist_ok=True)
    existing=sorted(cpdir.glob('step*'),key=lambda f:int(f.name[4:]))
    if existing and not smoke:
        cp=existing[-1];manifest=read_json(cp/'manifest.json')
        assert manifest['identity_sha256']==sha256(p/'experiment_identity.json')
        assert all(sha256(cp/n)==v for n,v in manifest['files'].items())
        weights=load_file(str(cp/'trainable.safetensors'))
        assert weights.keys()==dict(named).keys()
        with torch.no_grad():
            for n,v in named:v.copy_(weights[n])
        state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['step'];applied=state['applied'];skipped=state['skipped'];order=state['order'];offset=state['offset']
        rng.set_state(state['data_rng']);torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        log=p/'losses.jsonl'
        if log.exists():
            records=[json.loads(s) for s in log.read_text().splitlines()]
            if any(r['step']>start for r in records):
                shutil.copy2(log,p/f'losses_before_resume_{time.time_ns()}.jsonl')
                log.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=start))
    total=4 if smoke else 20000
    before=[v.detach().clone() for v in params] if smoke else None
    logpath=p/('smoke_losses.jsonl' if smoke else 'losses.jsonl')
    with logpath.open('a') as log:
        for step in range(start,total):
            began=time.monotonic()
            if offset+4>len(order):order=torch.randperm(5000,generator=rng).tolist();offset=0
            ids=order[offset:offset+4];offset+=4;batch=[data[i] for i in ids]
            z=torch.stack([r['mean'] for r in batch]).cuda()
            if name!='sana600m':z=z+torch.stack([r['std'] for r in batch]).cuda()*torch.randn_like(z)
            z=z.to(dtype);emb=torch.stack([r['embedding'] for r in batch]).cuda().to(dtype);mask=torch.stack([r['mask'] for r in batch]).cuda()
            noise=torch.randn_like(z)
            if name=='sana600m':
                indices=(torch.rand(4,device='cuda')*scheduler.config.num_train_timesteps).long()
                times=scheduler.timesteps.to('cuda')[indices]
                sigmas=scheduler.sigmas.to('cuda')[indices].to(z.dtype).reshape(4,1,1,1)
                noisy=(1-sigmas)*z+sigmas*noise;target=noise-z
            else:
                times=torch.randint(0,scheduler.config.num_train_timesteps,(4,),device='cuda')
                noisy=scheduler.add_noise(z,noise,times);assert scheduler.config.prediction_type=='epsilon';target=noise
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=1e-4*lr_factor(step)
            with torch.autocast('cuda',dtype=dtype):
                kw={'hidden_states':noisy,'encoder_hidden_states':emb,'encoder_attention_mask':mask,'timestep':times,'return_dict':False}
                if name!='sana600m':kw['added_cond_kwargs']={'resolution':None,'aspect_ratio':None}
                pred=model(**kw)[0]
                if name!='sana600m':pred=pred.chunk(2,dim=1)[0]
                loss=torch.nn.functional.mse_loss(pred.float(),target.float())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite native training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params,1.)
            oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            was_skipped=int(scaler.get_scale()<oldscale);skipped+=was_skipped;applied+=1-was_skipped
            row={'step':step+1,'loss':float(loss.detach()),'applied':applied,'skipped':skipped,'lr':optimizer.param_groups[0]['lr'],
                 'seconds':time.monotonic()-began,'grad_scale':scaler.get_scale()}
            log.write(json.dumps(row)+'\n');log.flush()
            if (step+1)%100==0:status(p,'train',**row)
            if not smoke and (step+1)%1250==0:
                assert all(torch.isfinite(v).all() for v in params)
                dest=cpdir/f'step{step+1}';temp=cpdir/f'.pending{step+1}';temp.mkdir(exist_ok=False)
                save_file({n:v.detach().cpu().contiguous() for n,v in named},str(temp/'trainable.safetensors'))
                torch.save({'step':step+1,'applied':applied,'skipped':skipped,'order':order,'offset':offset,
                    'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'data_rng':rng.get_state(),
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},temp/'state.pt')
                freeze(temp/'manifest.json',{'identity_sha256':sha256(p/'experiment_identity.json'),
                    'files':{f.name:sha256(f) for f in temp.iterdir() if f.is_file()},'lr_scheduler':'absolute-step function stored in code, no independent state'})
                temp.rename(dest)
    if smoke:
        assert applied>0 and any(not torch.equal(a,b) for a,b in zip(before,params))
        (p/'smoke_adapter').mkdir(exist_ok=True);config.save_pretrained(p/'smoke_adapter')
        save_file({k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()},str(p/'smoke_adapter/adapter_model.safetensors'))
        freeze(p/'smoke.json',{'passed':True,'updates':applied,'skipped':skipped,'real_model':True,'discarded_weights':True})
    else:
        (p/'adapter').mkdir(exist_ok=True)
        config.save_pretrained(p/'adapter')
        save_file({k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model).items()},str(p/'adapter/adapter_model.safetensors'))
        freeze(p/'trained.json',{'attempted_steps':total,'applied':applied,'skipped':skipped,
            'adapter_files':{f.name:sha256(f) for f in (p/'adapter').iterdir() if f.is_file()}})

def smoke_infer(name):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    p=paths(name);status(p,'smoke_inference');pipe=pipeline(name)
    load_adapter(pipe.transformer,p/'smoke_adapter')
    actual=get_peft_model_state_dict(pipe.transformer);expected=load_file(str(p/'smoke_adapter/adapter_model.safetensors'))
    assert actual.keys()==expected.keys()
    for k,v in actual.items():torch.testing.assert_close(v.cpu(),expected[k].to(v.dtype),atol=0,rtol=0)
    with torch.inference_mode():
        image=infer(pipe,name,prompt=read_json(SOURCE/'prompts.json')['validation'][0]['text'],height=512,width=512,
            num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator('cuda').manual_seed(87654321),
            use_resolution_binning=False,output_type='pt',**prompt_kwargs(name)).images
    assert torch.isfinite(image).all() and image.shape[-2:]==(512,512) and image.std()>1e-5
    freeze(p/'smoke_inference.json',{'passed':True,'adapter_reload_exact':True,'steps':4,'scientific_result':False})

def generate(name):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    p=paths(name);status(p,'generate_unadapted_and_student');pipe=pipeline(name)
    prompts=read_json(SOURCE/'prompts.json')['test'];assert len(prompts)==1000
    for label in ['unadapted','student']:
        if label=='student':
            load_adapter(pipe.transformer,p/'adapter')
            actual=get_peft_model_state_dict(pipe.transformer);expected=load_file(str(p/'adapter/adapter_model.safetensors'))
            assert actual.keys()==expected.keys()
            for n,t in actual.items():torch.testing.assert_close(t.cpu(),expected[n].to(t.dtype),atol=0,rtol=0)
        folder=p/'images'/label;folder.mkdir(parents=True,exist_ok=True)
        freeze(folder/'generation.json',{'label':label,'prompts_sha256':sha256(SOURCE/'prompts.json'),
            'model_revision':MODELS[name]['revision'],'steps':50,'cfg':7.5,'seed':'500000+i','batch':4,
            'scheduler':type(pipe.scheduler).__name__,'scheduler_config_sha256':sha256(Path(read_json(p/'model_identity.json')['path'])/'scheduler/scheduler_config.json'),
            'prompt_processing':prompt_kwargs(name)})
        for start in range(0,1000,4):
            ids=list(range(start,min(start+4,1000)));files=[folder/image_name(i) for i in ids]
            if all(f.exists() for f in files):continue
            with torch.inference_mode():
                images=infer(pipe,name,prompt=[prompts[i]['text'] for i in ids],height=512,width=512,num_inference_steps=50,
                    guidance_scale=7.5,generator=[torch.Generator('cuda').manual_seed(500000+i) for i in ids],
                    use_resolution_binning=False,**prompt_kwargs(name)).images
            for f,im in zip(files,images):
                temp=f.with_suffix('.partial.png');im.save(temp);temp.replace(f)
            status(p,'generate_unadapted_and_student',label=label,images=start+len(ids))
        freeze(folder/'image_hashes.json',[{'file_name':image_name(i),'sha256':sha256(folder/image_name(i))} for i in range(1000)])

def evaluate(name):
    import torch,lpips
    from semantic_teacher import extractor,pixels
    from semantic_carrier import SemanticCarrier
    sys.path.insert(0,str(ROOT))
    from acceptance_metrics import aggregate_metrics
    p=paths(name);status(p,'watermark_metrics');ext=extractor(SOURCE);carrier=SemanticCarrier.load(SOURCE/'carrier','cuda')
    prompts=read_json(SOURCE/'prompts.json')['test'];results={}
    for label in ['unadapted','student']:
        coords=[]
        for start in range(0,1000,8):
            ids=list(range(start,min(start+8,1000)))
            with torch.inference_mode():q=carrier.coordinates(ext.encode_image(pixels([p/'images'/label/image_name(i) for i in ids]),[prompts[i]['text'] for i in ids]))
            coords.extend(q.cpu().tolist())
        freeze(p/f'{label}_coordinates.json',coords)
        results[label]=aggregate_metrics(coords,carrier.signs.cpu().tolist(),sizes=(1,4,8,32,64,256))
    freeze(p/'watermark_results.json',results)
    del ext;gc.collect();torch.cuda.empty_cache()
    net=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False);ds=[]
    for start in range(0,1000,8):
        ids=range(start,min(start+8,1000))
        with torch.inference_mode():d=net(pixels([p/'images/student'/image_name(i) for i in ids]),pixels([p/'images/unadapted'/image_name(i) for i in ids])).flatten()
        ds.extend(d.cpu().tolist())
    freeze(p/'paired_lpips.json',{'n':1000,'lpips':sum(ds)/1000,'per_image':ds})
    # Run the established CLIP/FID/KID evaluator with the same clean references.
    audit=p/'utility_run';audit.mkdir(exist_ok=True)
    for n in ['config.json','prompts.json']:
        if not (audit/n).exists():(audit/n).symlink_to((SOURCE/n).resolve())
    for src,dest in [(SOURCE/'images/teacher_clean_test',audit/'images/teacher_clean_test'),(p/'images/student',audit/'images/student')]:
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():dest.symlink_to(src.resolve(),target_is_directory=True)
    with (p/'quality.log').open('a') as log:
        subprocess.run([sys.executable,str(HARNESS/'utility.py'),'--run',str(audit),'--labels','student'],check=True,stdout=log,stderr=subprocess.STDOUT)

def worker(name):
    p=paths(name)
    with (p/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (p/'model_identity.json').exists():snapshot(name)
        status(p,'queued_waiting_for_sd14_completion')
        while read_json(ROOT/'status.json').get('stage')!='completed':
            if read_json(ROOT/'status.json').get('stage')=='failed':status(p,'queued_sd14_failed_needs_attention')
            time.sleep(30)
        status(p,'queued_waiting_for_disjoint_coco_images')
        while not (ROOT/'coco_data_complete.json').exists():time.sleep(30)
        freeze(p/'input_reuse.json',{'source':str(SOURCE),'training_manifest_sha256':sha256(SOURCE/'images/teacher_wm_train/image_hashes.json'),
            'prompts_sha256':sha256(SOURCE/'prompts.json'),'teacher_unchanged':True})
        for stage,args in [('cache',[]),('smoke',[]),('smoke_infer',[]),('train',[]),('generate',[]),('evaluate',[])]:
            marker=p/(stage+'.complete.json')
            if marker.exists():continue
            with (p/(stage+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__)),name,stage],check=True,stdout=log,stderr=subprocess.STDOUT)
            freeze(marker,{'completed':time.time()})
        status(p,'completed')

if __name__=='__main__':
    name=sys.argv[1];assert name in MODELS
    try:
        if len(sys.argv)==2:worker(name)
        else:
            op=sys.argv[2]
            if op=='cache':cache(name)
            elif op=='smoke':train(name,True)
            elif op=='smoke_infer':smoke_infer(name)
            elif op=='train':train(name)
            elif op=='generate':generate(name)
            elif op=='evaluate':evaluate(name)
            else:raise ValueError(op)
    except Exception as exc:status(paths(name),'failed',error=repr(exc));raise

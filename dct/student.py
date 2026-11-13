"""Ordinary rank4 image-only LoRA. Clean native VAE/text; no detector in training."""
from pathlib import Path
import os,sys,json,time,random,gc,math
import numpy as np
import torch
from common import ROOT,CACHE,read,write,freeze,sha,config,policy,pixels,image_name,status,manifest
from saturday_channels import transform,SETTINGS
from training_checks import finite,checked_update

def model_path(arch):return read(CACHE/'models'/f'{arch}.json')['path']
def pipe_for(arch):
    from diffusers import StableDiffusionPipeline,SanaPipeline,PixArtAlphaPipeline
    cls={'sd14':StableDiffusionPipeline,'sana600m':SanaPipeline,'pixart512':PixArtAlphaPipeline}[arch]
    kw={'safety_checker':None,'requires_safety_checker':False} if arch=='sd14' else {}
    pipe=cls.from_pretrained(model_path(arch),torch_dtype=torch.float16 if arch=='sd14' else torch.bfloat16,local_files_only=True,**kw).to('cuda')
    pipe.vae.to(torch.float32).eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False);pipe.set_progress_bar_config(disable=True)
    return pipe
def prompt_kwargs(arch):return {} if arch=='sd14' else {'clean_caption':False,'max_sequence_length':77,**({'complex_human_instruction':None} if arch=='sana600m' else {})}
def lock(path):
    import fcntl
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('w');fcntl.flock(f,fcntl.LOCK_EX);return f
def cache(method,arch,attack):
    """Reuse text encodings across methods/attacks; cache exact tensor manipulations."""
    c=config();rows=read(ROOT/'prompts.json')['train'];source=ROOT/'teachers'/method/'images/train'
    folder=ROOT/'features'/method/arch/attack;text=ROOT/'text_features'/arch
    folder.mkdir(parents=True,exist_ok=True);text.mkdir(parents=True,exist_ok=True)
    identity={'method':method,'arch':arch,'attack':attack,'settings':SETTINGS[attack],
        'train_manifest_sha256':sha(source/'image_hashes.json'),'prompt_sha256':sha(ROOT/'prompts.json'),
        'model_manifest_sha256':sha(CACHE/'models'/f'{arch}.json'),'channel_code_sha256':sha(Path(__file__).parent/'saturday_channels.py')}
    freeze(folder/'identity.json',identity)
    with lock(folder/'lock'):
        if (folder/'complete.json').exists():return folder,text
        pipe=pipe_for(arch)
        with lock(text/'lock'):
            freeze(text/'identity.json',{'arch':arch,'prompts':sha(ROOT/'prompts.json'),'model':identity['model_manifest_sha256'],'processing':prompt_kwargs(arch)})
            for start in range(0,len(rows),32):
                f=text/f'{start:05d}.pt'
                if f.exists():continue
                items=[]
                for offset in range(start,min(start+32,len(rows)),4):
                    batch=rows[offset:min(offset+4,start+32,len(rows))]
                    with torch.inference_mode():
                        v=pipe.encode_prompt([r['text'] for r in batch],device='cuda',num_images_per_prompt=1,do_classifier_free_guidance=False,**prompt_kwargs(arch))
                    finite('cached text embedding',v[0])
                    for j in range(len(batch)):items.append({'embedding':v[0][j].cpu(),'mask':None if arch=='sd14' else v[1][j].cpu()})
                tmp=f.with_suffix('.tmp');torch.save(items,tmp);tmp.replace(f)
            freeze(text/'complete.json',{'n':len(rows),'files':{f.name:sha(f) for f in text.glob('*.pt')}})
        hashes=read(source/'image_hashes.json')
        if any(sha(source/r['file_name'])!=r['sha256'] for r in hashes):raise ValueError('Teacher image hash mismatch')
        for start in range(0,len(rows),32):
            f=folder/f'{start:05d}.pt'
            if f.exists():continue
            items=[]
            for offset in range(start,min(start+32,len(rows)),4):
                ids=list(range(offset,min(offset+4,start+32,len(rows))))
                x=pixels([source/image_name(i) for i in ids])
                x=torch.stack([transform(t,attack,i) for t,i in zip(x,ids)]).cuda()*2-1
                with torch.inference_mode():
                    posterior=pipe.vae.encode(x)
                    if arch=='sana600m':mean=posterior.latent*pipe.vae.config.scaling_factor;std=None
                    else:mean=posterior.latent_dist.mean*pipe.vae.config.scaling_factor;std=posterior.latent_dist.std*pipe.vae.config.scaling_factor
                finite('cached VAE latent mean',mean)
                if std is not None:finite('cached VAE latent std',std)
                for j,i in enumerate(ids):items.append({'index':i,'mean':mean[j].float().cpu(),'std':std[j].float().cpu() if std is not None else None})
            tmp=f.with_suffix('.tmp');torch.save(items,tmp);tmp.replace(f)
        freeze(folder/'complete.json',{'n':len(rows),'identity_sha256':sha(folder/'identity.json'),'files':{f.name:sha(f) for f in folder.glob('*.pt')}})
        del pipe;gc.collect();torch.cuda.empty_cache()
    return folder,text
def factor(step):return 1. if step<=7500 else .2+.8*.5*(1+math.cos(math.pi*min(1.,(step-7500)/5000)))
def training_model(arch):
    from diffusers import UNet2DConditionModel,SanaTransformer2DModel,PixArtTransformer2DModel
    cls={'sd14':UNet2DConditionModel,'sana600m':SanaTransformer2DModel,'pixart512':PixArtTransformer2DModel}[arch]
    return cls.from_pretrained(model_path(arch),subfolder='unet' if arch=='sd14' else 'transformer',torch_dtype=torch.float16 if arch=='sd14' else torch.bfloat16,local_files_only=True).cuda()
def attach_adapter(model,cfg):
    from peft import inject_adapter_in_model
    if getattr(model,'peft_config',None):raise RuntimeError('Model already has an adapter')
    # Diffusers UNet/Sana provide add_adapter; pinned PixArt does not.
    if callable(getattr(model,'add_adapter',None)):model.add_adapter(cfg)
    else:inject_adapter_in_model(cfg,model)
    for n,p in model.named_parameters():
        if 'lora_' in n:p.data=p.data.float()

def add_lora(model,arch):
    from peft import LoraConfig
    model.requires_grad_(False)
    targets=[n for n,m in model.named_modules() if (isinstance(m,torch.nn.Linear) and (arch!='sd14' or n.endswith(('to_q','to_k','to_v','to_out.0')))) or (isinstance(m,torch.nn.Conv2d) and m.groups==1)]
    cfg=LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=targets)
    if not targets:raise RuntimeError('No LoRA target modules found')
    attach_adapter(model,cfg)
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):
        raise RuntimeError('A base-model parameter was accidentally unfrozen')
    return cfg,targets
def load_adapter(model,folder):
    from peft import LoraConfig,set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    expected=load_file(str(Path(folder)/'adapter_model.safetensors'))
    for n,t in expected.items():finite('saved adapter '+n,t)
    attach_adapter(model,LoraConfig.from_pretrained(str(folder)))
    x=set_peft_model_state_dict(model,expected)
    if x.unexpected_keys or any('lora_' in n for n in x.missing_keys):
        raise RuntimeError('Adapter key mismatch: '+str(x))
    actual=get_peft_model_state_dict(model)
    if actual.keys()!=expected.keys():raise RuntimeError('Adapter tensor names differ after reload')
    for n,t in actual.items():torch.testing.assert_close(t.float().cpu(),expected[n].float(),rtol=0,atol=0)
    model.eval().requires_grad_(False)
def train(method,arch,attack):
    from diffusers import DDPMScheduler,FlowMatchEulerDiscreteScheduler
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file,load_file
    c=config();p=ROOT/'students'/method/arch/attack;p.mkdir(parents=True,exist_ok=True)
    status(p,'cache');folder,text=cache(method,arch,attack)
    identity={'feature_identity':sha(folder/'identity.json'),'text_identity':sha(text/'identity.json'),
        'code':sha(__file__),'numerical_checks':sha(Path(__file__).parent/'training_checks.py'),'rank':4,'steps':c['steps'],'seed':c['seed'],'batch':c['batch_size'],'schedule':'constant1e-4 to7500; cosine2e-5 at12500; constant afterwards',
        'objective':'ordinary image-only denoising/flow MSE; no detector, owner key, quality loss, or model optimization','selection':'fixed last step'}
    freeze(p/'identity.json',identity)
    if (p/'trained.json').exists():return
    data=[];emb=[]
    for root,result in [(folder,data),(text,emb)]:
        for n,h in sorted(read(root/'complete.json')['files'].items()):
            assert sha(root/n)==h;result.extend(torch.load(root/n,map_location='cpu',weights_only=True))
    assert len(data)==len(emb)==c['train_images']
    policy();random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=training_model(arch);lora,targets=add_lora(model,arch);model.train();model.enable_gradient_checkpointing()
    named={n:v for n,v in model.named_parameters() if v.requires_grad};params=list(named.values());assert params
    freeze(p/'targets.json',{'modules':targets,'parameters':sum(v.numel() for v in params),'rank':4,'clean_model_sha256':sha(CACHE/'models'/f'{arch}.json')})
    scheduler=(FlowMatchEulerDiscreteScheduler if arch=='sana600m' else DDPMScheduler).from_pretrained(model_path(arch),subfolder='scheduler',local_files_only=True)
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=arch=='sd14',init_scale=1024.);dtype=torch.float16 if arch=='sd14' else torch.bfloat16
    rng=torch.Generator('cpu').manual_seed(c['seed']);order=[];offset=0;start=applied=skipped=0
    cpdir=p/'checkpoints';cpdir.mkdir(exist_ok=True);existing=sorted((x for x in cpdir.glob('step*') if (x/'manifest.json').exists()),key=lambda x:int(x.name[4:]))
    if existing:
        cp=existing[-1];proof=read(cp/'manifest.json');assert proof['identity']==sha(p/'identity.json')
        for n,h in proof['files'].items():assert sha(cp/n)==h
        saved=load_file(str(cp/'trainable.safetensors'));assert saved.keys()==named.keys()
        with torch.no_grad():
            for n,t in named.items():t.copy_(saved[n])
        state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['step'];applied=state['applied'];skipped=state['skipped'];order=state['order'];offset=state['offset'];rng.set_state(state['data_rng'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        log=p/'losses.jsonl'
        if log.exists():
            a=[json.loads(x) for x in log.read_text().splitlines()];log.write_text(''.join(json.dumps(x)+'\n' for x in a if x['step']<=start))
    consecutive_skips=0
    status(p,'train',step=start,applied=applied,skipped=skipped)
    with (p/'losses.jsonl').open('a',buffering=1) as log:
        for step in range(start,c['steps']):
            began=time.monotonic();batch=c['batch_size']
            if offset+batch>len(order):order=torch.randperm(len(data),generator=rng).tolist();offset=0
            ids=order[offset:offset+batch];offset+=batch
            z=torch.stack([data[i]['mean'] for i in ids]).cuda()
            if arch!='sana600m':z=z+torch.stack([data[i]['std'] for i in ids]).cuda()*torch.randn_like(z)
            z=z.to(dtype);text_emb=torch.stack([emb[i]['embedding'] for i in ids]).cuda().to(dtype);mask=None if arch=='sd14' else torch.stack([emb[i]['mask'] for i in ids]).cuda();noise=torch.randn_like(z)
            if arch=='sana600m':
                ix=(torch.rand(batch,device='cuda')*scheduler.config.num_train_timesteps).long();t=scheduler.timesteps.cuda()[ix];sigma=scheduler.sigmas.cuda()[ix].to(dtype).reshape(batch,1,1,1)
                noisy=(1-sigma)*z+sigma*noise;target=noise-z
            else:
                t=torch.randint(0,scheduler.config.num_train_timesteps,(batch,),device='cuda');noisy=scheduler.add_noise(z,noise,t)
                target=noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(z,noise,t)
            opt.zero_grad(set_to_none=True)
            finite('training latent',noisy);finite('text embedding',text_emb);finite('training target',target)
            for group in opt.param_groups:group['lr']=1e-4*factor(step)
            with torch.autocast('cuda',dtype=dtype):
                if arch=='sd14':pred=model(noisy,t,encoder_hidden_states=text_emb).sample
                else:
                    kw={'hidden_states':noisy,'encoder_hidden_states':text_emb,'encoder_attention_mask':mask,'timestep':t,'return_dict':False}
                    if arch=='pixart512':kw['added_cond_kwargs']={'resolution':None,'aspect_ratio':None}
                    pred=model(**kw)[0]
                    if arch=='pixart512':pred=pred.chunk(2,dim=1)[0]
                loss=torch.nn.functional.mse_loss(pred.float(),target.float())
            try:
                finite('student prediction',pred)
                update=checked_update(loss,opt,scaler,params)
                skip=update['skipped'];skipped+=skip;applied+=1-skip
                consecutive_skips=consecutive_skips+1 if skip else 0
                if consecutive_skips>=16:raise RuntimeError('16 consecutive AMP overflows; training stopped')
            except Exception as e:
                write(p/'failure.json',{'arch':arch,'step':step+1,'image_indices':ids,'error':str(e),'applied':applied,'skipped':skipped})
                raise
            row={**update,'step':step+1,'loss':float(loss.detach()),'applied':applied,'skipped':skipped,'step_skipped':skip,'seconds':time.monotonic()-began,'lr':opt.param_groups[0]['lr']};log.write(json.dumps(row,allow_nan=False)+'\n')
            if (step+1)%100==0:status(p,'train',**row)
            if (step+1)%c['checkpoint_every']==0 or step+1==c['steps']:
                cp=cpdir/f'step{step+1}';tmp=cpdir/f'.pending{step+1}';tmp.mkdir(exist_ok=False)
                save_file({n:v.detach().cpu().contiguous() for n,v in named.items()},str(tmp/'trainable.safetensors'))
                torch.save({'step':step+1,'applied':applied,'skipped':skipped,'order':order,'offset':offset,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'data_rng':rng.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp/'state.pt')
                write(tmp/'manifest.json',{'identity':sha(p/'identity.json'),'files':{f.name:sha(f) for f in tmp.iterdir()}});tmp.rename(cp)
    if not applied:raise RuntimeError('No optimizer update was applied')
    dest=p/'adapter';dest.mkdir(exist_ok=True);lora.save_pretrained(dest)
    save_file({n:v.detach().cpu().contiguous() for n,v in get_peft_model_state_dict(model).items()},str(dest/'adapter_model.safetensors'))
    assert any(v.detach().abs().max()>0 for n,v in named.items() if 'lora_B' in n)
    freeze(p/'trained.json',{'step':c['steps'],'applied':applied,'skipped':skipped,'adapter_sha256':sha(dest/'adapter_model.safetensors')})
def generate(method,arch,attack):
    c=config();p=ROOT/'students'/method/arch/attack;status(p,'generate');rows=read(ROOT/'prompts.json')['test'];pipe=pipe_for(arch)
    model=pipe.unet if arch=='sd14' else pipe.transformer;load_adapter(model,p/'adapter')
    if arch=='sd14':pipe.vae.to(torch.float16)
    folder=p/'images';folder.mkdir(exist_ok=True)
    freeze(folder/'identity.json',{'prompts':sha(ROOT/'prompts.json'),'adapter_sha256':sha(p/'adapter/adapter_model.safetensors'),'model':sha(CACHE/'models'/f'{arch}.json'),'steps':c['inference_steps'],'cfg':c['cfg'],'seeds':'500000+i','post_generation_manipulation':False,'native_scheduler':type(pipe.scheduler).__name__})
    for start in range(0,len(rows),c['generation_batch']):
        ids=list(range(start,min(start+c['generation_batch'],len(rows))))
        if all((folder/image_name(i)).exists() for i in ids):continue
        with torch.inference_mode():
            kw={'prompt':[rows[i]['text'] for i in ids],'generator':[torch.Generator('cuda').manual_seed(500000+i) for i in ids],'height':512,'width':512,'num_inference_steps':c['inference_steps'],'guidance_scale':c['cfg'],**prompt_kwargs(arch)}
            if arch!='sd14':kw['use_resolution_binning']=False
            if arch=='pixart512':
                z=pipe(output_type='latent',**kw).images
                images=pipe.image_processor.postprocess(pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,return_dict=False)[0],output_type='pil')
            else:images=pipe(**kw).images
        from common import save_images
        save_images(images,folder,ids);status(p,'generate',images=start+len(ids))
    manifest(folder,rows)

"""Native merged UNet experiment with the frozen reader from the LALT study.

The reader architecture/score matches the recorded server implementation. No latent
transport, key or reader is used at deployment. Tail backpropagation is truncated,
but its prefix is freshly sampled with current teacher weights, never a clean cache.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from new_watermark.paths import read_json, write_json, freeze, sha256


class TransportReader(nn.Module):
    def __init__(self, bits=8, width=32, input_size=256, full_resolution=True):
        super().__init__()
        self.bits, self.input_size, self.full_resolution=bits,input_size,full_resolution
        layers=[]; previous=6
        for channels in (width,width*2,width*4,width*4):
            layers.extend([nn.Conv2d(previous,channels,3,padding=1),nn.SiLU(),
                           nn.Conv2d(channels,channels,3,stride=2,padding=1),nn.SiLU()])
            previous=channels
        self.body=nn.Sequential(*layers); self.head=nn.Conv2d(previous,bits+1,1)

    def at_size(self, images, size):
        x=F.interpolate(images,(size,size),mode='bilinear',align_corners=False,antialias=True)*2-1
        low=F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,stride=1)
        logits=self.head(self.body(torch.cat((x,(x-low)*4),1))).mean((-2,-1))
        return logits[:,0],logits[:,1:]

    def views(self,images):
        sizes=[min(self.input_size,images.shape[-1])]
        if self.full_resolution and images.shape[-1] not in sizes: sizes.append(images.shape[-1])
        return [self.at_size(images,s) for s in sizes]

    def forward(self,images):
        views=self.views(images)
        return tuple(torch.stack([v[i] for v in views]).mean(0) for i in range(2))


def key_bits(keys,device):
    return ((torch.tensor(keys,device=device)[:,None]>>torch.arange(8,device=device))&1).float()


def reader_features(reader,pixels):
    presence,bits=reader(pixels)
    return torch.cat((presence[:,None],bits),1)


def keyed_scores(features,keys):
    codes=key_bits(keys,features.device)
    return F.logsigmoid(features[:,:1])+F.logsigmoid(features[:,None,1:]*(2*codes[None]-1)).sum(-1)/math.sqrt(8)


def load_reader(run,device='cuda'):
    c=read_json(run/'config.json'); path=Path(c['owner_checkpoint'])
    if sha256(path)!=c['owner_sha256']: raise RuntimeError('Frozen owner checkpoint hash mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not state['config'].get('transport') or state['config']['bits']!=8:
        raise ValueError('Expected 8-bit transport reader checkpoint')
    reader=TransportReader(full_resolution=state['config']['reader_full_resolution'])
    reader.load_state_dict(state['reader'],strict=True)
    return reader.float().to(device).eval().requires_grad_(False)


def calibrate_native(run,cal):
    import numpy as np
    from new_watermark.method import score_features
    c=read_json(run/'config.json')
    d={'type':'frozen_lalt_reader_8bit','keys':[c['owner_key'],c['owner_wrong_key']],
       'reader_sha256':c['owner_sha256'],'bits':8,'test_used_for_calibration':False,
       'total_keyset_nominal_fpr':c['fpr'],'nominal_per_key_fpr':c['fpr']/2,
       'calibration_n':len(cal['features']),'calibration_features_sha256':sha256(run/'features/clean_calibration.json')}
    values=score_features(cal['features'],d)
    # Same conservative order statistic as the prior LALT study. nextafter permits >=.
    index=math.ceil((len(values)+1)*(1-c['fpr']/2))-1
    if index>=len(values): raise ValueError('Need more clean calibration images for two-key FPR')
    d['thresholds']=[float(np.nextafter(np.sort(values[:,i])[index],np.inf)) for i in range(2)]
    d['decision']='score >= nextafter(conformal clean quantile, +inf); two-key Bonferroni'
    freeze(run/'result/detector.json',d)


def trajectory(unet,scheduler_config,noise,text,steps,tail_steps,gradient):
    from diffusers import DDIMScheduler
    if not 0<=tail_steps<=steps or steps<1:
        raise ValueError('Invalid trajectory length')
    tail_steps=tail_steps or steps
    scheduler=DDIMScheduler.from_config(scheduler_config,clip_sample=False)
    scheduler.set_timesteps(steps,device=noise.device)
    z=noise*scheduler.init_noise_sigma
    for i,t in enumerate(scheduler.timesteps):
        active=gradient and i>=steps-tail_steps
        if i==steps-tail_steps: z=z.detach()
        def predict(x,t,h):
            return unet(x,t,h).sample
        with torch.enable_grad() if active else torch.no_grad():
            inp=scheduler.scale_model_input(torch.cat((z,z)),t)
            pred=checkpoint(predict,inp,t,text,use_reentrant=False) if active else predict(inp,t,text)
            u,v=pred.chunk(2)
            z=scheduler.step(u+7.5*(v-u),t,z,eta=0.).prev_sample
    return z


def learning_rate_factor(step,total,floor,warmup=0):
    if not 1<=step<=total or not 0<=warmup<total:
        raise ValueError('Invalid learning rate step/warmup')
    if warmup and step<=warmup:
        return step/warmup
    progress=(step-warmup-1)/max(1,total-warmup-1)
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def prior_loss(unet,scheduler,latent,text,noise,timestep,budget):
    """Preserve both CFG branches on clean noised latents, across training timesteps.

    This is a same-input denoiser trust region, NOT AquaLoRA's shifted-latent PPFT.
    Base prediction is stopped; the adapted prediction keeps parameter gradients.
    """
    noisy=scheduler.add_noise(latent,noise,timestep)
    inp=torch.cat((noisy,noisy))
    unet.disable_adapters()
    try:
        with torch.no_grad():
            reference=unet(inp,timestep,text).sample
    finally:
        unet.enable_adapters()
    prediction=checkpoint(lambda x,t,h:unet(x,t,h).sample,inp,timestep,text,use_reentrant=False)
    # Normalize per CFG branch; a larger reference cannot hide drift in the other.
    axes=tuple(range(1,prediction.ndim))
    relative=(prediction-reference).square().mean(axes)/reference.square().mean(axes).clamp_min(1e-6)
    ratio=relative/budget
    loss=(.01*ratio+.1*F.relu(ratio-1).square()).mean()
    return loss,{'prior_relative_mse':relative.mean(),'prior_loss':loss,
                 'prior_budget_exceeded':(ratio>1).float().mean()}


def decode(vae,z,gradient):
    def f(x): return vae.decode(x/vae.config.scaling_factor).sample
    x=checkpoint(f,z,use_reentrant=False) if gradient else f(z)
    x=(x/2+.5).clamp(0,1)
    rounded=(x*255).round()/255
    return x+(rounded-x).detach() if gradient else rounded


def bit_objective(logits,codes,tail_weight):
    errors=F.binary_cross_entropy_with_logits(logits,codes,reduction='none')
    # Smooth maximum per image, symmetric over all bit positions. No test-derived weights.
    tail=(.25*torch.logsumexp(errors/.25,dim=-1)-.25*math.log(errors.shape[-1])).mean()
    return errors.mean()+tail_weight*tail,errors.mean(),tail


def detection_loss(reader,pixels,detector,margin,bit_tail_weight=0.):
    features=reader_features(reader,pixels); scores=keyed_scores(features,detector['keys'])
    codes=key_bits(detector['keys'][:1],pixels.device).expand(len(pixels),-1)
    bit,average,tail=bit_objective(features[:,1:],codes,bit_tail_weight)
    detection=F.softplus(detector['thresholds'][0]+margin-scores[:,0]).mean()
    specificity=F.softplus(margin-(scores[:,0]-scores[:,1])).mean()
    return bit+detection+.5*specificity,{
        'bit_bce':average,'bit_tail_loss':tail,'weighted_bit_loss':bit,
        'detection_margin_loss':detection,'specificity_loss':specificity,
        'correct_score':scores[:,0].mean(),'wrong_score':scores[:,1].mean(),
        'presence_probability':features[:,0].sigmoid().mean(),
        'bit_accuracy':((features[:,1:]>0)==codes.bool()).float().mean()}


def reader_diagnostics(features,keys):
    """Describe existing scores without fitting or recalibrating the reader."""
    import numpy as np
    x=np.asarray(features,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=9 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty presence + 8-bit reader logits')
    codes=np.asarray([(keys[0]>>i)&1 for i in range(8)])
    matches=(x[:,1:]>0)==codes
    return {'n':len(x),'bit_order':'little endian, index 0 is least significant',
            'target_bits':codes.tolist(),'per_bit_accuracy':matches.mean(0).tolist(),
            'mean_bit_logits':x[:,1:].mean(0).tolist(),'mean_presence_logit':float(x[:,0].mean()),
            'exact_message_accuracy':float(matches.all(1).mean()),
            'note':'Presence confidence is not correct-key decoding. A fixed key and biased reader need not yield 50% bit accuracy on clean images.'}


def train_native(run):
    from peft import LoraConfig
    from diffusers import DDPMScheduler
    from safetensors.torch import load_file
    import lpips
    from new_watermark.data import load_base
    from new_watermark.teacher import save_adapter,verify_cache
    from student import checkpoint_hashes
    c=read_json(run/'config.json'); assets=read_json(run/'assets.json')
    detector=read_json(run/'result/detector.json'); cache=verify_cache(run)
    out=run/'teacher'; out.mkdir(exist_ok=True)
    freeze(out/'protocol.json',{'config_sha256':sha256(run/'config.json'),
        'detector_sha256':sha256(run/'result/detector.json'),'native_merged_weights':True,
        'teacher_sampling':'current adapted UNet at ALL steps; full gradient when tail=0, otherwise truncated',
        'gradient_steps':c['teacher_tail_steps'] or c['inference_steps'],
        'loss':'8-bit BCE + frozen-threshold margin + wrong-key separation + image quality + VAE channel + optional denoiser preservation and short LoRA transfer',
        'prior_weight':c.get('native_prior_weight',0.),
        'prior_budget':c.get('native_prior_budget',.01),
        'warmup_steps':c.get('teacher_warmup_steps',0),
        'bit_tail_weight':c.get('native_bit_tail_weight',0.),
        'meta_proxy':'two ordinary AdamW updates; disjoint predicted-x0 probe, not long-horizon equivalence',
        'reader_updated':False,'test_used_for_training':False,'quality_policy':'report and continue'})
    if (out/'complete.json').exists():
        for step,hashes in read_json(out/'complete.json')['snapshots'].items():
            if checkpoint_hashes(out/f'snapshot-{step}')!=hashes: raise RuntimeError('Changed native snapshot')
        return
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    pipe=load_base(run,dtype=torch.float32); pipe.text_encoder.to('cpu')
    unet=pipe.unet.requires_grad_(False)
    targets=[n for n,m in unet.named_modules() if any(n.endswith('.'+s) for s in ('to_q','to_k','to_v','to_out.0'))
             or isinstance(m,nn.Conv2d) and m.groups==1]
    unet.add_adapter(LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],
        init_lora_weights='gaussian',target_modules=targets))
    unet.disable_gradient_checkpointing(); unet.eval()
    pipe.vae.requires_grad_(False).eval()
    reader=load_reader(run); perceptual=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    params=[p for p in unet.parameters() if p.requires_grad]
    write_json(out/'adapter_audit.json',{'rank':c['teacher_rank'],'targets':targets,
        'trainable_parameters':sum(p.numel() for p in params),'scope':'attention and ungrouped convolution',
        'reader_trainable_parameters':sum(p.numel() for p in reader.parameters() if p.requires_grad)})
    optimizer=torch.optim.AdamW(params,lr=c['teacher_lr'],weight_decay=.01)
    scheduler=DDPMScheduler.from_pretrained(assets['student'],subfolder='scheduler',local_files_only=True)
    files=sorted(n for n in cache['files'] if n.startswith('teacher_train_'))
    probes=sorted(n for n in cache['files'] if n.startswith('meta_probe_'))
    rng=torch.Generator('cpu').manual_seed(c['seed']+219)
    applied=skipped=0; snapshots={}; surrogate=None
    with (out/'training.jsonl').open('w') as log:
        for step in range(1,max(c['teacher_checkpoints'])+1):
            optimizer.zero_grad(set_to_none=True); sums={}; meta_record={}
            factor=learning_rate_factor(step,max(c['teacher_checkpoints']),c['teacher_lr_floor'],
                                        c.get('teacher_warmup_steps',0))
            for group in optimizer.param_groups: group['lr']=c['teacher_lr']*factor
            finite=True
            for micro in range(c['teacher_accumulation']):
                data=load_file(str(run/'teacher_cache'/files[int(torch.randint(len(files),(1,),generator=rng))]))
                text=data['text'].cuda(); noise=torch.randn(data['latent'].shape,generator=rng).cuda()
                unet.disable_adapters()
                try:
                    with torch.no_grad():
                        clean=decode(pipe.vae,trajectory(unet,pipe.scheduler.config,noise,text,c['inference_steps'],c['teacher_tail_steps'],False),False)
                finally: unet.enable_adapters()
                marked=decode(pipe.vae,trajectory(unet,pipe.scheduler.config,noise,text,c['inference_steps'],c['teacher_tail_steps'],True),True)
                wm,terms=detection_loss(reader,marked,detector,c['score_margin'],c.get('native_bit_tail_weight',0.))
                mse=(marked-clean).square().mean(); lp=perceptual(marked*2-1,clean*2-1).mean()
                pr=mse/c['teacher_pixel_budget']; lr=lp/c['teacher_lpips_budget']
                quality=.005*(pr+lr)+.2*F.relu(pr-1).square()+.6*F.relu(lr-1).square()
                loss=wm+quality
                terms.update(watermark_loss=wm,quality_loss=quality,mse=mse,lpips=lp,
                             pixel_budget_exceeded=(pr>1).float(),lpips_budget_exceeded=(lr>1).float())
                if c['native_cycle_weight'] and step%4==0:
                    def enc(x):
                        dist=pipe.vae.encode(x*2-1).latent_dist
                        return dist.mean,dist.std
                    mean,std=checkpoint(enc,marked,use_reentrant=False)
                    eps=torch.randn(mean.shape,generator=rng).to(mean)
                    cycle=decode(pipe.vae,(mean+std*eps)*pipe.vae.config.scaling_factor,True)
                    cycle_loss,_=detection_loss(reader,cycle,detector,c['score_margin'],c.get('native_bit_tail_weight',0.))
                    loss=loss+c['native_cycle_weight']*cycle_loss; terms['cycle']=cycle_loss
                if c['meta_weight'] and step>c['meta_warmup'] and step%c['meta_every']==0 and micro==0:
                    from new_watermark.meta import make_surrogate,transfer_loss
                    if surrogate is None: surrogate=make_surrogate(assets['student'],c['meta_rank'],c['student_scope'])
                    probe=load_file(str(run/'teacher_cache'/probes[int(torch.randint(len(probes),(1,),generator=rng))]))
                    score_fn=lambda pixels:keyed_scores(reader_features(reader,pixels),detector['keys'])[:,0]
                    transfer,meta_record=transfer_loss(surrogate,pipe.vae,None,None,marked,clean,text[1:],
                        probe['latent'].cuda(),probe['text'].cuda(),scheduler,seed=c['seed']*100000+step,
                        steps=c['meta_steps'],lr=c['learning_rate'],margin=c['transfer_margin'],
                        target=detector['thresholds'][0]+c['score_margin'],score_fn=score_fn)
                    loss=loss+c['teacher_accumulation']*c['meta_weight']*transfer; terms['transfer']=transfer
                if not torch.isfinite(loss): finite=False; break
                terms['weighted_total_loss']=loss.detach()
                (loss/c['teacher_accumulation']).backward()
                for k,v in terms.items(): sums[k]=sums.get(k,0.)+float(v.detach())/c['teacher_accumulation']
                del marked,clean,loss,terms
                # Backward separately to avoid retaining a sampling graph beside this UNet graph.
                if c.get('native_prior_weight',0.):
                    latent=data['latent'].cuda()
                    prior_noise=torch.randn(latent.shape,generator=rng).to(latent)
                    prior_t=torch.randint(scheduler.config.num_train_timesteps,(1,),generator=rng).cuda()
                    pl,pt=prior_loss(unet,scheduler,latent,text,prior_noise,prior_t,c['native_prior_budget'])
                    if not torch.isfinite(pl): finite=False; break
                    (c['native_prior_weight']*pl/c['teacher_accumulation']).backward()
                    pt['weighted_total_loss']=c['native_prior_weight']*pl.detach()
                    pt['prior_timestep']=prior_t.float().mean()
                    for k,v in pt.items(): sums[k]=sums.get(k,0.)+float(v.detach())/c['teacher_accumulation']
                    del pl,pt
            norm=torch.nn.utils.clip_grad_norm_(params,1.) if finite else torch.tensor(float('nan'))
            finite=finite and bool(torch.isfinite(norm))
            if finite:
                optimizer.step(); applied+=1
                if not all(torch.isfinite(p).all() for p in params): raise RuntimeError('Corrupt native weights')
            else: skipped+=1
            record={'step':step,'applied':applied,'skipped':skipped,'lr':optimizer.param_groups[0]['lr'],
                'gradient_norm':float(norm) if finite else None,'terms':sums,'meta':meta_record,
                'gradient_clip_scale':min(1.,1./(float(norm)+1e-6)) if finite else None,
                'peak_gib':torch.cuda.max_memory_allocated()/1024**3}
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if step==1 or step%10==0: print('Native teacher',record,flush=True)
            if step in c['teacher_checkpoints']:
                folder=out/f'snapshot-{step}'; save_adapter(unet,folder)
                write_json(folder/'optimization_audit.json',{'attempted':step,'applied':applied,'skipped':skipped})
                snapshots[str(step)]=checkpoint_hashes(folder)
    freeze(out/'complete.json',{'snapshots':snapshots,'attempted':step,'applied':applied,'skipped':skipped})
    from new_watermark.training_audit import audit_training
    write_json(run/'result/teacher_training_audit.json',audit_training(out/'training.jsonl',c))

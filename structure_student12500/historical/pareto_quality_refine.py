"""Separately balanced quality + baseline score floor, with Adam-step surgery."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.checkpoint import checkpoint
import transfer_refine as engine
from common import read_json, write_json, freeze, sha256, image_name
from semantic_teacher_train import threshold_teacher_backward, checked_teacher_step, save_adapter
from semantic_checkpoint import latest_full_checkpoint, restore_training_state, save_training_state, align_loss_log
from transfer_proxy import decode, rgb8
from pareto_step import ProtectedAdamW, norm, dot


def sample_tail(unet, scheduler, noise, hidden, steps=50, backward_calls=10, checkpointed=True):
    """Same native forward trajectory; only last K denoiser calls have a graph.

    PNDM history is created in no_grad in the prefix. No stale cached prefix,
    no sampler reset at the boundary, no shortened forward generation.
    """
    scheduler.set_timesteps(steps, device=noise.device)
    if not 1 <= backward_calls <= len(scheduler.timesteps):
        raise ValueError('Invalid number of backward calls')
    cut = len(scheduler.timesteps) - backward_calls
    z = noise.float() * scheduler.init_noise_sigma
    def prediction(x, t, text):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            return unet(x, t, encoder_hidden_states=text).sample
    for i, t in enumerate(scheduler.timesteps):
        with torch.set_grad_enabled(i >= cut):
            x = scheduler.scale_model_input(torch.cat([z, z]), t)
            result = (checkpoint(prediction, x, t, hidden, use_reentrant=False)
                      if i >= cut and checkpointed else prediction(x, t, hidden))
            u, c = result.float().chunk(2)
            z = scheduler.step(u + 7.5 * (c - u), t, z).prev_sample
    return z


def dense(grads, params):
    return [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params)]


def balanced(gradient, reference_norm, fraction, max_gain):
    length = norm(gradient)
    if not torch.isfinite(torch.tensor(length)):
        raise RuntimeError('Nonfinite separate gradient')
    scale = min(max_gain, fraction * reference_norm / max(length, 1e-12)) if length > 1e-12 else 0.
    return scale, {'raw_norm': length, 'coefficient': scale, 'added_norm': length * scale,
                   'target_fraction': fraction, 'max_gain': max_gain}


class ParetoRefinement(engine.Refinement):
    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from diffusers import PNDMScheduler
        from semantic_teacher import pixels
        self.optimizer.constraints = []
        with torch.no_grad():
            clean = pixels([self.run/'images/teacher_clean_train'/image_name(i) for i in ids])
            z = self.ext.image_latent(clean)
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(z, noise.cuda(), t.cuda())
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
                                noisy, text, t.cuda(), 1., self.c, step, backward, {})
        if not force_proxy and (step + 1) % self.spec['native_every']:
            return {**values, 'native_active': False}
        original = dense([p.grad for p in self.params], self.params)
        # Clone: parameter .grad is modified below and then clipped by the
        # checked step. Protected objectives must retain their own gradients.
        original = [g.clone() for g in original]
        base_norm = norm(original)
        j = (step // self.spec['native_every']) % len(ids)
        index = ids[j]
        seed = self.c['seed'] * 1000000 + self.rows[index]['image_id']
        gen = torch.Generator('cuda').manual_seed(seed)
        initial = torch.randn((1,4,64,64), generator=gen, device='cuda', dtype=torch.float16)
        hidden = self.ext.text(['', self.rows[index]['text']]).half()
        scheduler = PNDMScheduler.from_pretrained(self.assets['teacher_clean'], subfolder='scheduler', local_files_only=True)
        latent = sample_tail(self.unet, scheduler, initial, hidden, self.c['inference_steps'], self.spec['backward_calls'])
        marked = rgb8(decode(self.ext.vae, latent.float()))
        lpips = self.perceptual(marked, clean[j:j+1]).mean()
        pixel = (marked-clean[j:j+1]).square().mean()
        quality = lpips + self.spec['pixel_weight'] * pixel
        def score(rgb):
            return self.carrier.scalar_score(self.ext.encode_latent(self.ext.image_latent(rgb), text_hidden=text[j:j+1])).mean()
        with torch.no_grad():
            reference = pixels([self.run/'images/baseline_wm_train'/image_name(index)])
            reference_score = score(reference)
            target = max(float(reference_score), values['target_score'])
        native_score = score(marked)
        quality_grad = dense(torch.autograd.grad(quality, self.params, retain_graph=True, allow_unused=True), self.params)
        score_grad = dense(torch.autograd.grad(-native_score, self.params, allow_unused=True), self.params)
        qscale, qstats = balanced(quality_grad, base_norm, self.spec['quality_fraction'], self.spec['max_gradient_gain'])
        # Hinge repair only when below source k=1 native score / calibrated
        # margin. Do not reward arbitrarily larger detector scores.
        deficit = max(0., target - float(native_score.detach()))
        sscale, sstats = balanced(score_grad, base_norm, self.spec['score_fraction'], self.spec['max_gradient_gain'])
        if deficit == 0:
            sscale = 0.; sstats['added_norm'] = 0.; sstats['coefficient'] = 0.
        with torch.no_grad():
            for p, b, qg, sg in zip(self.params, original, quality_grad, score_grad):
                p.grad = b + qscale * qg + sscale * sg
        self.optimizer.constraints = [original, score_grad, quality_grad]
        values.update(native_active=True, original_gradient_norm=base_norm,
            quality_lpips=float(lpips.detach()), native_pixel=float(pixel.detach()),
            native_quality=float(quality.detach()), native_score=float(native_score.detach()),
            baseline_score=float(reference_score), score_target=target, score_deficit=deficit,
            quality_gradient=qstats, score_gradient=sstats,
            quality_original_cosine=float(dot(quality_grad,original))/max(norm(quality_grad)*base_norm,1e-20),
            quality_score_cosine=float(dot(quality_grad,score_grad))/max(norm(quality_grad)*norm(score_grad),1e-20),
            native_training_index=index, native_seed=seed,
            native_denoiser_calls=len(scheduler.timesteps), backward_calls=self.spec['backward_calls'],
            loss_note='loss is original objective; separate auxiliary objectives and gradients logged individually')
        if force_proxy and (qstats['added_norm'] <= 0 or norm(score_grad) <= 0):
            raise RuntimeError('Missing quality or native score gradient')
        return values


def train(run, smoke=False):
    spec=read_json(run/'refine_spec.json');c=read_json(run/'config.json')
    folder=run/'teacher_fp/refined';folder.mkdir(parents=True,exist_ok=True)
    protocol={name:sha256(path) for name,path in {
        'config':run/'config.json','spec':run/'refine_spec.json','lineage':run/'teacher_choice.json',
        'carrier':run/'carrier/complete.json','code':run.parent/'code_manifest.json',
        'baseline_images':run/'images/baseline_wm_train/image_hashes.json'}.items()}
    protocol.update(optimizer='AdamW proposal; exact <=3-halfspace projection of displacement on auxiliary steps' if spec['project_step'] else 'AdamW unprojected matched ablation',
        gradients='original + separately normalized quality + hinge score repair; last10 native-call derivatives',
        teacher_selection='fixed final600', scientific_stopping_gates=False)
    freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists() and not smoke:
        for n,d in read_json(folder/'trained.json')['files'].items():
            if sha256(folder/n)!=d:raise RuntimeError('Completed teacher changed')
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=ParetoRefinement(run,spec)
    optimizer=ProtectedAdamW(model.params,project=spec['project_step'],lr=spec['teacher_lr'])
    model.optimizer=optimizer
    rng=torch.Generator('cpu').manual_seed(c['seed']);order=[];offset=0;start=0
    for label in ['teacher_clean_train','baseline_wm_train']:
        directory=run/'images'/label;entries=read_json(directory/'image_hashes.json')
        if len(entries)!=len(model.rows) or any(sha256(directory/v['file_name'])!=v['sha256'] for v in entries):
            raise RuntimeError('Training image integrity failure: '+label)
    saved=latest_full_checkpoint(folder) if not smoke else None
    if saved:
        state=restore_training_state(saved,model.unet,optimizer,None,rng,protocol)
        start,order,offset=state['step'],state['order'],state['offset']
        align_loss_log(folder/'losses.jsonl',start,expected_hash=state['loss_prefix_sha256'])
    elif not smoke:align_loss_log(folder/'losses.jsonl',0)
    if smoke:
        before=[p.detach().clone() for p in model.params];torch.cuda.reset_peak_memory_stats()
        values=checked_teacher_step(model.named,optimizer,None,
            lambda backward:model.compute(list(range(c['teacher_batch'])),
                torch.randn((c['teacher_batch'],4,64,64),generator=rng),
                torch.full((c['teacher_batch'],),100),0,backward,force_proxy=True),
            run/'smoke_failure.json',{'phase':'pareto_actual_model_smoke'})
        if not any(not torch.equal(a,b) for a,b in zip(before,model.params)):
            raise RuntimeError('Smoke produced no parameter update')
        write_json(run/'smoke.json',{'passed':True,'scientific_result':False,'checkpoint_written':False,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**values,'optimizer_step':optimizer.last_projection})
        print('SMOKE PASSED',flush=True);return
    with (folder/'losses.jsonl').open('a') as log:
        for step in range(start,spec['teacher_steps']):
            if offset+c['teacher_batch']>len(order):order=torch.randperm(len(model.rows),generator=rng).tolist();offset=0
            ids=order[offset:offset+c['teacher_batch']];offset+=len(ids)
            noise=torch.randn((len(ids),4,64,64),generator=rng)
            t=torch.randint(0,model.scheduler.config.num_train_timesteps,(len(ids),),generator=rng)
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            values=checked_teacher_step(model.named,optimizer,None,
                lambda backward:model.compute(ids,noise,t,step,backward),folder/'failure.json',{'step':step+1,'image_indices':ids})
            record={'step':step+1,'seconds':time.monotonic()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**values,'optimizer_step':optimizer.last_projection}
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if (step+1)%10==0:print(json.dumps(record),flush=True)
            if (step+1)%spec['checkpoint_every']==0 or step+1==spec['teacher_steps']:
                dest=folder/f'checkpoint-{step+1}';save_adapter(model.unet,dest,model.lora_config)
                save_training_state(dest,model.unet,optimizer,None,rng,order,offset,step+1,protocol)
    save_adapter(model.unet,folder,model.lora_config)
    freeze(folder/'trained.json',{'steps':spec['teacher_steps'],'files':{n:sha256(folder/n)
        for n in ['adapter_model.safetensors','adapter_config.json','losses.jsonl']}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();train(args.run,args.smoke)

"""Batch native quality + structural preservation + deficit-weighted repair.

Only teacher training changes. Original threshold_cfg_v1 and downstream
ordinary student remain unchanged. Warm start, not resume, of baseline k1.
"""
import argparse
import json
from pathlib import Path
import time
import torch
import transfer_refine as engine
from common import read_json, write_json, freeze, sha256, image_name
from semantic_teacher_train import threshold_teacher_backward, checked_teacher_step, save_adapter
from semantic_checkpoint import latest_full_checkpoint, restore_training_state, save_training_state, align_loss_log
from transfer_proxy import decode, rgb8
from pareto_step import ProtectedAdamW, norm, dot
from pareto_quality_refine import sample_tail, dense, balanced
from structure_margin_objectives import ms_ssim, margin_penalty


class StructureMarginRefinement(engine.Refinement):
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
        original = [g.clone() for g in dense([p.grad for p in self.params], self.params)]
        base_norm = norm(original)
        qgrad = [torch.zeros_like(p) for p in self.params]
        mgrad = [torch.zeros_like(p) for p in self.params]
        records = []
        sigma = self.c['teacher_calibration']['sigma_clean']
        if len(ids) != self.spec['native_batch']:
            raise RuntimeError('Native batch must match original batch, without dropping examples')
        micro = self.spec['native_micro_batch']
        for offset in range(0, len(ids), micro):
            picked = ids[offset:offset+micro]
            n = len(picked)
            prompts = [self.rows[i]['text'] for i in picked]
            seeds = [self.c['seed']*1000000 + self.rows[i]['image_id'] for i in picked]
            initial = torch.cat([torch.randn((1,4,64,64), generator=torch.Generator('cuda').manual_seed(seed),
                                  device='cuda', dtype=torch.float16) for seed in seeds])
            hidden = self.ext.text(['']*n + prompts).half()
            scheduler = PNDMScheduler.from_pretrained(self.assets['teacher_clean'], subfolder='scheduler', local_files_only=True)
            latent = sample_tail(self.unet, scheduler, initial, hidden, self.c['inference_steps'], self.spec['backward_calls'])
            marked = rgb8(decode(self.ext.vae, latent.float()))
            paired = clean[offset:offset+n]
            perceptual = self.perceptual(marked, paired).reshape(n, -1).mean(1)
            pixel = (marked-paired).square().mean((1,2,3))
            structural = 1-ms_ssim((marked+1)/2, (paired+1)/2)
            quality = perceptual + self.spec['pixel_weight']*pixel + self.spec['structure_weight']*structural
            def scores(rgb):
                features = self.ext.encode_latent(self.ext.image_latent(rgb), text_hidden=text[offset:offset+n])
                return self.carrier.scalar_score(features)
            with torch.no_grad():
                reference = pixels([self.run/'images/baseline_wm_train'/image_name(i) for i in picked])
                reference_score = scores(reference)
                targets = reference_score.clamp_min(values['target_score'])
            native_score = scores(marked)
            penalty = margin_penalty(native_score, targets, sigma)
            # Sum/N over microbatches is the exact batch objective gradient;
            # normalization is applied once after accumulation, never per sample.
            qg = dense(torch.autograd.grad(quality.sum()/len(ids), self.params, retain_graph=True, allow_unused=True), self.params)
            mg = dense(torch.autograd.grad(penalty.sum()/len(ids), self.params, allow_unused=True), self.params)
            with torch.no_grad():
                for a, b, c, d in zip(qgrad, mgrad, qg, mg):
                    a.add_(c); b.add_(d)
            for j, index in enumerate(picked):
                records.append({'index':index,'seed':seeds[j], 'score':float(native_score[j].detach()),
                    'reference_score':float(reference_score[j]),'target':float(targets[j]),
                    'deficit':float((targets[j]-native_score[j]).clamp_min(0).detach()),
                    'lpips':float(perceptual[j].detach()), 'pixel':float(pixel[j].detach()),
                    'structure_distance':float(structural[j].detach()), 'margin_loss':float(penalty[j].detach())})
            del latent, marked, native_score, quality, penalty, qg, mg, perceptual, pixel, structural
        qscale, qstats = balanced(qgrad, base_norm, self.spec['quality_fraction'], self.spec['max_gradient_gain'])
        mscale, mstats = balanced(mgrad, base_norm, self.spec['score_fraction'], self.spec['max_gradient_gain'])
        with torch.no_grad():
            for p, b, q, m in zip(self.params, original, qgrad, mgrad):
                p.grad = b + qscale*q + mscale*m
        # Aggregate shortfall constraint replaces unconditional negative-score
        # constraint. No gradient rewards scores already above their floor.
        self.optimizer.constraints = [original, mgrad, qgrad]
        self.optimizer.constraint_names = ['original_loss','batch_squared_margin_deficit','native_quality']
        mean = lambda key: sum(r[key] for r in records)/len(records)
        values.update(native_active=True, native_samples=records, original_gradient_norm=base_norm,
            quality_gradient=qstats, score_gradient=mstats, native_score=mean('score'),
            baseline_score=mean('reference_score'), score_deficit=mean('deficit'),
            score_deficit_max=max(r['deficit'] for r in records),
            deficient_images=sum(r['deficit']>0 for r in records),
            quality_lpips=mean('lpips'), structure_distance=mean('structure_distance'),
            margin_loss=mean('margin_loss'), native_batch=len(records), native_micro_batch=micro,
            native_denoiser_calls=len(scheduler.timesteps), backward_calls=self.spec['backward_calls'],
            quality_original_cosine=float(dot(qgrad,original))/max(norm(qgrad)*base_norm,1e-20),
            quality_margin_cosine=float(dot(qgrad,mgrad))/max(norm(qgrad)*norm(mgrad),1e-20),
            loss_note='loss is original objective; auxiliary losses/gradients reported separately')
        if force_proxy and qstats['added_norm'] <= 0:
            raise RuntimeError('Missing native quality gradient')
        if values['deficient_images'] and mstats['added_norm'] <= 0:
            raise RuntimeError('Missing margin gradient despite deficient images')
        return values


class MarginAdamW(ProtectedAdamW):
    def step(self, closure=None):
        result = super().step(closure)
        if self.last_projection.get('auxiliary_step'):
            self.last_projection['constraint_order'] = self.constraint_names
        return result


def train(run, smoke=False):
    spec=read_json(run/'refine_spec.json');c=read_json(run/'config.json')
    folder=run/'teacher_fp/refined';folder.mkdir(parents=True,exist_ok=True)
    protocol={name:sha256(path) for name,path in {
        'config':run/'config.json','spec':run/'refine_spec.json','lineage':run/'teacher_choice.json',
        'carrier':run/'carrier/complete.json','code':run.parent/'code_manifest.json',
        'baseline_images':run/'images/baseline_wm_train/image_hashes.json'}.items()}
    protocol.update(optimizer='AdamW proposal; exact <=3-halfspace projection of displacement on auxiliary steps' if spec['project_step'] else 'AdamW unprojected matched ablation',
        gradients='original + batch4 native quality + squared per-image margin deficits; last10 derivatives',
        teacher_selection='fixed final600', scientific_stopping_gates=False)
    freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists() and not smoke:
        for n,d in read_json(folder/'trained.json')['files'].items():
            if sha256(folder/n)!=d:raise RuntimeError('Completed teacher changed')
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=StructureMarginRefinement(run,spec)
    optimizer=MarginAdamW(model.params,project=spec['project_step'],lr=spec['teacher_lr'])
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
            run/'smoke_failure.json',{'phase':'structure_margin_actual_model_smoke'})
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

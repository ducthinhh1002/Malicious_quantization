"""Projected-native continuation with a frozen SD1.4 training-channel surrogate."""
import bootstrap
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
import transfer_refine as engine
from common import read_json, image_name
from semantic_teacher_train import threshold_teacher_backward, predicted_x0
from transfer_proxy import decode, encode, rgb8
from pareto_quality_refine import sample_tail, dense, balanced
from pareto_step import norm, dot
from objectives import robust_margin, bit_margin
from structure_metrics import ms_ssim


class TransferRefinement(engine.Refinement):
    def __init__(self, run, spec):
        super().__init__(run, spec)
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from student import load_adapter
        # Fixed owner-side surrogate trained only on historical training images.
        # Its weights are immutable; no gradients/keys are supplied to the final student.
        self.channel = StableDiffusionPipeline.from_pretrained(self.assets['student'],
            torch_dtype=torch.float32, safety_checker=None, local_files_only=True).to('cuda')
        proxy = read_json(run / 'channel_proxy.json')
        for name, digest in proxy['files'].items():
            if bootstrap.sha256(Path(proxy['directory']) / name) != digest:
                raise RuntimeError('Frozen channel proxy changed')
        load_adapter(self.channel.unet, Path(proxy['directory']))
        for module in (self.channel.vae, self.channel.text_encoder, self.channel.unet):
            module.eval().requires_grad_(False)
        self.channel_scheduler = DDPMScheduler.from_pretrained(self.assets['student'],
            subfolder='scheduler', local_files_only=True)

    def channel_text(self, prompts):
        tokens = self.channel.tokenizer(prompts, padding='max_length', truncation=True,
            max_length=self.channel.tokenizer.model_max_length, return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            return self.channel.text_encoder(tokens)[0]

    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from diffusers import PNDMScheduler
        from semantic_teacher import pixels
        self.optimizer.constraints = []
        with torch.no_grad():
            clean = pixels([self.run/'images/teacher_clean_train'/image_name(i) for i in ids])
            latent = self.ext.image_latent(clean)
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(latent, noise.cuda(), t.cuda())
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
            noisy, text, t.cuda(), 1., self.c, step, backward, {})
        if not force_proxy and (step + 1) % self.spec['native_every']:
            return {**values, 'native_active': False}
        original = [g.clone() for g in dense([p.grad for p in self.params], self.params)]
        base_norm = norm(original)
        qgrad = [torch.zeros_like(p) for p in self.params]
        mgrad = [torch.zeros_like(p) for p in self.params]
        sigma = self.c['teacher_calibration']['sigma_clean']
        tau = max(self.c['teacher_calibration']['threshold'],
                  read_json(self.run/'result/threshold_student.json')['threshold'])
        ramp = min(1., (step + 1) / self.spec['margin_ramp_steps'])
        floor = tau + (1. + ramp) * sigma
        denoise_floor = tau + (.5 + .5*ramp) * sigma
        records = []
        # Microbatch one limits simultaneous native, VAE, surrogate and reader graphs.
        for j, index in enumerate(ids):
            prompt = self.rows[index]['text']
            seed = self.c['seed']*1000000 + self.rows[index]['image_id']
            initial = torch.randn((1,4,64,64), generator=torch.Generator('cuda').manual_seed(seed),
                                  device='cuda', dtype=torch.float16)
            hidden = self.ext.text(['', prompt]).half()
            sampler = PNDMScheduler.from_pretrained(self.assets['teacher_clean'],
                subfolder='scheduler', local_files_only=True)
            z = sample_tail(self.unet, sampler, initial, hidden, self.c['inference_steps'],
                            self.spec['backward_calls'])
            marked = rgb8(decode(self.ext.vae, z.float()))
            paired = clean[j:j+1]
            perceptual = self.perceptual(marked, paired).mean()
            pixel = (marked-paired).square().mean()
            structural = (1-ms_ssim((marked+1)/2, (paired+1)/2)).mean()
            quality = perceptual + self.spec['pixel_weight']*pixel + self.spec['structure_weight']*structural
            def coordinates(rgb):
                features = self.ext.encode_latent(self.ext.image_latent(rgb), text_hidden=text[j:j+1])
                return self.carrier.coordinates(features)
            def scalar(coords):
                return (coords*self.carrier.signs).mean(-1)
            with torch.no_grad():
                ref = pixels([self.run/'images/baseline_wm_train'/image_name(index)])
                reference_score = scalar(coordinates(ref))
            raw_coordinates = coordinates(marked)
            rng = torch.Generator('cuda').manual_seed(31000000 + step*10007 + index)
            posterior_noise = torch.randn((1,4,64,64), generator=rng, device='cuda')
            student_z = encode(self.channel.vae, marked, posterior_noise)
            cycled = rgb8(decode(self.channel.vae, student_z))
            cycle_coordinates = coordinates(cycled)
            tt = torch.tensor([self.spec['channel_timesteps'][(step//self.spec['native_every']+j)
                % len(self.spec['channel_timesteps'])]], device='cuda')
            eps = torch.randn(student_z.shape, generator=rng, device='cuda')
            noised = self.channel_scheduler.add_noise(student_z, eps, tt)
            channel_text = self.channel_text(['', prompt])
            def predict(x):
                # BF16 denoiser, FP32 CFG arithmetic/x0/VAE/reader.
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = self.channel.unet(torch.cat([x,x]), tt,
                        encoder_hidden_states=channel_text).sample
                unconditional, conditional = result.float().chunk(2)
                return unconditional + 7.5*(conditional-unconditional)
            prediction = checkpoint(predict, noised, use_reentrant=False)
            x0 = predicted_x0(self.channel_scheduler, noised, prediction, tt)
            denoised = rgb8(decode(self.channel.vae, x0))
            channel_coordinates = coordinates(denoised)
            scores = torch.stack([scalar(raw_coordinates), scalar(cycle_coordinates),
                                  scalar(channel_coordinates)], dim=1)
            targets = scores.new_tensor([[max(float(reference_score),floor), floor, denoise_floor]])
            margin = robust_margin(scores, targets, sigma).mean()
            bits = torch.stack([bit_margin(c,self.carrier.signs).mean()
                for c in [raw_coordinates,cycle_coordinates,channel_coordinates]]).mean()
            ownership = margin + self.spec['bit_margin_weight']*bits
            qg = dense(torch.autograd.grad(quality/len(ids), self.params,
                retain_graph=True, allow_unused=True), self.params)
            mg = dense(torch.autograd.grad(ownership/len(ids), self.params,
                allow_unused=True), self.params)
            with torch.no_grad():
                for accq, accm, gq, gm in zip(qgrad,mgrad,qg,mg):
                    accq.add_(gq); accm.add_(gm)
            records.append(dict(index=index, seed=seed, timestep=int(tt),
                native_score=float(scores[0,0].detach()), cycle_score=float(scores[0,1].detach()),
                channel_score=float(scores[0,2].detach()), targets=targets[0].tolist(),
                reference_score=float(reference_score), lpips=float(perceptual.detach()),
                structure=float(structural.detach()), margin=float(margin.detach()),
                bit_penalty=float(bits.detach())))
            del marked,cycled,denoised,noised,z,student_z,x0,prediction,scores,ownership,quality,qg,mg
        qscale,qstats = balanced(qgrad,base_norm,self.spec['quality_fraction'],self.spec['max_gradient_gain'])
        mscale,mstats = balanced(mgrad,base_norm,self.spec['score_fraction'],self.spec['max_gradient_gain'])
        with torch.no_grad():
            for p,g,q,m in zip(self.params,original,qgrad,mgrad):
                p.grad = g + qscale*q + mscale*m
        self.optimizer.constraints = [original,mgrad,qgrad]
        self.optimizer.constraint_names = ['original_loss','finite_view_ownership_margin','native_quality']
        values.update(native_active=True, native_samples=records, margin_ramp=ramp,
            original_gradient_norm=base_norm, quality_gradient=qstats, score_gradient=mstats,
            quality_lpips=sum(r['lpips'] for r in records)/len(records),
            native_score=sum(r['native_score'] for r in records)/len(records),
            quality_margin_cosine=float(dot(qgrad,mgrad))/max(norm(qgrad)*norm(mgrad),1e-20),
            channel_proxy_frozen=True, final_student_used=False,
            loss_note='original loss plus separately balanced native quality and three-view margin gradients')
        if force_proxy:
            if qstats['added_norm'] <= 0 or mstats['added_norm'] <= 0:
                raise RuntimeError('Missing actual teacher quality/channel gradient')
            if any(p.grad is not None for p in self.channel.unet.parameters()):
                raise RuntimeError('Frozen proxy unexpectedly accumulated parameter gradients')
        return values

import argparse
import json
import time
from common import write_json, freeze, sha256
from semantic_teacher_train import checked_teacher_step, save_adapter
from semantic_checkpoint import latest_full_checkpoint, restore_training_state, save_training_state, align_loss_log
from pareto_step import ProtectedAdamW

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
        gradients='original + native structural quality + raw/SD1.4-VAE/frozen-proxy margin; last10 native derivatives',
        teacher_selection='fixed final step from frozen spec', scientific_stopping_gates=False)
    freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists() and not smoke:
        for n,d in read_json(folder/'trained.json')['files'].items():
            if sha256(folder/n)!=d:raise RuntimeError('Completed teacher changed')
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=TransferRefinement(run,spec)
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
            run/'smoke_failure.json',{'phase':'transfer90_actual_model_smoke'})
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

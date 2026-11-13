"""Full Structure warm start; replace native scalar repair with channel bit repair."""
import argparse
import json
from pathlib import Path
import time

import bootstrap
from bootstrap import ROOT, read, write, freeze, sha, now
import torch
import transfer_refine as engine
from common import image_name
from semantic_teacher_train import threshold_teacher_backward, checked_teacher_step, save_adapter
from semantic_checkpoint import latest_full_checkpoint, restore_training_state, save_training_state, align_loss_log
from transfer_proxy import decode, rgb8
from pareto_quality_refine import sample_tail, dense, balanced
from structure_margin_objectives import ms_ssim
from pareto_step import ProtectedAdamW, norm
from jpeg_channel import CHANNELS, transform, bit_margin, jpeg_quality
from validation import monitor


class JPEGRefinement(engine.Refinement):
    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from diffusers import PNDMScheduler
        from semantic_teacher import pixels
        self.optimizer.constraints = []
        with torch.no_grad():
            clean = pixels([self.run / 'images/teacher_clean_train' / image_name(i) for i in ids])
            z = self.ext.image_latent(clean)
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(z, noise.cuda(), t.cuda())
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
                                           noisy, text, t.cuda(), 1., self.c, step, backward, {})
        if not force_proxy and (step + 1) % self.spec['native_every']:
            return {**values, 'native_active': False}
        original = [g.clone() for g in dense([p.grad for p in self.params], self.params)]
        base_norm = norm(original)
        if base_norm <= 1e-12:
            raise RuntimeError('Missing base gradient for auxiliary normalization')
        qgrad = [torch.zeros_like(p) for p in self.params]
        mgrad = [torch.zeros_like(p) for p in self.params]
        records = []
        severity = 1. if force_proxy else min(1., .25 + .75 * (step + 1) / self.spec['curriculum_updates'])
        native_index = step // self.spec['native_every']
        native_ids = ids[:self.spec['native_batch']]
        for slot, index in enumerate(native_ids):
            row = self.rows[index]
            seed = self.c['seed'] * 1000000 + row['image_id']
            initial = torch.randn((1, 4, 64, 64), generator=torch.Generator('cuda').manual_seed(seed),
                                  device='cuda', dtype=torch.float32)
            hidden = self.ext.text(['', row['text']]).float()
            scheduler = PNDMScheduler.from_pretrained(self.assets['teacher_clean'], subfolder='scheduler', local_files_only=True)
            latent = sample_tail(self.unet, scheduler, initial, hidden,
                                 self.c['inference_steps'], self.spec['backward_calls'])
            marked = rgb8(decode(self.ext.vae, latent.float()))
            paired = clean[slot:slot + 1]
            perceptual = self.perceptual(marked, paired).mean()
            pixel = (marked - paired).square().mean()
            structure = 1 - ms_ssim((marked + 1) / 2, (paired + 1) / 2).mean()
            quality = perceptual + self.spec['pixel_weight'] * pixel + self.spec['structure_weight'] * structure
            qg = dense(torch.autograd.grad(quality / len(native_ids), self.params,
                                         retain_graph=True, allow_unused=True), self.params)
            with torch.no_grad():
                for target, value in zip(qgrad, qg):
                    target.add_(value)
            del qg, quality
            name = CHANNELS[(native_index * len(ids) + slot) % len(CHANNELS)]
            attack_seed = self.c['seed'] + 3000000 + step * 1009 + index
            views = []
            # Identity and one balanced sampled attack are two equal draws in
            # ONE per-bit objective. Each microbatch has its exact 1/N weight.
            for v, channel in enumerate(('identity', name)):
                changed = transform(marked, channel, attack_seed, severity)
                features = self.ext.encode_latent(self.ext.image_latent(changed), text_hidden=text[slot:slot + 1])
                q = self.carrier.coordinates(features)
                repair = bit_margin(q, self.carrier.signs, self.spec['mu']).mean()
                mg = dense(torch.autograd.grad(repair / (2 * len(native_ids)), self.params,
                                             retain_graph=(v == 0), allow_unused=True), self.params)
                with torch.no_grad():
                    for target, value in zip(mgrad, mg):
                        target.add_(value)
                    signed = (q * self.carrier.signs).squeeze(0)
                    views.append({'channel': channel, 'loss': float(repair.detach()),
                                  'signed_coordinates': signed.tolist(),
                                  'bit_accuracy': float((signed > 0).float().mean()),
                                  'wrong_bits': [j + 1 for j, a in enumerate(signed.tolist()) if a <= 0]})
                del changed, features, q, repair, mg
            records.append({'training_index': index, 'seed': seed, 'attack_seed': attack_seed,
                            'severity': severity, 'jpeg_quality': jpeg_quality(attack_seed, severity), 'views': views, 'lpips': float(perceptual.detach()),
                            'pixel': float(pixel.detach()), 'structure_distance': float(structure.detach())})
            del latent, marked, perceptual, pixel, structure
        qscale, qstats = balanced(qgrad, base_norm, self.spec['quality_fraction'], self.spec['max_gradient_gain'])
        mscale, mstats = balanced(mgrad, base_norm, self.spec['score_fraction'], self.spec['max_gradient_gain'])
        with torch.no_grad():
            for p, b, q, m in zip(self.params, original, qgrad, mgrad):
                p.grad = b + qscale * q + mscale * m
        self.optimizer.constraints = [original, mgrad, qgrad]
        views = [v for r in records for v in r['views']]
        per_bit = torch.tensor([v['signed_coordinates'] for v in views])
        values.update(native_active=True, native_samples=records,
                      original_gradient_norm=base_norm, quality_gradient=qstats, margin_gradient=mstats,
                      margin_loss=sum(v['loss'] for v in views) / len(views),
                      per_bit_accuracy=(per_bit > 0).float().mean(0).tolist(),
                      per_bit_signed_mean=per_bit.mean(0).tolist(),
                      quality_lpips=sum(r['lpips'] for r in records) / len(records),
                      structure_distance=sum(r['structure_distance'] for r in records) / len(records),
                      mu=self.spec['mu'], severity=severity,
                      loss_note='base loss only; native quality and unified per-bit channel loss logged separately')
        if force_proxy and (qstats['raw_norm'] <= 0 or mstats['raw_norm'] <= 0):
            raise RuntimeError('Missing actual-model quality or channel margin gradient')
        return values


def train(run, smoke=False):
    spec = read(run / 'refine_spec.json')
    c = read(run / 'config.json')
    folder = run / 'teacher_fp/refined'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {name: sha(run / rel) for name, rel in {
        'config': 'config.json', 'spec': 'refine_spec.json', 'lineage': 'teacher_choice.json',
        'carrier': 'carrier/complete.json', 'code': 'code_manifest.json',
        'train_images': 'images/teacher_clean_train/image_hashes.json'}.items()}
    protocol.update(method='all32-bit margin, native identity plus one real JPEG view',
                    optimizer='original protected AdamW displacement, three first-order constraints',
                    resume='full optimizer/RNG state within this run; weights-only warm start from historical Full Structure',
                    projection_limit='local truncated derivatives, not a nonlinear quality guarantee')
    freeze(folder / 'protocol.json', protocol)
    if (folder / 'trained.json').exists() and not smoke:
        return
    manifest = read(run / 'code_manifest.json')
    for path, expected in manifest.items():
        if sha(path) != expected:
            raise RuntimeError('Run code changed: ' + path)
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    choice = read(run / 'teacher_choice.json')
    if sha(Path(choice['source_checkpoint']) / 'adapter_model.safetensors') != choice['source_adapter_sha256']:
        raise RuntimeError('Warm-start adapter hash changed')
    model = JPEGRefinement(run, spec)
    optimizer = ProtectedAdamW(model.params, project=True, lr=spec['teacher_lr'])
    model.optimizer = optimizer
    rng = torch.Generator('cpu').manual_seed(c['seed'])
    order, offset, start = [], 0, 0
    saved = latest_full_checkpoint(folder) if not smoke else None
    if saved:
        state = restore_training_state(saved, model.unet, optimizer, None, rng, protocol)
        start, order, offset = state['step'], state['order'], state['offset']
        align_loss_log(folder / 'losses.jsonl', start, expected_hash=state['loss_prefix_sha256'])
    elif not smoke:
        align_loss_log(folder / 'losses.jsonl', 0)
    if not smoke and not read(run / 'smoke.json')['passed']:
        raise RuntimeError('Actual model smoke required')
    if not smoke and not (run/'validation'/f'step_{start:04d}.json').exists():
        monitor(model,start)
    total = 1 if smoke else spec['teacher_steps']
    before = [p.detach().clone() for p in model.params] if smoke else None
    with (folder / ('smoke_losses.jsonl' if smoke else 'losses.jsonl')).open('a') as log:
        for step in range(start, total):
            if offset + c['teacher_batch'] > len(order):
                order = torch.randperm(len(model.rows), generator=rng).tolist()
                offset = 0
            ids = order[offset:offset + c['teacher_batch']]
            offset += len(ids)
            noise = torch.randn((len(ids), 4, 64, 64), generator=rng)
            t = torch.randint(0, model.scheduler.config.num_train_timesteps, (len(ids),), generator=rng)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            values = checked_teacher_step(model.named, optimizer, None,
                lambda backward: model.compute(ids, noise, t, step, backward, force_proxy=smoke),
                folder / ('smoke_failure.json' if smoke else 'failure.json'), {'step': step + 1, 'ids': ids})
            record = {'step': step + 1, 'seconds': time.monotonic() - started,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                      **values, 'optimizer_step': optimizer.last_projection}
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            write(run / 'train_status.json', {'state': 'running', 'step': step + 1, 'total': total,
                                             'seconds': record['seconds'], 'updated': now(),
                                             'native_active': values['native_active'], 'mu': spec['mu']})
            print(json.dumps({k: record[k] for k in ['step', 'seconds', 'peak_allocated_gib', 'native_active']}) , flush=True)
            if not smoke and (step + 1) % 100 == 0:
                monitor(model,step+1)
            if smoke:
                if not any(not torch.equal(a, b) for a, b in zip(before, model.params)):
                    raise RuntimeError('Smoke did not update trainable parameters')
                write(run / 'smoke.json', {'passed': True, 'scientific_result': False,
                                          'checkpoint_written': False, **record})
            elif (step + 1) % spec['checkpoint_every'] == 0 or step + 1 == total:
                dest = folder / f'checkpoint-{step + 1}'
                save_adapter(model.unet, dest, model.lora_config)
                save_training_state(dest, model.unet, optimizer, None, rng, order, offset, step + 1, protocol)
    if not smoke:
        save_adapter(model.unet, folder, model.lora_config)
        freeze(folder / 'trained.json', {'steps': total, 'files': {n: sha(folder / n) for n in
                ['adapter_model.safetensors', 'adapter_config.json', 'losses.jsonl']}})
        write(run / 'train_status.json', {'state': 'completed', 'step': total, 'updated': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train(args.run, args.smoke)

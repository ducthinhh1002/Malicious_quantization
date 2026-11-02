"""Warm-start threshold_cfg_v1; bounded auxiliary perceptual / learner gradients."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Malicious_quantization/bitmark_lora'))
from common import read_json, write_json, freeze, sha256, image_name, verify_vendor
from semantic_teacher_train import (threshold_teacher_backward, predicted_x0,
                                    checked_teacher_step, save_adapter)
from semantic_threshold import cfg_prediction
from semantic_checkpoint import (latest_full_checkpoint, restore_training_state,
                                 save_training_state, align_loss_log)
from transfer_proxy import PairedLearner, decode


def norm_of(values):
    values = [v for v in values if v is not None]
    if not values:
        return torch.tensor(0., device='cuda' if torch.cuda.is_available() else 'cpu')
    return torch.stack([v.detach().double().square().sum() for v in values]).sum().sqrt()


def add_bounded_gradient(parameters, gradients, base_norm, fraction):
    norm = norm_of(gradients)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite auxiliary gradient')
    coefficient = min(1., fraction * float(base_norm) / (float(norm) + 1e-12))
    dot = sum(float((p.grad.detach().double() * g.detach().double()).sum())
              for p, g in zip(parameters, gradients) if g is not None and p.grad is not None)
    with torch.no_grad():
        for p, g in zip(parameters, gradients):
            if g is not None:
                if p.grad is None:
                    p.grad = g.detach() * coefficient
                else:
                    p.grad.add_(g.detach(), alpha=coefficient)
    return {'norm': float(norm), 'coefficient': coefficient, 'added_norm': float(norm) * coefficient,
            'dot_with_accumulated_gradient': dot}


class Refinement:
    def __init__(self, run, spec):
        import lpips
        from diffusers import DDPMScheduler
        from peft import LoraConfig
        from semantic_teacher import extractor, numerical_policy
        from semantic_carrier import SemanticCarrier
        from student import load_adapter
        verify_vendor(); numerical_policy()
        # Higher-order derivatives require math SDPA; fused attention backward is not double-differentiable.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        self.run, self.spec = run, spec
        self.c = read_json(run / 'config.json')
        self.c['teacher_calibration'] = read_json(run / 'teacher_fp/calibration_target.json')
        self.assets = read_json(run / 'assets.json')
        self.ext = extractor(run)
        self.unet = copy.deepcopy(self.ext.unet)
        source = Path(read_json(run / 'teacher_choice.json')['source_checkpoint'])
        load_adapter(self.unet, source)
        self.lora_config = LoraConfig.from_pretrained(str(source))
        for n, p in self.unet.named_parameters():
            p.requires_grad_('lora_' in n)
        self.unet.train().enable_gradient_checkpointing()
        self.named = [(n, p) for n, p in self.unet.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        self.scheduler = DDPMScheduler.from_pretrained(self.assets['teacher_clean'], subfolder='scheduler', local_files_only=True)
        self.carrier = SemanticCarrier.load(run / 'carrier', 'cuda')
        self.perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
        self.proxy = (PairedLearner(self.assets['student'], self.ext, self.carrier,
                        rank=spec['proxy_rank'], inner_steps=spec['inner_steps'], query_steps=spec['query_steps'])
                      if spec['transfer'] else None)
        self.rows = read_json(run / 'prompts.json')['train']

    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from semantic_teacher import pixels
        images = self.run / 'images/teacher_clean_train'
        with torch.no_grad():
            z = self.ext.image_latent(pixels([images / image_name(i) for i in ids]))
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(z, noise.to('cuda'), t.to('cuda'))
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
                         noisy, text, t.cuda(), 1., self.c, step, backward, {})
        base_norm = norm_of([p.grad for p in self.params])
        active = self.proxy is not None and (force_proxy or (step + 1) % self.spec['transfer_every'] == 0)
        g = torch.Generator('cuda').manual_seed(self.c['seed'] + 500000 + step)
        # A small independent low-noise view; keep the original objective's timestep distribution unchanged.
        j = step % len(ids)
        tt = torch.randint(10, 201, (1,), generator=g, device='cuda')
        zz = z[j:j+1]
        hidden = text[j:j+1]
        empty = self.ext.text([''])
        nn = torch.randn(zz.shape, generator=g, device='cuda')
        xx = self.scheduler.add_noise(zz, nn, tt)
        with torch.no_grad():
            clean_pred = cfg_prediction(self.ext.unet(xx, tt, encoder_hidden_states=hidden).sample,
                                        self.ext.unet(xx, tt, encoder_hidden_states=empty).sample)
            clean = decode(self.ext.vae, predicted_x0(self.scheduler, xx, clean_pred, tt), False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cond = self.unet(xx, tt, encoder_hidden_states=hidden).sample
            uncond = self.unet(xx, tt, encoder_hidden_states=empty).sample
        marked = decode(self.ext.vae, predicted_x0(self.scheduler, xx, cfg_prediction(cond, uncond), tt))
        quality = self.perceptual(marked, clean).mean()
        quality_grad = torch.autograd.grad(self.spec['quality_weight'] * quality, self.params,
                                           retain_graph=active, allow_unused=True)
        qstats = add_bounded_gradient(self.params, quality_grad, base_norm, self.spec['aux_gradient_fraction'])
        del quality_grad
        values.update(quality_lpips=float(quality.detach()), quality_gradient=qstats,
                      original_gradient_norm=float(base_norm), proxy_active=active, quality_timestep=int(tt))
        values['loss'] += self.spec['quality_weight'] * float(quality.detach())
        if active:
            query = (ids[j] + 1009 + step) % len(self.rows)
            while self.rows[query]['text'] == self.rows[ids[j]]['text']:
                query = (query + 1) % len(self.rows)
            loss, metrics = self.proxy.objective(marked, clean, self.rows[ids[j]]['text'],
                                                 self.rows[query]['text'], self.c['seed'] + 900000 + step)
            grads = torch.autograd.grad(self.spec['transfer_weight'] * loss, self.params, allow_unused=True)
            stats = add_bounded_gradient(self.params, grads, base_norm, self.spec['aux_gradient_fraction'])
            if not stats['norm'] > 0:
                raise RuntimeError('No hypergradient from paired learner to teacher')
            values.update(metrics, transfer_gradient=stats, query_training_index=query)
            values['loss'] += self.spec['transfer_weight'] * float(loss.detach())
        return values


def train(run, smoke=False):
    spec = read_json(run / 'refine_spec.json')
    folder = run / 'teacher_fp/refined'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {'config_sha256': sha256(run / 'config.json'), 'spec_sha256': sha256(run / 'refine_spec.json'),
                'carrier_sha256': sha256(run / 'carrier/complete.json'),
                'lineage_sha256': sha256(run / 'teacher_choice.json'),
                'code_manifest_sha256': sha256(run.parent / 'code_manifest.json'),
                'optimizer': 'fresh AdamW for refinement; exact state resume within this experiment',
                'teacher_selection': 'fixed last step; no test/validation selection',
                'auxiliary_gradient_budget': 'each <=25% original-objective gradient norm before global clipping'}
    freeze(folder / 'protocol.json', protocol)
    if (folder / 'trained.json').exists() and not smoke:
        for n, digest in read_json(folder / 'trained.json')['files'].items():
            if sha256(folder / n) != digest:
                raise RuntimeError('Completed teacher changed')
        return
    c = read_json(run / 'config.json')
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    model = Refinement(run, spec)
    optimizer = torch.optim.AdamW(model.params, lr=spec['teacher_lr'])
    rng = torch.Generator('cpu').manual_seed(c['seed'])
    order, offset, start = [], 0, 0
    saved = latest_full_checkpoint(folder) if not smoke else None
    if saved:
        state = restore_training_state(saved, model.unet, optimizer, None, rng, protocol)
        start, order, offset = state['step'], state['order'], state['offset']
        align_loss_log(folder / 'losses.jsonl', start, expected_hash=state['loss_prefix_sha256'])
    elif not smoke:
        align_loss_log(folder / 'losses.jsonl', 0)
    if smoke:
        torch.cuda.reset_peak_memory_stats()
        before = [p.detach().clone() for p in model.params]
        values = checked_teacher_step(model.named, optimizer, None,
            lambda backward: model.compute(list(range(c['teacher_batch'])),
                torch.randn((c['teacher_batch'], 4, 64, 64), generator=rng),
                torch.tensor([100] * c['teacher_batch']), 0, backward, force_proxy=True),
            run / 'smoke_failure.json', {'phase': 'technical_smoke'})
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.params))
        if not changed or not values['quality_gradient']['norm'] > 0:
            raise RuntimeError('Teacher/quality gradient smoke failed')
        write_json(run / 'smoke.json', {'passed': True, 'parameters_changed': changed,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                   'scientific_result': False, 'checkpoint_written': False, **values})
        print('SMOKE PASSED', flush=True)
        return
    images = run / 'images/teacher_clean_train'
    entries = read_json(images / 'image_hashes.json')
    if len(entries) != len(model.rows) or any(sha256(images / v['file_name']) != v['sha256'] for v in entries):
        raise RuntimeError('Training images changed')
    with (folder / 'losses.jsonl').open('a') as log:
        for step in range(start, spec['teacher_steps']):
            if offset + c['teacher_batch'] > len(order):
                order = torch.randperm(len(model.rows), generator=rng).tolist(); offset = 0
            ids = order[offset:offset+c['teacher_batch']]; offset += len(ids)
            noise = torch.randn((len(ids), 4, 64, 64), generator=rng)
            t = torch.randint(0, model.scheduler.config.num_train_timesteps, (len(ids),), generator=rng)
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            values = checked_teacher_step(model.named, optimizer, None,
                  lambda backward: model.compute(ids, noise, t, step, backward),
                  folder / 'failure.json', {'step': step + 1, 'image_indices': ids})
            record = {'step': step + 1, 'seconds': time.monotonic() - started,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, **values}
            log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
            if (step + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
            if (step + 1) % spec['checkpoint_every'] == 0 or step + 1 == spec['teacher_steps']:
                dest = folder / f'checkpoint-{step+1}'
                save_adapter(model.unet, dest, model.lora_config)
                save_training_state(dest, model.unet, optimizer, None, rng, order, offset, step+1, protocol)
    save_adapter(model.unet, folder, model.lora_config)
    freeze(folder / 'trained.json', {'steps': spec['teacher_steps'], 'files': {n: sha256(folder / n)
           for n in ('adapter_model.safetensors', 'adapter_config.json', 'losses.jsonl')}})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    train(args.run, args.smoke)

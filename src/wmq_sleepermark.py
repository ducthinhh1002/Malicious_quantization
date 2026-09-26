"""Blind UNet interventions for SleeperMark; owner trigger/key used only after freeze.

W4 means dequantized integer-grid weights, not a native INT4 inference kernel.
Natural images train timestep denoising with empty text conditioning, not VAE loss.
"""
import argparse
import hashlib
import json
import os
import string
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from PIL import Image, ImageOps

from wmq_blind import RoundingGrid, finetune_quantized_weight, pair_metrics, save_json, save_csv, release_branch_memory
from prepare_sleepermark import prepare, load_extractor, digest

EQUIV_METHODS = ('equivariance_qat', 'adversarial_equivariance_qat')
CFG_METHODS = ('cfg_reconstruction', 'prefix_consistency_qat') + EQUIV_METHODS
METHODS = ('fixed_ptq', 'model_reconstruction', 'natural_rounding', 'natural_finetune', 'natural_joint_finetune') + CFG_METHODS


def selected_weights(unet, scope):
    names = [name for name, p in unet.named_parameters() if p.ndim >= 2 and name.endswith('.weight')
             and (scope == 'all' or (name.startswith('up_blocks.') and '.attentions.' in name))]
    if not names:
        raise ValueError('No UNet weight matrices selected')
    return names


def state_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def noise_target(scheduler, latent, noise, timestep):
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return noise
    if kind == 'v_prediction':
        return scheduler.get_velocity(latent, noise, timestep)
    if kind == 'sample':
        return latent
    raise ValueError(f'Unsupported diffusion prediction type: {kind}')


class QuantizedWeight(nn.Module):
    def __init__(self, weight, finetune=False, group_size=0):
        super().__init__()
        self.enabled = True
        self.quantized = True
        self.finetune = finetune
        self.group_size = group_size
        if finetune:
            self.weight = nn.Parameter(weight.detach().clone())
        else:
            from wmq_grouped_quant import GroupedW4
            self.grid = GroupedW4(weight, group_size) if group_size else RoundingGrid(weight, 4, learn_scale=True)

    def forward(self, original):
        if not self.enabled:
            return original
        if self.finetune:
            from wmq_grouped_quant import grouped_fake_quant
            if not self.quantized:
                return self.weight
            return grouped_fake_quant(self.weight, self.group_size) if self.group_size else finetune_quantized_weight(self.weight, 4)
        return self.grid(self.training)


def attach(unet, names, finetune=False, group_size=0):
    modules = []
    for name in names:
        parent, leaf = name.rsplit('.', 1)
        module = unet.get_submodule(parent)
        adapter = QuantizedWeight(getattr(module, leaf), finetune, group_size)
        parametrize.register_parametrization(module, leaf, adapter)
        modules.append((module, leaf, adapter))
    return modules


def switch(modules, enabled=True, quantized=True):
    for _, _, adapter in modules:
        adapter.enabled, adapter.quantized = enabled, quantized


def detach(modules):
    for module, leaf, _ in modules:
        parametrize.remove_parametrizations(module, leaf, leave_parametrized=False)


def snapshot(unet, names):
    # get_parameter does not resolve parametrized weight tensors.
    return {name: getattr(unet.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1])
            .detach().cpu().contiguous().clone() for name in names}


@torch.no_grad()
def restore(unet, state):
    for name, value in state.items():
        unet.get_parameter(name).copy_(value)


@torch.no_grad()
def encode_text(pipe, prompts):
    ids = pipe.tokenizer(prompts, padding='max_length', truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(next(pipe.text_encoder.parameters()).device)
    return pipe.text_encoder(ids)[0]


def load_image(path):
    with Image.open(path) as im:
        image = ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'), (512, 512), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)


def train_branch(pipe, scheduler, dataset, names, method, args, output):
    device = next(pipe.unet.parameters()).device
    is_finetune = method in ('natural_finetune', 'natural_joint_finetune')
    modules = attach(pipe.unet, names, is_finetune, getattr(args, "quant_group_size", 0))
    params = [p for _, _, adapter in modules for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.ft_lr if is_finetune else args.lr,
                                 weight_decay=0, foreach=False)
    pipe.unet.train()
    # Parametrizations remain attached during recomputation; unlike functional_call,
    # native diffusers gradient checkpointing sees the same weights on backward.
    pipe.unet.enable_gradient_checkpointing()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    order = np.random.default_rng(args.seed)
    auxiliary_order = np.random.default_rng(args.seed + 7919)
    rows = []
    cfg = method in CFG_METHODS
    equiv = method in EQUIV_METHODS
    empty = encode_text(pipe, ['']).detach() if cfg else None
    trajectory = cfg and bool(dataset) and len(dataset[0]) == 4
    late_indices = ([i for i, item in enumerate(dataset)
                     if item[3] <= args.late_fraction * scheduler.config.num_train_timesteps] if equiv and trajectory else [])
    if equiv and (not trajectory or not late_indices):
        detach(modules)
        pipe.unet.disable_gradient_checkpointing()
        pipe.unet.eval()
        raise ValueError('Spatial QAT requires trajectory calibration with late-timestep records')
    count = 0 if method == 'fixed_ptq' else args.steps
    started = time.perf_counter()
    forward_calls = [0]
    def count_forward(module, inputs):
        forward_calls[0] += 1
    counter = pipe.unet.register_forward_pre_hook(count_forward)
    try:
        for step in range(count):
            indices = order.integers(len(dataset), size=args.train_batch_size)
            z = torch.cat([dataset[i][0] for i in indices]).to(device)
            condition = torch.cat([dataset[i][1] for i in indices]).to(device)
            if trajectory:
                noisy = z
                t = torch.tensor([dataset[i][3] for i in indices], device=device, dtype=torch.long)
            else:
                noise = torch.randn(z.shape, device=z.device, generator=rng)
                t = torch.randint(scheduler.config.num_train_timesteps, (len(z),), device=z.device, generator=rng)
                noisy = scheduler.add_noise(z, noise, t)
            switch(modules, enabled=False)
            with torch.no_grad():
                marked = pipe.unet(noisy, t, encoder_hidden_states=condition).sample.detach()
            if cfg:
                unconditional = empty.expand(len(z), -1, -1)
                with torch.no_grad():
                    ref_u = pipe.unet(noisy, t, encoder_hidden_states=unconditional).sample.detach()
                    ref_cfg = ref_u + 7.5 * (marked - ref_u)
                optimizer.zero_grad(set_to_none=True)
                switch(modules)
                pred_u = pipe.unet(noisy, t, encoder_hidden_states=unconditional).sample
                pred_c = pipe.unet(noisy, t, encoder_hidden_states=condition).sample
                ordinary_loss = F.mse_loss(pred_u + 7.5 * (pred_c - pred_u), ref_cfg) / 7.5**2
                ordinary_loss = ordinary_loss + F.mse_loss(pred_u, ref_u)
                ordinary_loss.backward()
                del pred_u, pred_c
                losses = [ordinary_loss.detach()]
                spatial_metrics = {}
                if equiv and args.spatial_weight > 0 and step / max(count, 1) > .1:
                    from wmq_sleeper_equivariance import probe_condition, spatial_target, predicted_x0
                    # Extra late-time batch; ordinary CFG loss above still covers the full trajectory.
                    late = auxiliary_order.choice(late_indices, size=args.train_batch_size)
                    late_z = torch.cat([dataset[i][0] for i in late]).to(device)
                    late_t = torch.tensor([dataset[i][3] for i in late], device=device, dtype=torch.long)
                    context = torch.cat([dataset[i][1] for i in late]).to(device)
                    shift = tuple(int(a * b) for a, b in zip(auxiliary_order.choice([-1, 1], size=2),
                        auxiliary_order.integers(1, args.spatial_shift + 1, size=2)))
                    switch(modules, enabled=False)
                    ramp = max(0., min(1., (step / max(count, 1) - .1) / .2))
                    if method == 'adversarial_equivariance_qat' and step % args.probe_every == 0 and ramp > 0:
                        context, spatial_metrics = probe_condition(pipe.unet, late_z, late_t, context,
                            shift, scheduler, args.probe_radius, args.probe_steps, args.probe_tokens, rng)
                    target_x0, diagnostics = spatial_target(pipe.unet, late_z, late_t, context,
                        shift, scheduler, args.max_spatial_correction)
                    spatial_metrics.update(diagnostics)
                    switch(modules)
                    prediction = pipe.unet(late_z, late_t, encoder_hidden_states=context).sample
                    student_x0 = predicted_x0(prediction, late_z, late_t, scheduler)
                    spatial_loss = args.spatial_weight * ramp * F.mse_loss(student_x0, target_x0)
                    spatial_loss.backward()
                    losses.append(spatial_loss.detach())
                    del student_x0, prediction, target_x0
                if method == 'prefix_consistency_qat':
                    # Sample from punctuation alphabet, independent of owner assets.
                    prefixes = [''.join(order.choice(list(string.punctuation), size=int(order.integers(1, 9))))
                                + ' ' + dataset[i][2] for i in indices]
                    prefix_cond = encode_text(pipe, prefixes)
                    pred = pipe.unet(noisy, t, encoder_hidden_states=prefix_cond).sample
                    ramp = max(0., min(1., (step / max(count, 1) - .2) / .3))
                    prefix_loss = args.prefix_weight * ramp * F.mse_loss(pred, marked)
                    prefix_loss.backward()
                    losses.append(prefix_loss.detach())
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                if (step + 1) % args.log_every == 0 or step + 1 == count:
                    row = {'method': method, 'step': step + 1,
                           'ordinary_cfg_loss': float(losses[0]),
                           'auxiliary_loss': float(losses[1]) if len(losses) > 1 else 0.,
                           'elapsed_seconds': time.perf_counter() - started,
                           'logical_unet_forwards': forward_calls[0],
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.,
                           **{k: float(v) for k, v in spatial_metrics.items()}}
                    rows.append(row)
                    save_csv(output / f'{method}_training.csv', rows)
                    print(row, flush=True)
                continue
            target = marked if method == 'model_reconstruction' else noise_target(scheduler, z, noise, t)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Backward each pass before changing quantization mode, so checkpoint
            # recomputation uses exactly the mode used in its forward.
            modes = [False, True] if method == 'natural_joint_finetune' else ([False] if is_finetune else [True])
            for quantized in modes:
                switch(modules, enabled=True, quantized=quantized)
                prediction = pipe.unet(noisy, t, encoder_hidden_states=condition).sample
                loss = F.mse_loss(prediction.float(), target.float())
                if method != 'model_reconstruction':
                    loss = loss + args.preserve_weight * F.mse_loss(prediction.float(), marked.float())
                (loss / len(modes)).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % args.log_every == 0 or step + 1 == count:
                row = {'method': method, 'step': step + 1,
                       'training_loss': float(torch.stack(losses).mean()),
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.}
                rows.append(row)
                save_csv(output / f'{method}_training.csv', rows)
                print(row, flush=True)
        pipe.unet.eval()
        switch(modules)
        result = {method + '_w4': snapshot(pipe.unet, names)}
        if is_finetune:
            switch(modules, quantized=False)
            result[method + '_fp32'] = snapshot(pipe.unet, names)
        return result
    finally:
        counter.remove()
        detach(modules)
        pipe.unet.disable_gradient_checkpointing()
        pipe.unet.eval()


@torch.no_grad()
def generate(pipe, prompts, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    # Per-image seeds and batch size one keep paired generation and memory bounded.
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=args.inference_steps, guidance_scale=7.5,
                     generator=torch.Generator(device='cuda').manual_seed(args.seed + 100000 + i)).images[0]
        image.save(folder / f'{i:06d}.png')
        if (i+1) % 10 == 0:
            print(f'{folder.name}: {i+1}/{len(prompts)}', flush=True)


@torch.no_grad()
def owner_scores(pipe, extractor, key, folder, fpr, seed):
    from evaluate_blind_watermark import detection_threshold, wilson_interval
    upper = detection_threshold(len(key), fpr, 'double')
    counts = []
    # Match official extraction: encode rendered RGB, sample posterior, scale latent.
    # Reuse independent posterior RNG seed across paired branches.
    rng = torch.Generator(device='cuda').manual_seed(seed + 200000)
    for path in sorted(folder.glob('*.png')):
        x = load_image(path).cuda()
        z = pipe.vae.encode(x * 2 - 1).latent_dist.sample(generator=rng) * pipe.vae.config.scaling_factor
        prediction = torch.sigmoid(extractor(z)).round()
        counts.append(int((prediction.flatten() == key).sum()))
    matches = np.asarray(counts)
    detected = (matches >= upper) | (matches <= len(key) - upper)
    low, high = wilson_interval(int(detected.sum()), len(matches))
    return {'bit_accuracy': float(matches.mean() / len(key)), 'tpr': float(detected.mean()),
            'detected': int(detected.sum()), 'n': len(matches), 'tpr_ci95_low': low, 'tpr_ci95_high': high,
            'threshold_upper_inclusive': upper, 'threshold_lower_inclusive': len(key) - upper,
            'bit_matches': counts}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--methods', nargs='+', choices=METHODS,
                   default=['fixed_ptq', 'cfg_reconstruction', *EQUIV_METHODS])
    p.add_argument('--quant-group-size', type=int, default=64, help='0 reproduces legacy channel quantization')
    p.add_argument('--prefix-weight', type=float, default=.25)
    p.add_argument('--calibration-mode', choices=['trajectory', 'renoised'], default='trajectory')
    p.add_argument('--trajectory-points', type=int, default=8)
    p.add_argument('--late-fraction', type=float, default=.35)
    p.add_argument('--spatial-shift', type=int, default=4, help='Latent cells; used only during training')
    p.add_argument('--spatial-weight', type=float, default=1.)
    p.add_argument('--max-spatial-correction', type=float, default=.05, help='Per-sample RMS cap in x0 latent units')
    p.add_argument('--probe-radius', type=float, default=.15, help='Relative L2 radius on selected context tokens')
    p.add_argument('--probe-steps', type=int, default=2)
    p.add_argument('--probe-every', type=int, default=4)
    p.add_argument('--probe-tokens', type=int, default=4)
    p.add_argument('--scope', choices=['up_attentions', 'all'], default='up_attentions')
    p.add_argument('--train-n', type=int, default=256)
    p.add_argument('--test-n', type=int, default=100)
    p.add_argument('--train-batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ft-lr', type=float, default=1e-5)
    p.add_argument('--preserve-weight', type=float, default=.5)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--inference-steps', type=int, default=50)
    p.add_argument('--fpr', type=float, default=.001)
    p.add_argument('--natural-images', type=Path)
    p.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompt.txt'))
    p.add_argument('--fid-real-reference', type=Path)
    p.add_argument('--skip-fid', action='store_true')
    p.add_argument('--report-min-ssim', type=float, default=.8, help='Report-only triggered-image quality threshold')
    p.add_argument('--report-min-psnr', type=float, default=25., help='Report-only threshold; never stops training')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.steps, args.train_n, args.test_n, args.train_batch_size, args.log_every, args.inference_steps) < 1:
        p.error('Counts must be positive')
    if not 0 < args.fpr < 1 or min(args.lr, args.ft_lr) <= 0 or args.preserve_weight < 0:
        p.error('Invalid FPR, learning rate or preservation weight')
    if args.quant_group_size < 0 or not np.isfinite(args.prefix_weight) or args.prefix_weight < 0:
        p.error('Invalid quantization group size or prefix weight')
    if (min(args.trajectory_points, args.spatial_shift, args.probe_every, args.probe_tokens) < 1
            or args.probe_steps < 0 or not 0 < args.late_fraction <= 1
            or not all(np.isfinite(v) and v >= 0 for v in
                       (args.spatial_weight, args.max_spatial_correction, args.probe_radius))):
        p.error('Invalid spatial/probe calibration settings')
    if any(m in EQUIV_METHODS for m in args.methods) and args.calibration_mode != 'trajectory':
        p.error('Spatial QAT requires --calibration-mode trajectory')
    if len(set(args.methods)) != len(args.methods):
        p.error('Duplicate methods')
    if not 0 <= args.report_min_ssim <= 1 or not np.isfinite(args.report_min_psnr):
        p.error('Invalid report quality thresholds')
    if not torch.cuda.is_available():
        raise RuntimeError('SleeperMark experiment requires CUDA')
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
    from safetensors.torch import save_file, load_file
    project = Path(__file__).resolve().parents[1]
    heavy = Path(os.environ.get('WMQ_HEAVY_ROOT', project / 'output_artifacts')).resolve()
    name = 'sleepermark_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(os.environ.get('WMQ_ATTACK_OUTPUT_ROOT', project / 'output_attack')) / name
    artifacts, images = heavy / 'checkpoints' / name, heavy / 'images' / name
    for folder in (output, artifacts, images):
        folder.mkdir(parents=True, exist_ok=False)
    print(f'Results: {output}\nHeavy artifacts: {artifacts}\nImages: {images}', flush=True)
    unet_path, extractor_path, key_path, extractor_source = prepare(heavy / 'models' / 'sleepermark')
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
        unet=UNet2DConditionModel.from_pretrained(unet_path), torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.unet.requires_grad_(False).eval()
    # Frozen components are checked at the end; interventions must touch the UNet.
    frozen_hashes = {'vae': state_hash(pipe.vae), 'text_encoder': state_hash(pipe.text_encoder)}
    names = selected_weights(pipe.unet, args.scope)
    original = snapshot(pipe.unet, names)
    original_hash = state_hash(pipe.unet)
    prompts = list(dict.fromkeys(s.strip() for s in args.prompts.read_text(encoding='utf-8').splitlines() if s.strip()))
    if len(prompts) <= args.test_n:
        raise ValueError('Need test-n held-out unique prompts plus at least one training prompt')
    shuffled = np.random.default_rng(args.seed).permutation(len(prompts))
    test_prompts = [prompts[i] for i in shuffled[:args.test_n]]
    train_prompts = [prompts[i] for i in shuffled[args.test_n:]]
    model_data, natural_data, trajectory_data = [], [], []
    if any(m in ('model_reconstruction', *CFG_METHODS) for m in args.methods):
        for i in range(args.train_n):
            prompt = train_prompts[i % len(train_prompts)]
            from wmq_sleeper_equivariance import TrajectoryCapture
            capture = (TrajectoryCapture(pipe.unet, args.inference_steps, args.trajectory_points)
                       if args.calibration_mode == 'trajectory' and any(m in CFG_METHODS for m in args.methods) else None)
            try:
                with torch.no_grad():
                    latent = pipe(prompt, output_type='latent', num_inference_steps=args.inference_steps,
                                  guidance_scale=7.5, generator=torch.Generator(device='cuda').manual_seed(args.seed+i)).images
                    condition = encode_text(pipe, [prompt]).cpu()
                    model_data.append((latent.cpu(), condition, prompt))
                    if capture is not None:
                        if len(capture.records) != min(args.trajectory_points, args.inference_steps):
                            raise RuntimeError('Incomplete inference trajectory capture')
                        trajectory_data.extend((z, condition, prompt, t) for z, t in capture.records)
            finally:
                if capture is not None:
                    capture.close()
            if (i+1) % 10 == 0:
                print(f'Model-only latent calibration {i+1}/{args.train_n}', flush=True)
    natural_files = []
    if any(m.startswith('natural_') for m in args.methods):
        if args.natural_images is None:
            from prepare_natural_images import prepare as prepare_natural
            args.natural_images = heavy / 'datasets' / f'sleepermark_coco_{args.train_n}_{args.seed}'
            prepare_natural(args.natural_images, args.train_n, args.seed, 8)
        from wmq_fid import image_files
        natural_files = image_files(args.natural_images)
        if len(natural_files) < args.train_n:
            raise ValueError('Insufficient natural images')
        natural_files = [natural_files[i] for i in np.random.default_rng(args.seed).permutation(len(natural_files))[:args.train_n]]
        empty = encode_text(pipe, ['']).detach().cpu()
        for path in natural_files:
            with torch.no_grad():
                z = pipe.vae.encode(load_image(path).cuda() * 2 - 1).latent_dist.mode() * pipe.vae.config.scaling_factor
                natural_data.append((z.cpu(), empty))
    manifest = {'watermark': 'SleeperMark', 'base_model': 'CompVis/stable-diffusion-v1-4',
        'reference_label': 'marked_reference_test', 'image_root': str(images), 'artifact_root': str(artifacts),
        'test_branches': [{'label': 'marked_reference_test', 'role': 'reference'}],
        'attack_component': 'unet', 'scope': args.scope, 'selected_weight_names': names,
        'test_used_for_selection': False, 'owner_assets_used_for_training': False,
        'selection': 'Predeclared final step; no owner or test quality selection',
        'quality_policy': 'report_only; SSIM and FID never reject or choose checkpoints',
        'triggered_quality_thresholds': {'ssim': args.report_min_ssim, 'psnr': args.report_min_psnr},
        'natural_conditioning': 'Empty text; unpaired public images, not paired reconstruction',
        'cfg_calibration': args.calibration_mode,
        'trajectory_records': len(trajectory_data),
        'trajectory_timesteps': sorted(set(record[3] for record in trajectory_data)),
        'spatial_objective': 'Late-time aligned spatial teacher ensemble; bounded x0 highpass correction, not ownership oracle',
        'probe_objective': 'Continuous context spatial-defect maximization; no trigger recovery claim',
        'prefix_calibration': 'TRAIN prompts with independently sampled punctuation; no owner trigger used',
        'quantization_group_size': args.quant_group_size,
        'precision': 'FP32 activations; W4 fake quantization only on selected matrices',
        'training_timesteps': 'CFG: uniform cached trajectory records (or renoised legacy); spatial auxiliary: late records; natural: uniform DDPM',
        'train_prompts': train_prompts, 'test_prompts': test_prompts,
        'natural_train_files': natural_files, 'original_unet_sha256': original_hash,
        'frozen_components': frozen_hashes,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    checkpoints = {}
    for method in args.methods:
        restore(pipe.unet, original)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        print(f'Training UNet method: {method}', flush=True)
        dataset = (trajectory_data if method in CFG_METHODS and args.calibration_mode == 'trajectory' else
                   model_data if method in ('model_reconstruction', *CFG_METHODS) else natural_data)
        states = train_branch(pipe, scheduler, dataset,
                              names, method, args, output)
        for label, state in states.items():
            changed = sum(not torch.equal(state[n], original[n]) for n in names)
            if not changed:
                raise RuntimeError(f'{label} did not modify the UNet')
            path = artifacts / (label + '.safetensors')
            save_file(state, str(path))
            checkpoints[label] = {'path': str(path), 'sha256': digest(path), 'changed_matrices': changed}
            manifest['test_branches'].append({'label': label, 'role': 'attack',
                'threat_model': ('restricted_weight_finetune' if label.endswith('_fp32') else
                                'restricted_weight_finetune_plus_quantization') if 'finetune' in label else 'quantizer_only'})
        del states
        release_branch_memory('cuda')
    restore(pipe.unet, original)
    if state_hash(pipe.unet) != original_hash:
        raise RuntimeError('Unselected UNet weights changed')
    save_json(output / 'manifest.json', manifest)
    save_json(output / 'selection_frozen.json', {'phase': 'before_test_generation',
              'manifest_sha256': digest(output / 'manifest.json'), 'checkpoints': checkpoints,
              'test_used_for_selection': False})
    # Owner-only protocol starts here. Never pass this trigger, key or extractor to train_branch.
    extractor, key = load_extractor(extractor_source, extractor_path, key_path, 'cuda')
    triggered = ['*[Z]& ' + prompt for prompt in test_prompts]
    rows = []
    for branch in manifest['test_branches']:
        label = branch['label']
        restore(pipe.unet, original)
        if label in checkpoints:
            info = checkpoints[label]
            if digest(info['path']) != info['sha256']:
                raise ValueError('Frozen UNet checkpoint changed')
            restore(pipe.unet, load_file(info['path']))
        generate(pipe, triggered, images / label, args)
        row = {'label': label, **owner_scores(pipe, extractor, key, images / label, args.fpr, args.seed)}
        # Untriggered outputs measure ordinary utility and supply empirical false positives.
        ordinary = images / (label + '_ordinary')
        generate(pipe, test_prompts, ordinary, args)
        ordinary_owner = owner_scores(pipe, extractor, key, ordinary, args.fpr, args.seed)
        row['untriggered_detection_rate'] = ordinary_owner['tpr']
        quality = []
        triggered_quality = []
        for i in range(args.test_n):
            filename = f'{i:06d}.png'
            psnr, ssim = pair_metrics(load_image(ordinary / filename).cuda(),
                                     load_image(images / 'marked_reference_test_ordinary' / filename).cuda())
            quality.append({'index': i, 'psnr': float(psnr[0]), 'ssim': float(ssim[0])})
            p_trigger, s_trigger = pair_metrics(load_image(images / label / filename).cuda(),
                load_image(images / 'marked_reference_test' / filename).cuda())
            matches = row['bit_matches'][i]
            upper, lower = row['threshold_upper_inclusive'], row['threshold_lower_inclusive']
            base_matches = (rows[0] if rows else row)['bit_matches'][i]
            baseline_detected = base_matches >= upper or base_matches <= lower
            evaded = lower < matches < upper
            quality_ok = float(p_trigger[0]) >= args.report_min_psnr and float(s_trigger[0]) >= args.report_min_ssim
            triggered_quality.append({'index': i, 'psnr': float(p_trigger[0]), 'ssim': float(s_trigger[0]),
                'bit_matches': matches, 'baseline_detected': baseline_detected, 'evaded': evaded,
                'quality_pass': quality_ok, 'joint_success': baseline_detected and evaded and quality_ok})
        row.update(psnr=float(np.mean([r['psnr'] for r in quality])),
                   ssim=float(np.mean([r['ssim'] for r in quality])))
        baseline_n = sum(r['baseline_detected'] for r in triggered_quality)
        row.update(triggered_psnr=float(np.mean([r['psnr'] for r in triggered_quality])),
                   triggered_ssim=float(np.mean([r['ssim'] for r in triggered_quality])),
                   triggered_quality_pass_fraction=float(np.mean([r['quality_pass'] for r in triggered_quality])),
                   evasion_rate=1-row['tpr'], baseline_detected_n=baseline_n,
                   joint_success_count=sum(r['joint_success'] for r in triggered_quality),
                   joint_success_rate=sum(r['joint_success'] for r in triggered_quality)/baseline_n if baseline_n else None)
        row['valid_clean_baseline'] = (row['tpr'] if not rows else rows[0]['tpr']) >= .9
        rows.append(row)
        save_json(output / f'{label}_quality.json', quality)
        save_csv(output / f'{label}_triggered_quality.csv', triggered_quality)
        save_csv(output / 'watermark_retention.csv', rows)
        save_json(output / 'owner_evaluation.json', {'rows': rows, 'detector': 'double', 'total_fpr': args.fpr,
            'extractor_sha256': digest(extractor_path), 'secret_sha256': digest(key_path),
            'posterior_protocol': 'sample, matched RNG across branches, image-to-latent scaling applied',
            'baseline_warning': None if rows[0]['valid_clean_baseline'] else
                'Invalid clean baseline: do not interpret attack TPR as successful removal'})
        print(row, flush=True)
    for component, expected in frozen_hashes.items():
        if state_hash(getattr(pipe, component)) != expected:
            raise RuntimeError(f'{component} changed during UNet experiment')
    save_json(output / 'report.json', {'status': 'complete', 'rows': rows,
              'frozen_components_verified': True, 'test_used_for_selection': False})
    del pipe, extractor
    release_branch_memory('cuda')
    if not args.skip_fid:
        from wmq_fid import evaluate
        try:
            evaluate(output, args.fid_real_reference)
            evaluate(output, args.fid_real_reference, ordinary=True)
        except Exception as error:
            save_json(output / 'fid_error.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            print(f'WARNING: FID failed; owner results retained. See {output / "fid_error.json"}', flush=True)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()

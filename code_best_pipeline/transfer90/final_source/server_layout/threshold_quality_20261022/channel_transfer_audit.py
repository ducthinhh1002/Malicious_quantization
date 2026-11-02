"""Read-only paired validation audit of the RGB -> student learning channel.

These are reconstruction probes, NOT trained-student radioactivity estimates.
No test images, parameter updates, generated-image exports, or threshold fitting.
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Malicious_quantization/bitmark_lora'))
from common import freeze, read_json, sha256, image_name, write_json


def validation_indices(count, n, seed):
    if not 0 < n <= count:
        raise ValueError('Audit size must be within the validation split')
    return sorted(random.Random(seed).sample(range(count), n))


def paired_summary(clean, marked, threshold):
    import numpy as np
    a, b = np.asarray(clean, dtype=float), np.asarray(marked, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Expected equal nonempty paired vectors')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite scores')
    delta = b - a
    rng = np.random.default_rng(20261024)
    means = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
    return {'n': len(a), 'clean_mean': float(a.mean()), 'marked_mean': float(b.mean()),
            'paired_signal': float(delta.mean()),
            'paired_signal_ci95': np.quantile(means, [.025, .975]).tolist(),
            'marked_p10_margin': float(np.quantile(b, .1) - threshold),
            'clean_fraction_above_frozen_threshold': float((a >= threshold).mean()),
            'marked_fraction_above_frozen_threshold': float((b >= threshold).mean())}


def audit(args):
    import torch
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from semantic_teacher import extractor, pixels
    from semantic_carrier import SemanticCarrier
    from semantic_teacher_train import predicted_x0

    source = ROOT / 'kappa1_manojb_5k/source'
    original = ROOT / 'kappa1_manojb_5k/students'
    recovery = ROOT / 'scale085_recovery_20261024/students'
    folders = {
        'clean': source / 'images/teacher_clean_validation',
        'scale100': original / 'k1_evolution/images/teacher_wm_validation',
        'scale085': original / 'k1_scale085/images/teacher_wm_validation',
        'conv_restore': recovery / 'scale085_conv_restore/images/teacher_wm_validation',
    }
    rows = read_json(source / 'prompts.json')['validation']
    indices = validation_indices(len(rows), args.n, args.seed)
    assets = read_json(source / 'assets.json')
    threshold_path = source / 'result/threshold_teacher.json'
    threshold = read_json(threshold_path)['threshold']
    identity = {}
    for name, folder in folders.items():
        entries = read_json(folder / 'image_hashes.json')
        if len(entries) != len(rows):
            raise RuntimeError(f'Validation size mismatch: {name}')
        proof = []
        for i in indices:
            entry = entries[i]
            if entry['file_name'] != image_name(i) or entry['text'] != rows[i]['text']:
                raise RuntimeError(f'Validation identity mismatch: {name}/{i}')
            if entry['sha256'] != sha256(folder / image_name(i)):
                raise RuntimeError(f'Changed validation image: {name}/{i}')
            proof.append(entry)
        identity[name] = proof
    for entries in identity.values():
        if [v['seed'] for v in entries] != [v['seed'] for v in identity['clean']]:
            raise RuntimeError('Unpaired generation seeds')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {'split': 'validation', 'indices': indices, 'seed': args.seed,
            'batch_size': args.batch, 'timesteps': [100, 300, 500],
            'posterior_draws': 2, 'common_random_numbers': True,
            'threshold': threshold, 'threshold_sha256': sha256(threshold_path),
            'carrier_manifest_sha256': sha256(source / 'carrier/complete.json'),
            'script_sha256': sha256(Path(__file__)), 'images': identity,
            'student_model': assets['student'], 'teacher_model': assets['teacher_clean'],
            'student_parameters_updated': False,
            'limitation': 'Unadapted conditional one-step x0 probes, no CFG; not training or full regeneration. '
                          'Channel-transformed nulls are not recalibrated: threshold fractions are diagnostic, '
                          'not TPR at 1% FPR. Image bootstrap is not training-seed uncertainty.'}
    freeze(args.output / 'plan.json', plan)
    ext = extractor(source)
    carrier = SemanticCarrier.load(source / 'carrier', 'cuda')
    student = StableDiffusionPipeline.from_pretrained(assets['student'], torch_dtype=torch.float16,
                          safety_checker=None, local_files_only=True).to('cuda')
    student.vae.to(dtype=torch.float32)
    for module in (student.vae, student.unet, student.text_encoder):
        module.eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(assets['student'], subfolder='scheduler', local_files_only=True)
    raw = {}

    def sample_noise(shape, ids, salt):
        # Individual generators make the draws invariant to batching and branch order.
        return torch.cat([torch.randn((1, *shape[1:]), generator=torch.Generator('cuda').manual_seed(
            args.seed + 100000 * salt + i), device='cuda', dtype=torch.float32) for i in ids])

    with torch.inference_mode():
        for name, folder in folders.items():
            channels = {'rgb': [], 'student_vae_mean': [], 'student_vae_sample0': [], 'student_vae_sample1': []}
            losses = {str(t): [] for t in plan['timesteps']}
            for t in plan['timesteps']:
                channels[f'unadapted_x0_t{t}'] = []
            for start in range(0, len(indices), args.batch):
                ids = indices[start:start + args.batch]
                prompts = [rows[i]['text'] for i in ids]
                x = pixels([folder / image_name(i) for i in ids])
                hidden = ext.text(prompts)

                def score(rgb):
                    return carrier.scalar_score(ext.encode_latent(ext.image_latent(rgb.clamp(-1, 1)),
                                                   text_hidden=hidden)).cpu().tolist()

                channels['rgb'].extend(score(x))
                posterior = student.vae.encode(x).latent_dist
                sf = student.vae.config.scaling_factor
                channels['student_vae_mean'].extend(score(student.vae.decode(posterior.mean).sample))
                z = None
                for draw in range(2):
                    sampled = posterior.mean + posterior.std * sample_noise(posterior.mean.shape, ids, draw + 1)
                    channels[f'student_vae_sample{draw}'].extend(score(student.vae.decode(sampled).sample))
                    if draw == 0:
                        z = sampled * sf
                tokens = student.tokenizer(prompts, padding='max_length', truncation=True,
                         max_length=student.tokenizer.model_max_length, return_tensors='pt')
                text = student.text_encoder(tokens.input_ids.to('cuda'))[0]
                for timestep in plan['timesteps']:
                    t = torch.full((len(ids),), timestep, device='cuda', dtype=torch.long)
                    noise = sample_noise(z.shape, ids, 10 + timestep)
                    noisy = scheduler.add_noise(z, noise, t)
                    prediction = student.unet(noisy.to(torch.float16), t, encoder_hidden_states=text).sample.float()
                    kind = scheduler.config.prediction_type
                    if kind == 'epsilon':
                        target = noise
                    elif kind == 'v_prediction':
                        target = scheduler.get_velocity(z, noise, t)
                    elif kind == 'sample':
                        target = z
                    else:
                        raise ValueError(kind)
                    losses[str(timestep)].extend((prediction-target).square().mean((1, 2, 3)).cpu().tolist())
                    x0 = predicted_x0(scheduler, noisy, prediction, t)
                    channels[f'unadapted_x0_t{timestep}'].extend(score(student.vae.decode(x0 / sf).sample))
                print(f'{name}: {start + len(ids)}/{len(indices)}', flush=True)
            raw[name] = {'scores': channels, 'denoising_mse': losses}
            write_json(args.output / 'partial.json', raw)

    import numpy as np
    summary = {}
    for name, data in raw.items():
        if name == 'clean':
            continue
        summary[name] = {}
        for channel, marked in data['scores'].items():
            summary[name][channel] = paired_summary(raw['clean']['scores'][channel], marked, threshold)
        direct = summary[name]['rgb']['paired_signal']
        for value in summary[name].values():
            value['paired_signal_retention_vs_rgb'] = value['paired_signal'] / direct if abs(direct) > 1e-8 else None
        summary[name]['denoising_mse_marked_minus_clean'] = {
            t: float(np.mean(np.array(v) - np.array(raw['clean']['denoising_mse'][t])))
            for t, v in data['denoising_mse'].items()}
    if sha256(threshold_path) != plan['threshold_sha256']:
        raise RuntimeError('Detector changed during audit')
    freeze(args.output / 'result.json', {'summary': summary, 'raw': raw, 'plan_sha256': sha256(args.output / 'plan.json')})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('batch must be positive')
    audit(args)

"""Optimize real 50-step PNDM outputs; fixed verifier and ordinary final student."""
import argparse
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
import transfer_refine as engine
from transfer_refine import norm_of, add_bounded_gradient
from transfer_proxy import decode, rgb8
from common import read_json, write_json, image_name
from semantic_teacher_train import threshold_teacher_backward, checked_teacher_step
from semantic_threshold import calibrated_shift_loss


def sample_native(unet, scheduler, noise, hidden, steps=50, checkpointed=True):
    """Entire original trajectory, including PNDM history. No cached prefix.

    Scheduler is fresh for each image. Original CFG and timestep schedule are
    preserved; BF16 denoiser/FP32 state stabilize training, while production
    remains merged FP16. Recompute each UNet
    invocation under checkpointing, keeping all sampler steps differentiable.
    """
    scheduler.set_timesteps(steps, device=noise.device)
    # Long FP16 backward chains can overflow before a finite gradient can be
    # clipped (observed at training step65). Preserve seeded noise values, but
    # keep sampler state/CFG in FP32 and use BF16 activations on CUDA. Final
    # production rendering remains the original merged FP16 pipeline.
    z = noise.float() * scheduler.init_noise_sigma
    def prediction(x, t, text):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            return unet(x, t, encoder_hidden_states=text).sample
    for timestep in scheduler.timesteps:
        x = scheduler.scale_model_input(torch.cat([z, z]), timestep)
        result = (checkpoint(prediction, x, timestep, hidden, use_reentrant=False)
                  if checkpointed else prediction(x, timestep, hidden))
        u, c = result.float().chunk(2)
        z = scheduler.step(u + 7.5 * (c - u), timestep, z).prev_sample
    return z


class NativeRefinement(engine.Refinement):
    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from diffusers import PNDMScheduler
        from semantic_teacher import pixels
        with torch.no_grad():
            images = pixels([self.run / 'images/teacher_clean_train' / image_name(i) for i in ids])
            z = self.ext.image_latent(images)
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(z, noise.cuda(), t.cuda())
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
                              noisy, text, t.cuda(), 1., self.c, step, backward, {})
        base_norm = norm_of([p.grad for p in self.params])
        values.update(original_gradient_norm=float(base_norm), native_active=False)
        if not self.spec['native'] or (not force_proxy and (step + 1) % self.spec['native_every']):
            return values
        # No held-out prompts: uniformly cycle the shuffled training batches.
        j = (step // self.spec['native_every']) % len(ids)
        index = ids[j]
        seed = self.c['seed'] * 1000000 + self.rows[index]['image_id']
        generator = torch.Generator('cuda').manual_seed(seed)
        initial = torch.randn((1, 4, 64, 64), generator=generator, device='cuda', dtype=torch.float16)
        hidden = self.ext.text(['', self.rows[index]['text']]).half()
        scheduler = PNDMScheduler.from_pretrained(self.assets['teacher_clean'], subfolder='scheduler', local_files_only=True)
        latent = sample_native(self.unet, scheduler, initial, hidden, self.c['inference_steps'])
        marked = rgb8(decode(self.ext.vae, latent.float()))
        clean = images[j:j+1]
        quality = self.perceptual(marked, clean).mean()
        pixel = (marked - clean).square().mean()
        def coordinates(rgb):
            latent = self.ext.image_latent(rgb)
            return self.carrier.coordinates(self.ext.encode_latent(latent, text_hidden=text[j:j+1]))
        with torch.no_grad():
            qclean = coordinates(clean)
        qmarked = coordinates(marked)
        frozen = self.c['teacher_calibration']
        semantic, stats = calibrated_shift_loss(qclean, qmarked, self.carrier.signs,
                                  frozen['threshold'], frozen['sigma_clean'], 1.)
        loss = (self.spec['native_lpips_weight'] * quality + self.spec['native_pixel_weight'] * pixel
                + self.spec['native_semantic_weight'] * semantic)
        gradients = torch.autograd.grad(loss, self.params, allow_unused=True)
        budget = add_bounded_gradient(self.params, gradients, base_norm, self.spec['aux_gradient_fraction'])
        if force_proxy and (not budget['norm'] > 0 or not budget['added_norm'] > 0):
            raise RuntimeError('Native full-sampler auxiliary gradient absent')
        values.update(native_active=True, quality_lpips=float(quality.detach()), native_pixel=float(pixel.detach()),
            native_semantic=float(semantic.detach()), native_score=stats['mean_rfp'], native_clean_score=stats['mean_r0'],
            native_loss=float(loss.detach()), quality_gradient=budget,
            native_training_index=index, native_seed=seed, native_scheduler=type(scheduler).__name__,
            native_denoiser_calls=len(scheduler.timesteps), native_gradient_steps=len(scheduler.timesteps),
            native_training_precision='BF16 denoiser, FP32 CFG/scheduler state, FP16 seeded noise values')
        values['loss'] += float(loss.detach())
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = read_json(args.run / 'refine_spec.json')
    engine.Refinement = NativeRefinement
    if args.smoke and not spec['native']:
        config = read_json(args.run / 'config.json')
        torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        model = NativeRefinement(args.run, spec)
        optimizer = torch.optim.AdamW(model.params, lr=spec['teacher_lr'])
        before = [p.detach().clone() for p in model.params]
        generator = torch.Generator('cpu').manual_seed(config['seed'])
        measured = checked_teacher_step(model.named, optimizer, None,
            lambda backward: model.compute(list(range(config['teacher_batch'])),
                torch.randn((config['teacher_batch'],4,64,64), generator=generator),
                torch.full((config['teacher_batch'],),100),0,backward),
            args.run / 'smoke_failure.json', {'phase':'continuation_original_smoke'})
        changed = any(not torch.equal(a,b) for a,b in zip(before,model.params))
        if not changed:raise RuntimeError('Original-objective continuation did not update teacher')
        write_json(args.run / 'smoke.json', {'passed':True,'scientific_result':False,
                   'checkpoint_written':False,'parameters_changed':changed,**measured})
    else:
        engine.train(args.run, args.smoke)


if __name__ == '__main__':
    main()

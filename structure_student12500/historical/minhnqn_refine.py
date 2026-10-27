"""Isolated refinement entry point. Reuses pinned training/checkpoint machinery."""
import argparse
import copy
from pathlib import Path
import torch
import transfer_refine as engine
from transfer_refine import norm_of, add_bounded_gradient
from transfer_proxy import decode
from minhnqn_objectives import GradientProbe, patch_cka
from common import read_json, write_json, image_name
from semantic_teacher_train import threshold_teacher_backward, predicted_x0
from semantic_threshold import cfg_prediction


class MinhRefinement(engine.Refinement):
    def __init__(self, run, spec):
        super().__init__(run, spec)
        self.reference = None
        if spec['method'] == 'gradient':
            self.reference = copy.deepcopy(self.unet).eval().requires_grad_(False)
            self.reference.disable_gradient_checkpointing()
            self.gradient_probe = GradientProbe(self.assets['student'], spec['sketch_width'])

    def compute(self, ids, noise, t, step, backward, force_proxy=False):
        from semantic_teacher import pixels
        with torch.no_grad():
            z = self.ext.image_latent(pixels([self.run / 'images/teacher_clean_train' / image_name(i) for i in ids]))
            text = self.ext.text([self.rows[i]['text'] for i in ids])
            noisy = self.scheduler.add_noise(z, noise.cuda(), t.cuda())
        values = threshold_teacher_backward(self.unet, self.ext, self.carrier, self.scheduler,
                       noisy, text, t.cuda(), 1., self.c, step, backward, {})
        base_norm = norm_of([p.grad for p in self.params])
        active = force_proxy or (step + 1) % self.spec['method_every'] == 0
        generator = torch.Generator('cuda').manual_seed(self.c['seed'] + 500000 + step)
        j = step % len(ids)
        tt = torch.randint(10, 201, (1,), generator=generator, device='cuda')
        zz, hidden = z[j:j+1], text[j:j+1]
        empty = self.ext.text([''])
        xx = self.scheduler.add_noise(zz, torch.randn(zz.shape, generator=generator, device='cuda'), tt)

        def view(unet, checkpointed):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                conditional = unet(xx, tt, encoder_hidden_states=hidden).sample
                unconditional = unet(xx, tt, encoder_hidden_states=empty).sample
            return decode(self.ext.vae, predicted_x0(self.scheduler, xx,
                          cfg_prediction(conditional, unconditional), tt), checkpointed)

        with torch.no_grad():
            # Match the existing LPIPS-only branch's FP32 clean target.
            clean_prediction = cfg_prediction(self.ext.unet(xx, tt, encoder_hidden_states=hidden).sample,
                                              self.ext.unet(xx, tt, encoder_hidden_states=empty).sample)
            clean = decode(self.ext.vae, predicted_x0(self.scheduler, xx, clean_prediction, tt), False)
            reference = view(self.reference, False) if active and self.reference is not None else None
        marked = view(self.unet, True)
        quality = self.perceptual(marked, clean).mean()
        grads = torch.autograd.grad(self.spec['quality_weight'] * quality, self.params,
                                    retain_graph=active, allow_unused=True)
        quality_stats = add_bounded_gradient(self.params, grads, base_norm, self.spec['aux_gradient_fraction'])
        del grads
        values.update(quality_lpips=float(quality.detach()), quality_gradient=quality_stats,
                      original_gradient_norm=float(base_norm), method_active=active, quality_timestep=int(tt))
        values['loss'] += self.spec['quality_weight'] * float(quality.detach())
        if active:
            if self.spec['method'] == 'gradient':
                # Technical smoke deliberately leaves the equality point; no saved training state.
                candidate = (marked + .002 * torch.sin(marked * 7)).clamp(-1, 1) if force_proxy else marked
                loss, metrics = self.gradient_probe.objective(candidate, clean, reference,
                    self.rows[ids[j]]['text'], self.c['seed'] + 900000 + step)
                metrics['smoke_input_perturbation'] = bool(force_proxy)
            else:
                loss = patch_cka(self.perceptual, marked, clean, self.spec['cka_grid'])
                metrics = {'cka_loss': float(loss.detach())}
            grads = torch.autograd.grad(self.spec['method_weight'] * loss, self.params, allow_unused=True)
            stats = add_bounded_gradient(self.params, grads, base_norm, self.spec['aux_gradient_fraction'])
            if force_proxy and not stats['norm'] > 0:
                raise RuntimeError('New objective has no hypergradient to owner teacher')
            values.update(metrics, method_gradient=stats)
            values['loss'] += self.spec['method_weight'] * float(loss.detach())
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    engine.Refinement = MinhRefinement
    engine.train(args.run, args.smoke)
    if args.smoke:
        result = read_json(args.run / 'smoke.json')
        if not result['method_gradient']['norm'] > 0:
            raise RuntimeError('Auxiliary gradient verification failed')
        result['method_verified'] = read_json(args.run / 'refine_spec.json')['method']
        write_json(args.run / 'smoke.json', result)


if __name__ == '__main__':
    main()

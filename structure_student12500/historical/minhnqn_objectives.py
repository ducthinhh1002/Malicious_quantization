"""Owner-side adaptations of gradient sketching and centered feature alignment.

No dependency on or changes to the MinhNQN repository. These are new objectives,
not a reproduction of VLM distillation. Final student training is unchanged.
"""
import hashlib
import torch
import torch.nn.functional as F
from transfer_proxy import PairedLearner, prediction, encode, rgb8, unroll


def signed_sketch(vector, width=512, seed=42, chunk=65536):
    """CountSketch: preserve signed magnitude, O(P) work and bounded temporary RAM.

    Different from GVENDI's dense Rademacher projection. Same seeded hash/signs
    are used for clean/reference/candidate. No normalization of the gradient.
    """
    vector = vector.reshape(-1).float()
    generator = torch.Generator(vector.device).manual_seed(seed)
    result = vector.new_zeros(width)
    for start in range(0, vector.numel(), chunk):
        part = vector[start:start + chunk]
        index = torch.randint(width, (part.numel(),), generator=generator, device=part.device)
        signs = torch.randint(2, (part.numel(),), generator=generator, device=part.device) * 2 - 1
        result = result.scatter_add(0, index, part * signs)
    return result


def relative_delta_loss(candidate, reference, clean):
    target = (reference - clean).detach()
    delta = candidate - clean.detach()
    energy = target.square().sum().detach()
    loss = (delta - target).square().sum() / energy.clamp_min(1e-12)
    with torch.no_grad():
        metrics = {'gradient_reference_energy': float(energy),
                   'gradient_delta_cosine': float(F.cosine_similarity(delta[None], target[None], eps=1e-12)),
                   'gradient_delta_norm_ratio': float(delta.norm() / target.norm().clamp_min(1e-12))}
    return loss, metrics


def centered_cka(x, y):
    """Rows must be corresponding spatial patches. Does not match key or scale."""
    x, y = x.float(), y.detach().float()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError('CKA requires at least three corresponding patches')
    x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
    a, b = x @ x.T, y @ y.T
    denominator = a.square().sum().clamp_min(1e-12).sqrt() * b.square().sum().clamp_min(1e-12).sqrt()
    return 1 - (a * b).sum() / denominator


def patch_cka(perceptual, marked, clean, grid=8):
    # Reuse the frozen LPIPS AlexNet; no new model/checkpoint or resized RGB.
    with torch.no_grad():
        target = perceptual.net(perceptual.scaling_layer(clean))
    features = perceptual.net(perceptual.scaling_layer(marked))
    losses = []
    for layer in (1, 2):
        a = F.adaptive_avg_pool2d(features[layer], (grid, grid)).flatten(2).transpose(1, 2)
        b = F.adaptive_avg_pool2d(target[layer], (grid, grid)).flatten(2).transpose(1, 2)
        losses.extend(centered_cka(x, y) for x, y in zip(a, b))
    return torch.stack(losses).mean()


class GradientProbe(PairedLearner):
    """Same SD1.4 rank32 attention+conv probe for clean/reference/candidate.

    One detached clean AdamW warmup activates both LoRA factors. Then match
    signed changes in ordinary denoising gradients. No detector in this proxy.
    """
    def __init__(self, model, width=512):
        super().__init__(model, None, None, rank=32, inner_steps=1, query_steps=0)
        self.width = width

    def gradient_sketch(self, z, text, parameters, noise, timestep, differentiable):
        pred = prediction(self.pipe.unet, self.scheduler.add_noise(z, noise, timestep),
                          timestep, text, parameters, differentiable)
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(z, noise, timestep)
        else:
            raise ValueError('Unsupported ordinary denoising target')
        loss = (pred.float() - target.float()).square().mean()
        gradients = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=differentiable)
        sketches = []
        for (name, _), gradient in zip(parameters.items(), gradients):
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little') % (2**63 - 1)
            sketches.append(signed_sketch(gradient, self.width, seed))
        return torch.cat(sketches)

    def objective(self, marked, clean, reference, prompt, seed):
        generator = torch.Generator('cuda').manual_seed(seed)
        shape = (marked.shape[0], 4, 64, 64)
        posterior_noise = torch.randn(shape, generator=generator, device='cuda')
        text = self.text([prompt])
        with torch.no_grad():
            z_clean = encode(self.pipe.vae, rgb8(clean), posterior_noise, False)
            z_reference = encode(self.pipe.vae, rgb8(reference), posterior_noise, False)
        warm_noise = torch.randn(shape, generator=generator, device='cuda')
        warm_t = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), generator=generator, device='cuda')
        parameters = unroll(self.pipe.unet, self.initial, z_clean, text, self.scheduler,
                            [(warm_noise, warm_t)], False)
        noise = torch.randn(shape, generator=generator, device='cuda')
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), generator=generator, device='cuda')
        clean_sketch = self.gradient_sketch(z_clean, text, parameters, noise, timestep, False).detach()
        ref_sketch = self.gradient_sketch(z_reference, text, parameters, noise, timestep, False).detach()
        z_marked = encode(self.pipe.vae, rgb8(marked), posterior_noise)
        candidate = self.gradient_sketch(z_marked, text, parameters, noise, timestep, True)
        loss, metrics = relative_delta_loss(candidate, ref_sketch, clean_sketch)
        if metrics['gradient_reference_energy'] <= 1e-12:
            raise RuntimeError('Degenerate reference watermark gradient; inspect probe')
        return loss, {**metrics, 'gradient_matching_loss': float(loss.detach()),
                      'probe_timestep': int(timestep), 'probe_warmup_timestep': int(warm_t)}

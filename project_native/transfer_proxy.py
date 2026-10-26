"""Owner-only differentiable ordinary learner; fixed semantic verifier.

Adapted from the previously tested learning_channel learner/sampling primitives.
The final rank32-conv student remains an independent upstream training process.
"""
import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


def prediction(unet, z, t, text, parameters, checkpointed=True):
    names = tuple(parameters)
    def call(x, time, hidden, *weights):
        return functional_call(unet, dict(zip(names, weights)), (x, time),
                               {'encoder_hidden_states': hidden}, strict=False).sample
    args = (z, t, text, *parameters.values())
    return checkpoint(call, *args, use_reentrant=False) if checkpointed else call(*args)


def adamw(parameters, gradients, state, step, lr=1e-4):
    # Same AdamW defaults/clipping as the final trainer; FP32 proxy arithmetic.
    norm = torch.stack([g.double().square().sum() for g in gradients]).sum().clamp_min(1e-24).sqrt()
    factor = (1. / (norm + 1e-6)).clamp(max=1.).float()
    updated, next_state = {}, {}
    for (name, p), g in zip(parameters.items(), gradients):
        g = g * factor
        old_m, old_v = state.get(name, (torch.zeros_like(p), torch.zeros_like(p)))
        m, v = .9 * old_m + .1 * g, .999 * old_v + .001 * g.square()
        # Declared floor regularizes the otherwise undefined sqrt derivative at zero.
        denominator = (v / (1 - .999**step)).clamp_min(1e-24).sqrt() + 1e-8
        updated[name] = p * (1 - lr * .01) - lr * (m / (1 - .9**step)) / denominator
        next_state[name] = (m, v)
    return updated, next_state


def unroll(unet, initial, z, text, scheduler, draws, create_graph):
    params = {k: v.detach().clone().requires_grad_(True) for k, v in initial.items()}
    state = {}
    for step, (noise, t) in enumerate(draws, 1):
        noisy = scheduler.add_noise(z, noise, t)
        if scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif scheduler.config.prediction_type == 'v_prediction':
            target = scheduler.get_velocity(z, noise, t)
        else:
            raise ValueError('Unsupported ordinary learner target')
        pred = prediction(unet, noisy, t, text, params, create_graph)
        loss = (pred.float() - target.float()).square().mean()
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph)
        params, state = adamw(params, gradients, state, step)
        if not create_graph:
            params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
            state = {k: tuple(v.detach() for v in pair) for k, pair in state.items()}
    return params


def rgb8(x):
    # Forward really crosses the RGB8 channel; backward uses an explicitly declared STE.
    x = x.clamp(-1, 1)
    quantized = ((x + 1) * 127.5).round() / 127.5 - 1
    return x + (quantized - x).detach()


def decode(vae, z, checkpointed=True):
    def call(value):
        return vae.decode(value / vae.config.scaling_factor).sample
    return (checkpoint(call, z, use_reentrant=False) if checkpointed else call(z)).clamp(-1, 1)


def encode(vae, rgb, noise, checkpointed=True):
    def call(value):
        posterior = vae.encode(value).latent_dist
        return posterior.mean, posterior.std
    mean, std = checkpoint(call, rgb, use_reentrant=False) if checkpointed else call(rgb)
    return (mean + std * noise) * vae.config.scaling_factor


def trajectory(pipe, params, noise, text, steps=8, checkpointed=True):
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    scheduler.set_timesteps(steps, device=noise.device)
    z = noise * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(torch.cat((z, z)), t)
        u, c = prediction(pipe.unet, x, t, text, params, checkpointed).chunk(2)
        z = scheduler.step(u + 7.5 * (c - u), t, z, eta=0.).prev_sample
    return z


class PairedLearner:
    def __init__(self, model, ext, carrier, rank=8, inner_steps=2, query_steps=8):
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from peft import LoraConfig
        from semantic_teacher_train import adapter_targets
        self.ext, self.carrier = ext, carrier
        self.inner_steps, self.query_steps = inner_steps, query_steps
        self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float32,
                         safety_checker=None, local_files_only=True).to('cuda')
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        for model in (self.pipe.vae, self.pipe.text_encoder, self.pipe.unet):
            model.eval().requires_grad_(False)
        self.pipe.unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian',
                                             target_modules=adapter_targets(self.pipe.unet)))
        self.initial = {n: p.detach().clone() for n, p in self.pipe.unet.named_parameters() if p.requires_grad}
        self.pipe.unet.requires_grad_(False)
        if not self.initial or not any('lora_B' in n and p.ndim == 4 for n, p in self.initial.items()):
            raise RuntimeError('Expected attention AND convolution surrogate LoRA')

    def text(self, prompts):
        tokens = self.pipe.tokenizer(prompts, padding='max_length', truncation=True,
                    max_length=self.pipe.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            return self.pipe.text_encoder(tokens.input_ids.cuda())[0]

    def objective(self, marked, clean, train_prompt, query_prompt, seed):
        if train_prompt == query_prompt:
            raise ValueError('Transfer query must be a different training-pool prompt')
        g = torch.Generator('cuda').manual_seed(seed)
        shape = (1, 4, 64, 64)
        noise = torch.randn(shape, generator=g, device='cuda')
        with torch.no_grad():
            z_clean = encode(self.pipe.vae, rgb8(clean), noise, False)
        z_marked = encode(self.pipe.vae, rgb8(marked), noise)
        text = self.text([train_prompt])
        draws = [(torch.randn(shape, generator=g, device='cuda'),
                  torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), generator=g, device='cuda'))
                 for _ in range(self.inner_steps)]
        clean_params = unroll(self.pipe.unet, self.initial, z_clean, text, self.scheduler, draws, False)
        query_text = self.text(['', query_prompt])
        query_noise = torch.randn(shape, generator=g, device='cuda')
        detector_text = self.ext.text([query_prompt])
        with torch.no_grad():
            clean_out = rgb8(decode(self.pipe.vae, trajectory(self.pipe, clean_params, query_noise,
                                      query_text, self.query_steps, False), False))
            q_clean = self.carrier.coordinates(self.ext.encode_latent(self.ext.image_latent(clean_out),
                                                                     text_hidden=detector_text))
        del clean_params, clean_out
        marked_params = unroll(self.pipe.unet, self.initial, z_marked, text, self.scheduler, draws, True)
        marked_out = rgb8(decode(self.pipe.vae, trajectory(self.pipe, marked_params, query_noise,
                                                        query_text, self.query_steps)))
        q_marked = self.carrier.coordinates(self.ext.encode_latent(self.ext.image_latent(marked_out),
                                                                  text_hidden=detector_text))
        gain = ((q_marked - q_clean) * self.carrier.signs).mean()
        # Bounded objective targets a shared signed gain, not low raw denoising MSE.
        # 0.1 sets units of the surrogate gain, not the final detection threshold.
        loss = torch.nn.functional.softplus(-gain / .1)
        return loss, {'proxy_score_gain': gain.detach().item(), 'proxy_loss': loss.detach().item(),
                      'inner_updates': self.inner_steps, 'query_steps': self.query_steps}

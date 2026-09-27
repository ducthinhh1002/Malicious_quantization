"""Owner-independent timestep coverage, context augmentation, and loss scaling."""
import string
import torch


class TimestepSampler:
    def __init__(self, dataset, rng):
        self.rng = rng
        self.groups = {}
        for i, row in enumerate(dataset):
            self.groups.setdefault(int(row[3]), []).append(i)
        self.order = []

    def sample(self, count):
        result = []
        for _ in range(count):
            if not self.order:
                self.order = list(self.rng.permutation(sorted(self.groups)))
            timestep = self.order.pop()
            result.append(int(self.rng.choice(self.groups[timestep])))
        return result


def augmented_prompts(prompts, rng):
    # Distribution fixed before owner evaluation, no secret or recovered tokens.
    # Broader conditioning coverage is not evidence of activating a watermark.
    return [''.join(rng.choice(list(string.punctuation), size=int(rng.integers(1, 9))))
            + ' ' + prompt for prompt in prompts]


def spatial_reconstruction_loss(student_x0, target_x0, timestep, scheduler,
                                guidance=7.5, mode='noise', floor=.01):
    error = (student_x0-target_x0).square().flatten(1).mean(1)
    if mode == 'x0':
        return error.mean()
    alpha = scheduler.alphas_cumprod.to(student_x0.device, student_x0.dtype)[timestep.long()]
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        jacobian_squared = (1-alpha)/alpha.clamp_min(1e-6)
    elif kind == 'v_prediction':
        jacobian_squared = 1-alpha
    elif kind == 'sample':
        jacobian_squared = torch.ones_like(alpha)
    else:
        raise ValueError(f'Unsupported prediction type: {kind}')
    if mode != 'noise':
        raise ValueError('Unknown spatial loss normalization')
    return (error/jacobian_squared.clamp_min(floor)/guidance**2).mean()

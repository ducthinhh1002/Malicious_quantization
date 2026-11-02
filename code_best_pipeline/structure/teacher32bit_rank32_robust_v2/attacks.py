"""Frozen mixed-setting panel from the best 8-bit table, now tested on 32 bits.

RGB8 uses truncation (ToPILImage), not rounding. Training uses an STE only for
that quantizer; evaluation is the exact non-differentiable channel. No JPEG/SP.
"""
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode as I

ATTACKS = ('clean', 'horizontal_flip', 'rotation', 'crop_scale', 'random_drop',
           'gaussian_noise', 'brightness', 'gaussian_blur')
CONFIG = {
    'version': 2, 'panel': 'teacher_manipulation_BEST_no_salt_20261030',
    'clean': {'channel': 'tensor'},
    'horizontal_flip': {'fixed': True, 'channel': 'tensor', 'seed_base': 20261030},
    'rotation': {'degrees': [-90, 90], 'interpolation': 'nearest', 'fill': .5,
                 'channel': 'rgb8_truncate', 'seed_base': 20261029},
    'crop_scale': {'area': .75, 'ratio': 1, 'position': 'random',
                   'interpolation': 'bilinear', 'antialias': True,
                   'channel': 'tensor', 'seed_base': 20261030},
    'random_drop': {'keep_pixel_probability': .64, 'fill': .5,
                    'shared_rgb_mask': True, 'channel': 'tensor', 'seed_base': 20261030},
    'gaussian_noise': {'sigma': .1, 'clip': [0, 1], 'channel': 'tensor',
                       'seed_base': 20261030, 'independent_per_image': True},
    'brightness': {'factor': .7, 'channel': 'rgb8_truncate', 'seed_base': 20261030},
    'gaussian_blur': {'kernel': 15, 'sigma': [.1, 2.], 'channel': 'rgb8_truncate',
                      'seed_base': 20261029},
    'jpeg': False, 'salt_pepper': False,
}

def quantize(x, ste=False):
    x = x.clamp(0, 1)
    y = (x * 255).to(torch.uint8).to(x.dtype) / 255
    return x + (y-x).detach() if ste else y

def apply_attack(x, name, training=False, progress=1.):
    if name not in ATTACKS:
        raise ValueError(name)
    if name == 'horizontal_flip':
        x = F.hflip(x)
    elif name == 'rotation':
        # Curriculum changes training only; test is always full +/-90 degrees.
        limit = 15 + 75 * min(1., progress) if training else 90
        angle = T.RandomRotation.get_params([-limit, limit])
        x = F.rotate(x, angle, interpolation=I.NEAREST, fill=.5)
    elif name == 'crop_scale':
        h, w = x.shape[-2:]
        ch, cw = round(h * .75**.5), round(w * .75**.5)
        top = int(torch.randint(h-ch+1, ()).item())
        left = int(torch.randint(w-cw+1, ()).item())
        x = F.resized_crop(x, top, left, ch, cw, [h, w], I.BILINEAR, antialias=True)
    elif name == 'random_drop':
        keep = torch.rand((1, *x.shape[-2:]), device=x.device) < .64
        x = torch.where(keep, x, .5)
    elif name == 'gaussian_noise':
        x = x + torch.randn_like(x) * .1
    elif name == 'brightness':
        x = F.adjust_brightness(x, .7)
    elif name == 'gaussian_blur':
        sigma = T.GaussianBlur.get_params(.1, 2.)
        x = F.gaussian_blur(x, [15, 15], sigma)
    x = x.clamp(0, 1)
    if CONFIG[name]['channel'] == 'rgb8_truncate':
        x = quantize(x, ste=training)
    return x

def eval_attack(x, name, index):
    # Exact old per-image random draws on CPU, shared by before/after.
    if x.device.type != 'cpu':
        raise ValueError('Evaluation transforms must run on CPU for historical RNG parity')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CONFIG[name].get('seed_base', 20261030) + index)
        return apply_attack(x, name)

def training_pair(step, sample_index):
    # Two geometry views for every one flip view; each pair has a photometric view.
    geometry = ('rotation', 'crop_scale', 'rotation', 'crop_scale', 'horizontal_flip')
    photometric = ('random_drop', 'gaussian_noise', 'brightness', 'gaussian_blur')
    return geometry[(step+sample_index) % len(geometry)], photometric[(step+sample_index) % len(photometric)]


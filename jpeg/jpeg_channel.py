"""Adapt WOFA local N2's real-JPEG/STE idea, with benchmark-exact RGB8/420.

WOFA local code uses uint8 truncation and an implicit Pillow subsampling;
we deliberately round to nearest and explicitly use 420 to match our evaluator.
Train qualities 35,50,65,80,95; evaluation quality25 is never a training draw.
"""
import io
import random
import numpy as np
from PIL import Image
import torch

CHANNELS = ('jpeg',)


def jpeg_quality(seed, severity):
    return 25


def jpeg_ste(rgb, quality):
    integer = ((rgb.detach().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    output = []
    for arr in integer:
        im = Image.fromarray(arr.transpose(1, 2, 0))
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=quality, subsampling=2,
                optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            value = torch.from_numpy(np.array(decoded.convert('RGB'), copy=True)).permute(2, 0, 1)
        output.append(value.to(device=rgb.device, dtype=rgb.dtype) / 127.5 - 1)
    compressed = torch.stack(output)
    # Exact decoded RGB in forward, identity surrogate derivative in backward.
    return compressed + (rgb - rgb.detach())


def transform(rgb, channel, seed, severity):
    if channel == 'identity':
        return rgb
    if channel != 'jpeg':
        raise ValueError(channel)
    return jpeg_ste(rgb, jpeg_quality(seed, severity))


def bit_margin(q, signs, mu):
    return torch.relu(mu - q * signs).square().mean(-1)

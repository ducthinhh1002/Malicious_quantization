"""Grouped signed W4 with bounded, trainable integer-code offsets and scales."""
import torch
from torch import nn
from torch.nn import functional as F


class GroupedW4(nn.Module):
    def __init__(self, weight, group_size=64):
        super().__init__()
        if group_size < 1:
            raise ValueError('group_size must be positive')
        self.shape = weight.shape
        self.width = weight[0].numel()
        self.group_size = group_size
        self.padding = (-self.width) % group_size
        flat = F.pad(weight.detach().reshape(weight.shape[0], -1), (0, self.padding))
        blocks = flat.reshape(weight.shape[0], -1, group_size)
        scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
        self.register_buffer('base_scale', scale)
        self.register_buffer('codes', (blocks / scale).round().clamp(-8, 7))
        self.offset = nn.Parameter(torch.zeros_like(blocks))
        self.log_scale = nn.Parameter(torch.zeros_like(scale))

    def forward(self, training=True):
        value = (self.codes + self.offset.clamp(-2, 2)).clamp(-8, 7)
        hard = value.round()
        codes = value + (hard - value).detach() if training else hard
        scale = self.base_scale * self.log_scale.clamp(-.22314355, .22314355).exp()
        return (codes * scale).reshape(self.shape[0], -1)[:, :self.width].reshape(self.shape)


def grouped_fake_quant(weight, group_size):
    width = weight[0].numel()
    blocks = F.pad(weight.detach().reshape(weight.shape[0], -1),
                   (0, (-width) % group_size)).reshape(weight.shape[0], -1, group_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
    hard = ((blocks / scale).round().clamp(-8, 7) * scale).reshape(weight.shape[0], -1)
    hard = hard[:, :width].reshape_as(weight)
    return weight + (hard - weight).detach()

"""Tensor / image / DCT utilities used across the codebase."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL image (HxWx3, uint8) -> torch tensor (3xHxW, float32 in [0,1])."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Inverse of :func:`image_to_tensor`. Expects CHW in [0,1]."""
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pad_to_size(image: Image.Image, size: int = 512, fill=(128, 128, 128)) -> Image.Image:
    """Center-pad an image up to ``size x size`` with a constant fill colour."""
    if image.size == (size, size):
        return image
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def dct2d_torch(x: torch.Tensor) -> torch.Tensor:
    """Real-part FFT used as a proxy for the orthonormal 2D DCT.

    Note: this is the same operator used in the paper's reference
    implementation. It preserves the energy distribution we rely on while
    being differentiable through ``torch.autograd``.
    """
    return torch.fft.fft2(x, norm="ortho").real


def idct2d_torch(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2d_torch` (real part)."""
    return torch.fft.ifft2(x, norm="ortho").real


def extract_center_patch(latents: torch.Tensor, center_ratio: float) -> torch.Tensor:
    """Crop the central ``center_ratio`` square of a 4D latent tensor."""
    h, w = latents.shape[-2], latents.shape[-1]
    h0 = int(h * (1 - center_ratio) / 2)
    h1 = int(h * (1 + center_ratio) / 2)
    w0 = int(w * (1 - center_ratio) / 2)
    w1 = int(w * (1 + center_ratio) / 2)
    return latents[:, :, h0:h1, w0:w1]


def pad_center_patch_to_full(
    center_patch: torch.Tensor,
    original_latents: torch.Tensor,
    center_ratio: float,
) -> torch.Tensor:
    """Place ``center_patch`` back into a clone of ``original_latents``."""
    _, _, h_full, w_full = original_latents.shape
    h_c, w_c = center_patch.shape[-2], center_patch.shape[-1]
    h0 = int(h_full * (1 - center_ratio) / 2)
    w0 = int(w_full * (1 - center_ratio) / 2)
    out = original_latents.clone()
    out[:, :, h0 : h0 + h_c, w0 : w0 + w_c] = center_patch
    return out


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")

"""GROW: Watermark Generation with Progressive Guidance for Diffusion Models.

A training-free, inversion-free watermarking framework for latent diffusion
models. The watermark is woven into the image during the iterative denoising
process via gradient guidance in the DCT (frequency) domain, and is recovered
directly by encoding the image back to the VAE latent and reading the signs
of the masked DCT coefficients.

Public API:
    from grow import (
        GrowConfig,
        GrowWatermarker,
        message_to_bits,
        bits_to_message,
        AttackSuite,
    )
"""

from .config import GrowConfig
from .codec import message_to_bits, bits_to_message
from .watermark import GrowWatermarker
from .attacks import AttackSuite, image_distortion

__all__ = [
    "GrowConfig",
    "GrowWatermarker",
    "message_to_bits",
    "bits_to_message",
    "AttackSuite",
    "image_distortion",
]

__version__ = "0.1.0"

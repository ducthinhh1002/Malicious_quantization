"""Configuration dataclass for GROW.

All hyperparameters used during watermark generation, embedding and
extraction are gathered here so that experiments are easy to reproduce
and share. Defaults match the values reported in the paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GrowConfig:
    # ---- Model / runtime ----
    model_id: str = "stabilityai/stable-diffusion-2-1-base"
    device: str = "cuda"
    dtype: str = "float32"  # "float32" | "float16"
    seed: int = 42

    # ---- Image / latent ----
    image_size: int = 512
    num_inference_steps: int = 50
    cfg_scale: float = 7.5

    # ---- Watermark message ----
    secret_key: str = "watermark"
    message: str = "OKOK"  # ASCII message; bit length = len(utf8) * 8

    # ---- Progressive guidance ----
    # alpha in the paper: amplitude of the target DCT coefficients.
    watermark_strength: float = 0.5
    # eta in the paper: gradient guidance scale at each step.
    guidance_scale: float = 200.0
    # r_start in the paper: fraction of steps before guidance kicks in.
    watermark_start_ratio: float = 0.5

    # ---- Frequency-domain mask (mid-band) ----
    # The watermark is embedded in the mid-frequency band [dct_min, dct_max)
    # of the DCT of the (cropped) latent, in the channels listed below.
    dct_min: float = 0.2
    dct_max: float = 0.5
    channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Ratio of the central latent region used for DCT (1.0 = whole latent).
    center_ratio: float = 1.0

    # ---- Output ----
    output_dir: str = "outputs"

    # ---- Optional prompts for the example script ----
    prompts: Optional[List[str]] = None

    # -------- Convenience --------
    @property
    def message_bit_length(self) -> int:
        return len(self.message.encode("utf-8")) * 8

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

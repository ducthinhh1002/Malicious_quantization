"""Attack suite used to evaluate watermark robustness.

The attacks reproduced here mirror the ones reported in Table 1 of the
paper: rotation, JPEG compression, crop & scale, Gaussian blur, additive
Gaussian noise and brightness shift.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFilter

from .utils import set_random_seed


def image_distortion(
    img: Image.Image,
    seed: int = 0,
    r_degree: Optional[float] = None,
    jpeg_ratio: Optional[int] = None,
    crop_scale: Optional[float] = None,
    crop_ratio: Optional[float] = None,
    gaussian_blur_r: Optional[float] = None,
    gaussian_std: Optional[float] = None,
    brightness_factor: Optional[float] = None,
) -> Image.Image:
    """Apply one or more standard image-domain distortions to ``img``."""
    out = img.copy()

    if r_degree is not None:
        out = T.RandomRotation((r_degree, r_degree))(out)

    if jpeg_ratio is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=jpeg_ratio)
        out = Image.open(buf)

    if crop_scale is not None and crop_ratio is not None:
        set_random_seed(seed)
        out = T.Compose(
            [
                T.CenterCrop(
                    (int(out.size[0] * crop_scale), int(out.size[1] * crop_scale))
                ),
                T.Resize(out.size),
            ]
        )(out)

    if gaussian_blur_r is not None:
        out = out.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

    if gaussian_std is not None:
        shape = np.array(out).shape
        noise = (np.random.normal(0, gaussian_std, shape) * 255).astype(np.uint8)
        out = Image.fromarray(np.clip(np.array(out) + noise, 0, 255))

    if brightness_factor is not None:
        out = T.ColorJitter(brightness=(0.2, brightness_factor))(out)

    return out


# Default attack profile used for evaluation tables.
DEFAULT_ATTACKS: Dict[str, Dict] = {
    "Clean": {},
    "Rot 75": {"r_degree": 75},
    "JPEG 25": {"jpeg_ratio": 25},
    "C&S 75": {"crop_scale": 0.75, "crop_ratio": 1},
    "Blur 4": {"gaussian_blur_r": 4},
    "Noise 0.05": {"gaussian_std": 0.05},
    "Brightness 6": {"brightness_factor": 6},
}


@dataclass
class AttackSuite:
    """Convenience wrapper to evaluate a watermarker against many attacks."""

    extractor: Callable[[Image.Image, str], str]
    """Callable ``(image, attack_name) -> extracted_message``.

    Typical usage in :func:`scripts/run_demo.py` is to dispatch to
    ``GrowWatermarker.extract`` for most attacks and to the brute-force
    helpers for rotation / crop&scale.
    """

    expected_message: str
    output_dir: Optional[str] = None
    attacks: Dict[str, Dict] = None

    def __post_init__(self):
        if self.attacks is None:
            self.attacks = DEFAULT_ATTACKS

    def run(self, image: Image.Image, seed: int = 0):
        results = {}
        success = 0
        for name, params in self.attacks.items():
            distorted = image_distortion(image, seed=seed, **params)
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                distorted.save(
                    os.path.join(self.output_dir, f"attack_{name.replace(' ', '_')}.png")
                )
            extracted = self.extractor(distorted, name)
            ok = extracted == self.expected_message
            results[name] = {"message": extracted, "success": ok}
            success += int(ok)

        total = len(self.attacks)
        accuracy = 100.0 * success / total
        return results, accuracy

    @staticmethod
    def pretty_print(results: Dict[str, Dict], accuracy: float) -> None:
        print("\n| Attack Type      | Extracted Message      | Success |")
        print("|------------------|------------------------|---------|")
        for name, r in results.items():
            mark = "[OK] Yes" if r["success"] else "[X] No"
            msg = f"'{r['message']}'"
            print(f"| {name:<16} | {msg:<22} | {mark:<7} |")
        print(f"\nOverall Accuracy: {accuracy:.2f}%")

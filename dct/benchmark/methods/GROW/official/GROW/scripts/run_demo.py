"""End-to-end demo: generate a watermarked image and evaluate robustness.

Example
-------
    python scripts/run_demo.py \
        --prompt "a sharp wildlife photo of an owl with feather details" \
        --message OKOK \
        --output-dir outputs/owl
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from PIL import Image

from grow import AttackSuite, GrowConfig, GrowWatermarker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GROW watermark demo")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--message", type=str, default="OKOK")
    p.add_argument("--secret-key", type=str, default="watermark")
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--strength", type=float, default=0.5, help="alpha in the paper")
    p.add_argument("--guidance", type=float, default=200.0, help="eta in the paper")
    p.add_argument("--start-ratio", type=float, default=0.5, help="r_start in the paper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/demo")
    p.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu (auto-detect if omitted)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    cfg = GrowConfig(
        model_id=args.model_id,
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        secret_key=args.secret_key,
        message=args.message,
        watermark_strength=args.strength,
        guidance_scale=args.guidance,
        watermark_start_ratio=args.start_ratio,
        output_dir=args.output_dir,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[GROW] message='{cfg.message}' bits={cfg.message_bit_length} channels={cfg.channels}")
    wm = GrowWatermarker(cfg)

    print("\n[1/3] Generating non-watermarked reference image ...")
    ref = wm.generate_normal(args.prompt, seed=cfg.seed)
    ref.save(os.path.join(cfg.output_dir, "reference.png"))

    print("\n[2/3] Generating GROW-watermarked image ...")
    wm_image = wm.generate_with_watermark(args.prompt, seed=cfg.seed)
    wm_image.save(os.path.join(cfg.output_dir, "watermarked.png"))

    print("\n[3/3] Evaluating extraction robustness ...")

    def extractor(distorted: Image.Image, attack_name: str) -> str:
        if attack_name == "Rot 75":
            return wm.extract_robust_rotation(distorted)
        if attack_name == "C&S 75":
            return wm.extract_robust_crop_scale(distorted)
        return wm.extract(distorted)

    suite = AttackSuite(
        extractor=extractor,
        expected_message=cfg.message,
        output_dir=cfg.output_dir,
    )
    results, accuracy = suite.run(wm_image, seed=cfg.seed)
    suite.pretty_print(results, accuracy)


if __name__ == "__main__":
    main()

"""Stand-alone extraction script.

Loads a (possibly attacked) image from disk and recovers the embedded
watermark using GROW's inversion-free extractor. Useful for verification
on third-party images.

Example
-------
    python scripts/extract.py \
        --image outputs/demo/watermarked.png \
        --message OKOK \
        --secret-key watermark
"""

from __future__ import annotations

import argparse

from PIL import Image

from grow import GrowConfig, GrowWatermarker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract a GROW watermark from an image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--message", type=str, required=True, help="Expected message length / value (used for confidence)")
    p.add_argument("--secret-key", type=str, default="watermark")
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--robust", choices=["none", "rotation", "crop_scale"], default="none")
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
        secret_key=args.secret_key,
        message=args.message,
    )
    wm = GrowWatermarker(cfg)
    image = Image.open(args.image).convert("RGB")

    if args.robust == "rotation":
        msg = wm.extract_robust_rotation(image)
    elif args.robust == "crop_scale":
        msg = wm.extract_robust_crop_scale(image)
    else:
        msg = wm.extract(image)

    print(f"Extracted message: '{msg}'")
    print(f"Expected   message: '{args.message}'")
    print(f"Match: {msg == args.message}")


if __name__ == "__main__":
    main()

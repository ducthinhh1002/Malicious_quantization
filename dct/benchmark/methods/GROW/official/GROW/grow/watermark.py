"""Core implementation of GROW.

Reference
---------
Luo et al., "GROW: Watermark Generation with Progressive Guidance for
Diffusion Models."

Overview
--------
This module exposes a single high-level class, :class:`GrowWatermarker`,
that wraps a Hugging Face :class:`~diffusers.StableDiffusionPipeline`
and implements the three operations of Algorithm 1 of the paper:

1. **Target construction.** Build a deterministic mid-frequency DCT
   target ``S`` and binary mask ``M`` from the secret key and payload.
2. **Progressive-guidance generation.** At every denoising step after
   ``r_start``, compute a frequency-domain MSE between the DCT of the
   predicted clean latent ``\\hat z_0`` and ``S``, backprop a single
   gradient step, and convert the guided ``\\hat z_0`` back into an
   equivalent noise prediction for the scheduler (Eq. 1 / 5-7).
3. **Inversion-free extraction.** VAE-encode the image, DCT the latent,
   read the sign of each masked coefficient, and majority-vote across
   redundant copies.

The class is side-effect free (never writes to disk) — all I/O is left
to calling scripts under :mod:`scripts`.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm

from .codec import bits_to_message, message_to_bits
from .config import GrowConfig
from .utils import (
    dct2d_torch,
    extract_center_patch,
    idct2d_torch,
    image_to_tensor,
    pad_center_patch_to_full,
    parse_dtype,
    tensor_to_image,
)


class GrowWatermarker:
    """High-level API for GROW watermark generation and extraction.

    Parameters
    ----------
    cfg : GrowConfig
        Hyper-parameter bundle. See :class:`grow.config.GrowConfig`.
    pipe : StableDiffusionPipeline, optional
        Pre-instantiated diffusion pipeline. If ``None``, a fresh
        pipeline is built from ``cfg.model_id`` using a
        :class:`~diffusers.DDIMScheduler`.
    """

    def __init__(
        self,
        cfg: GrowConfig,
        pipe: Optional[StableDiffusionPipeline] = None,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = parse_dtype(cfg.dtype)

        if pipe is None:
            scheduler = DDIMScheduler.from_pretrained(
                cfg.model_id, subfolder="scheduler"
            )
            pipe = StableDiffusionPipeline.from_pretrained(
                cfg.model_id,
                scheduler=scheduler,
                torch_dtype=self.dtype,
            ).to(self.device)

        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    # ------------------------------------------------------------------
    # Target signal / mask construction  (Eq. 4)
    # ------------------------------------------------------------------
    def build_target_signal(
        self,
        center_h: int,
        center_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the DCT target ``S`` and binary mask ``M``.

        The secret key seeds (i) the shuffle of mid-band coordinates and
        (ii) the per-coordinate bit assignment, ensuring deterministic
        reconstruction at extraction time.
        """
        cfg = self.cfg
        key_seed = sum(ord(c) for c in cfg.secret_key)
        torch.manual_seed(key_seed)

        h0, h1 = int(center_h * cfg.dct_min), int(center_h * cfg.dct_max)
        w0, w1 = int(center_w * cfg.dct_min), int(center_w * cfg.dct_max)
        capacity = (h1 - h0) * (w1 - w0)
        if capacity <= 0:
            raise ValueError(
                f"Empty frequency band: center={center_h}x{center_w}, "
                f"dct_min={cfg.dct_min}, dct_max={cfg.dct_max}"
            )

        coords = [(h, w) for h in range(h0, h1) for w in range(w0, w1)]
        np.random.default_rng(key_seed).shuffle(coords)

        bits = message_to_bits(cfg.message)
        if not bits:
            zero = torch.zeros(
                1, 4, center_h, center_w, device=self.device, dtype=self.dtype
            )
            return zero, torch.zeros_like(zero, dtype=torch.bool)

        target = torch.zeros(
            1, 4, center_h, center_w, device=self.device, dtype=self.dtype
        )
        mask = torch.zeros_like(target, dtype=torch.bool)

        n_ch = len(cfg.channels)
        bits_per_ch = len(bits) // n_ch
        remainder = len(bits) % n_ch

        for i, ch in enumerate(cfg.channels):
            start = i * bits_per_ch + min(i, remainder)
            end = (i + 1) * bits_per_ch + min(i + 1, remainder)
            ch_bits = bits[start:end]
            if not ch_bits:
                continue

            # Map {0, 1} -> {-1, +1} and tile across the full budget.
            signal = (
                torch.tensor(ch_bits, device=self.device, dtype=self.dtype) * 2 - 1
            )
            tiled = signal.repeat(capacity // len(ch_bits) + 1)[:capacity]

            for j, (h, w) in enumerate(coords[:capacity]):
                target[0, ch, h, w] = tiled[j] * cfg.watermark_strength
                mask[0, ch, h, w] = True

        return target, mask

    # ------------------------------------------------------------------
    # Text conditioning
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return a classifier-free embedding `[uncond; cond]` for ``prompt``."""
        tok_text = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok_uncond = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_emb = self.text_encoder(tok_text.input_ids.to(self.device))[0]
            uncond_emb = self.text_encoder(tok_uncond.input_ids.to(self.device))[0]
        return torch.cat([uncond_emb, text_emb])

    # ------------------------------------------------------------------
    # Vanilla (non-watermarked) generation — useful as a clean reference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_normal(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Run text-to-image without any watermark guidance."""
        cfg = self.cfg
        seed = cfg.seed if seed is None else seed
        gen = torch.Generator(device=self.device).manual_seed(seed)
        embeddings = self._encode_prompt(prompt)

        latents = torch.randn(
            (1, self.unet.config.in_channels, cfg.latent_size, cfg.latent_size),
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma
        self.pipe.scheduler.set_timesteps(cfg.num_inference_steps)

        for t in tqdm(self.pipe.scheduler.timesteps, desc="Reference"):
            model_in = torch.cat([latents] * 2)
            model_in = self.pipe.scheduler.scale_model_input(model_in, t)
            noise_pred = self.unet(model_in, t, encoder_hidden_states=embeddings).sample
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + cfg.cfg_scale * (cond - uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return tensor_to_image((image / 2 + 0.5).squeeze(0))

    # ------------------------------------------------------------------
    # Progressive-guidance watermarked generation  (Algorithm 1)
    # ------------------------------------------------------------------
    def generate_with_watermark(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a GROW-watermarked image for ``prompt``."""
        cfg = self.cfg
        seed = cfg.seed if seed is None else seed
        gen = torch.Generator(device=self.device).manual_seed(seed)
        embeddings = self._encode_prompt(prompt)

        center_h = int(cfg.latent_size * cfg.center_ratio)
        center_w = int(cfg.latent_size * cfg.center_ratio)
        target, mask = self.build_target_signal(center_h, center_w)

        latents = torch.randn(
            (1, self.unet.config.in_channels, cfg.latent_size, cfg.latent_size),
            generator=gen,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma
        self.pipe.scheduler.set_timesteps(cfg.num_inference_steps)

        wm_start_step = int(cfg.num_inference_steps * cfg.watermark_start_ratio)

        bar = tqdm(
            enumerate(self.pipe.scheduler.timesteps),
            total=len(self.pipe.scheduler.timesteps),
            desc="GROW",
        )
        for step_idx, t in bar:
            model_in = torch.cat([latents] * 2)
            model_in = self.pipe.scheduler.scale_model_input(model_in, t)

            with torch.no_grad():
                noise_pred = self.unet(model_in, t, encoder_hidden_states=embeddings).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)

            if step_idx >= wm_start_step:
                # --- Eq. 1: predicted clean latent z0_hat ---
                alpha_t = self.pipe.scheduler.alphas_cumprod[t]
                beta_t = 1.0 - alpha_t
                _, model_in_text = model_in.chunk(2)
                pred_x0 = (model_in_text - beta_t.sqrt() * noise_cond) / alpha_t.sqrt()
                pred_x0.requires_grad_(True)

                # --- Eq. 5: frequency-domain MSE watermark loss ---
                center = extract_center_patch(pred_x0, cfg.center_ratio)
                center_dct = dct2d_torch(center)

                if center_dct.shape[-2:] != target.shape[-2:]:
                    target, mask = self.build_target_signal(
                        center_dct.shape[-2], center_dct.shape[-1]
                    )

                loss = F.mse_loss(center_dct[mask], target[mask])

                # --- Eq. 6: single-step gradient guidance on z0_hat ---
                grad = torch.autograd.grad(loss, pred_x0)[0]
                pred_x0_guided = pred_x0 - cfg.guidance_scale * grad

                # --- Eq. 7: convert back to an effective noise prediction ---
                noise_cond_guided = (
                    model_in_text - alpha_t.sqrt() * pred_x0_guided
                ) / beta_t.sqrt()
                noise_final = noise_uncond + cfg.cfg_scale * (
                    noise_cond_guided - noise_uncond
                )
                bar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                noise_final = noise_uncond + cfg.cfg_scale * (noise_cond - noise_uncond)

            latents = self.pipe.scheduler.step(noise_final, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return tensor_to_image((image / 2 + 0.5).squeeze(0))

    # ------------------------------------------------------------------
    # Inversion-free extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _read_bits(self, image: Image.Image) -> List[int]:
        """Decode one bit per masked DCT coefficient, with majority voting."""
        cfg = self.cfg
        if image.mode != "RGB":
            image = image.convert("RGB")

        x = image_to_tensor(image).unsqueeze(0).to(self.device, self.dtype) * 2 - 1
        latents = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

        center = extract_center_patch(latents, cfg.center_ratio)
        h_c, w_c = center.shape[-2], center.shape[-1]

        h0, h1 = int(h_c * cfg.dct_min), int(h_c * cfg.dct_max)
        w0, w1 = int(w_c * cfg.dct_min), int(w_c * cfg.dct_max)
        capacity = (h1 - h0) * (w1 - w0)

        coords = [(h, w) for h in range(h0, h1) for w in range(w0, w1)]
        key_seed = sum(ord(c) for c in cfg.secret_key)
        np.random.default_rng(key_seed).shuffle(coords)

        center_dct = dct2d_torch(center)

        n_ch = len(cfg.channels)
        bits_per_ch = cfg.message_bit_length // n_ch
        remainder = cfg.message_bit_length % n_ch

        decoded: List[int] = []
        for i, ch in enumerate(cfg.channels):
            ch_bit_len = bits_per_ch + (1 if i < remainder else 0)
            if ch_bit_len == 0:
                continue
            n_rep = capacity // ch_bit_len
            if n_rep == 0:
                decoded.extend([0] * ch_bit_len)
                continue

            embed_coords = coords[: n_rep * ch_bit_len]
            raw = [int(center_dct[0, ch, h, w].item() > 0) for h, w in embed_coords]

            voted: List[int] = []
            for j in range(ch_bit_len):
                votes = [raw[k * ch_bit_len + j] for k in range(n_rep)]
                voted.append(Counter(votes).most_common(1)[0][0])
            decoded.extend(voted)
        return decoded

    def extract(self, image: Image.Image) -> str:
        """Inversion-free watermark extraction (see Algorithm 1, EXTRACT)."""
        return bits_to_message(self._read_bits(image))

    def extract_with_confidence(
        self, image: Image.Image
    ) -> Tuple[str, float]:
        """Same as :meth:`extract` but also returns a confidence ∈ [0, 1]."""
        msg = self.extract(image)
        target = self.cfg.message
        if msg == target:
            return msg, 1.0
        if "[ERROR" in msg:
            return msg, 0.0
        if len(msg) == len(target):
            match = sum(1 for a, b in zip(msg, target) if a == b)
            return msg, match / len(target)
        return msg, 0.1

    # ------------------------------------------------------------------
    # Robust-extraction helpers for rotation and crop&scale
    # ------------------------------------------------------------------
    def extract_robust_rotation(
        self,
        image: Image.Image,
        angles: Optional[List[float]] = None,
        scales: Optional[List[float]] = None,
        confidence_threshold: float = 0.5,
    ) -> str:
        """Brute-force search over rotation angles and scales."""
        angles = angles or [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75, -90]
        scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]

        size = self.cfg.image_size
        best_msg, best_conf = "", 0.0
        for s in scales:
            new_size = (int(image.width * s), int(image.height * s))
            scaled = image.resize(new_size, Image.Resampling.LANCZOS)
            if scaled.size != (size, size):
                canvas = Image.new("RGB", (size, size), (128, 128, 128))
                canvas.paste(
                    scaled,
                    ((size - scaled.width) // 2, (size - scaled.height) // 2),
                )
                scaled = canvas
            for a in angles:
                rotated = scaled.rotate(a, resample=Image.BILINEAR, expand=False)
                msg, conf = self.extract_with_confidence(rotated)
                if conf > best_conf:
                    best_msg, best_conf = msg, conf
        if best_conf > confidence_threshold:
            return best_msg
        return "[ERROR: Low confidence under rotation search]"

    def extract_robust_crop_scale(
        self,
        image: Image.Image,
        scales: Optional[List[float]] = None,
        confidence_threshold: float = 0.3,
    ) -> str:
        """Brute-force scale search for crop-and-scale style attacks."""
        scales = scales or [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
        best_msg, best_conf = "", 0.0
        for s in scales:
            new_size = (int(image.width * s), int(image.height * s))
            scaled = image.resize(new_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", image.size, (0, 0, 0))
            canvas.paste(
                scaled,
                (
                    (image.width - scaled.width) // 2,
                    (image.height - scaled.height) // 2,
                ),
            )
            msg, conf = self.extract_with_confidence(canvas)
            if conf > best_conf:
                best_msg, best_conf = msg, conf
        if best_conf > confidence_threshold:
            return best_msg
        return "[ERROR: Low confidence under crop/scale search]"

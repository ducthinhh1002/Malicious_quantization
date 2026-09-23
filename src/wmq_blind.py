"""Blind VAE quantization experiment; no watermark detector or secret inputs.

The suppression target is a hypothesis, not a watermark oracle.
The original route uses only the marked pipeline. Optional natural-image routes
use a separately declared dataset of unpaired images and the marked encoder.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import platform
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from wmq_diagnostics import quality_warning
from wmq_runtime import batches, batch_size, DataCache, all_finite, EVENTS, BATCH_LIMITS, image01, decoded01
from wmq_objectives import spectral_loss, NaturalDiscriminator, discriminator_update
from wmq_science import QualityBudget, audit_prompt_protocol

FLOAT_METHODS = ("natural_full_finetune", "natural_gan_finetune")
QAT_METHODS = ("natural_qat_purification", "natural_residual_qat", "natural_qat_scale", "natural_gan_qat")
RESIDUAL_METHODS = ("natural_residual", "natural_residual_qat", "natural_random_subspace",
                    "natural_frequency_subspace", "natural_contrastive_subspace")
NATURAL_METHODS = ("natural_rounding", "natural_rounding_scale", "natural_residual",
                   *QAT_METHODS, *FLOAT_METHODS, "natural_spectral",
                   "natural_random_subspace", "natural_frequency_subspace", "natural_contrastive_subspace",
                   "natural_teacher_rounding")


def parse_method_bit_exclusions(values):
    quantized_methods = {"fixed_ptq", "sensitivity", "rounding", "rounding_scale",
                         "reconstruction", "block_reconstruction", *NATURAL_METHODS} - set(FLOAT_METHODS)
    exclusions = set()
    for value in values:
        method, separator, bit_text = value.rpartition(":")
        if (not separator or method not in quantized_methods or not bit_text.isdigit()
                or int(bit_text) not in (2, 3, 4, 6, 8)):
            raise ValueError(f"Invalid --exclude-method-bits value: {value}")
        exclusions.add((method, int(bit_text)))
    if len(exclusions) != len(values):
        raise ValueError("Duplicate --exclude-method-bits values")
    return exclusions


def blur(x):
    """Fixed Gaussian [1,4,6,4,1]/16, reflection padding, no learned prior."""
    k = x.new_tensor([1, 4, 6, 4, 1]) / 16
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, 5, 5)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel,
                    groups=x.shape[1])


def semantic_lowpass(x, factor=8):
    """Content-preserving view that discards small decoder residuals."""
    if x.ndim != 4 or min(x.shape[-2:]) < factor:
        raise ValueError("Semantic low-pass requires an NCHW image at least as large as its factor")
    pooled = F.avg_pool2d(x, factor, factor)
    return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


def pseudo_target(reference, strength):
    # Both inputs and targets retain unknown watermark content. Never call clean.
    image01(reference, "pseudo-target reference")
    return reference.lerp(blur(reference), strength)


def natural_image_manifest(directory, train_n, search_n, seed, resolution=512):
    """Deterministic disjoint split, deduplicated by file content; no owner oracle."""
    root = Path(directory).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    unique = {}
    for path in paths:
        unique.setdefault(sha(path), path)
    entries = [{"path": str(p), "sha256": h} for h, p in unique.items()]
    order = np.random.default_rng(seed).permutation(len(entries))
    if len(entries) < train_n + search_n:
        raise ValueError(f"Need at least {train_n + search_n} distinct natural image files")
    chosen = [entries[i] for i in order[:train_n + search_n]]
    provenance = root / "dataset_provenance.json"
    return {"train": chosen[:train_n], "search": chosen[train_n:],
            "dataset_provenance": json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else None,
            "preprocessing": f"EXIF transpose, RGB, bicubic fit center crop {resolution}x{resolution}",
            "assumption": "User-supplied natural unwatermarked images; not detector-certified",
            "paired_with_generated_images": False}


@torch.no_grad()
def cache_natural(vae, entries, eval_batch_size=1, resolution=512):
    from PIL import Image, ImageOps
    device = next(vae.parameters()).device
    latents, images = [], []
    for entry in entries:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Natural image changed after manifest creation")
        with Image.open(entry["path"]) as im:
            im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (resolution, resolution),
                              method=Image.Resampling.BICUBIC)
            x = torch.from_numpy(np.array(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        images.append(x)
        image01(x, "natural image")
    def encode(chunk):
        # encode returns VAE-space latents: do NOT apply diffusion scaling_factor here.
        z = vae.encode(torch.cat(chunk).to(device) * 2 - 1).latent_dist.mode()
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite natural-image latent")
        return z.cpu()
    for _, z in batches(images, encode, eval_batch_size, "natural_encode"):
        latents.extend(z.split(1))
    return latents, images


def texture_metrics(x, ref):
    """Visual detail diagnostics; these cannot identify ownership frequencies."""
    high, ref_high = x - blur(x), ref - blur(ref)
    energy = ref_high.square().mean().item()
    return {"highpass_mse": F.mse_loss(high, ref_high).item(),
            "highpass_energy_ratio": high.square().mean().item() / energy if energy > 1e-12 else None}


@torch.no_grad()
def save_pseudo_diagnostic(refs, strength, args, folder):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=False)
    psnr, ssim, texture = [], [], []
    def evaluate(chunk):
        ref = torch.cat(chunk)
        x = pseudo_target(ref, strength).clamp(0, 1)
        p, s = pair_metrics(x, ref)
        high, ref_high = x - blur(x), ref - blur(ref)
        mse = lambda v: v.square().flatten(1).mean(1)
        energy = mse(ref_high)
        ratio = torch.where(energy > 1e-12, mse(high) / energy.clamp_min(1e-12), float("nan"))
        return torch.stack([p, s, mse(high - ref_high), ratio], 1).cpu().tolist(), x.cpu()
    for start, (metrics, pixels) in batches(refs, evaluate, batch_size(getattr(args, "eval_batch_size", 0), refs[0].device), "pseudo_target"):
        for j, (p, s, error, ratio) in enumerate(metrics):
            psnr.append(p); ssim.append(s)
            texture.append({"highpass_mse": error, "highpass_energy_ratio": ratio if math.isfinite(ratio) else None})
            Image.fromarray((pixels[j].permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(folder / f"{start+j:04d}.png")
    return {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim)),
            "feasible": bool(np.mean(psnr) >= args.min_psnr and np.mean(ssim) >= args.min_ssim
                             and min(psnr) >= args.min_image_psnr),
            "per_image_texture": texture, "per_image_psnr": psnr, "per_image_ssim": ssim,
            "interpretation": "Direct smoothing hypothesis, not a clean target or quantized model"}


class FloatWeight(nn.Module):
    """Independent FP32 decoder parameter for the unrestricted control."""
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().float().clone())

    def forward(self, differentiable=True):
        return self.weight if differentiable else self.weight.detach()


def scheduled_lr(base, step, total, warmup):
    """Linear warmup then cosine decay to 10% of the peak learning rate."""
    warmup = min(warmup, max(total - 1, 0))
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup - 1) / max(total - warmup - 1, 1)
    return base * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


class RoundingGrid(nn.Module):
    """Hard rounding, with optional bounded per-channel scale optimization.

    Forward uses hard integers with a sigmoid straight-through gradient. Export
    uses exactly the same grid; unconstrained full-precision weights never train.
    """
    def __init__(self, weight, bits, clip=1.0, learn_scale=False, learn_code_offsets=False):
        super().__init__()
        if bits not in (2, 3, 4, 6, 8) or not 0 < clip <= 1:
            raise ValueError("Invalid bitwidth/clip")
        w = weight.detach().float()
        if not torch.isfinite(w).all():
            raise ValueError("Nonfinite source weights")
        # Use all 2**bits signed codes, e.g. INT4 is [-8, 7].  A zero-point of
        # zero does not require the deliberately narrow [-7, 7] fake-quant range.
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.bits = bits
        self.register_buffer("source", w.clone())
        flat = w.flatten(1)
        positive_scale = flat.amax(1).clamp_min(0) / self.qmax
        negative_scale = (-flat.amin(1)).clamp_min(0) / (-self.qmin)
        scale = torch.maximum(positive_scale, negative_scale).clamp_min(1e-8) * clip
        scale = scale.reshape([-1] + [1] * (w.ndim - 1))
        value = (w / scale).clamp(self.qmin, self.qmax)
        self.register_buffer("base_scale", scale)
        self.log_scale = nn.Parameter(torch.zeros_like(scale), requires_grad=learn_scale)
        frac = value - value.floor()
        self.alpha = nn.Parameter(torch.logit(frac.clamp(1e-4, 1 - 1e-4)),
                                  requires_grad=not learn_code_offsets)
        # Dimensionless offsets let QAT cross more than the two adjacent cells
        # available to AdaRound-style alpha. They are bounded and the source
        # marked weights remain immutable.
        self.code_offset = nn.Parameter(torch.zeros_like(w) if learn_code_offsets else w.new_zeros(()),
                                        requires_grad=learn_code_offsets)
        self.learn_code_offsets = learn_code_offsets

    @property
    def scale(self):
        return self.base_scale * self.log_scale.clamp(math.log(.8), math.log(1.25)).exp()

    def forward(self, differentiable=True):
        scale = self.scale
        if self.learn_code_offsets:
            value = (self.source / scale + self.code_offset).clamp(self.qmin, self.qmax)
            hard = value.round()
            codes = hard + (value - value.detach()) if differentiable else hard
            return codes * scale
        soft = self.alpha.sigmoid()
        hard = (self.alpha >= 0).to(soft.dtype)
        rounding = hard + (soft - soft.detach()) if differentiable else hard
        floor = (self.source / scale).clamp(self.qmin, self.qmax).detach().floor()
        return (floor + rounding).clamp(self.qmin, self.qmax) * scale

    @torch.no_grad()
    def code_change_fraction(self):
        """Fraction of exported codes differing from ordinary RTN on this grid."""
        baseline = (self.source / self.base_scale).round().clamp(self.qmin, self.qmax)
        current = (self(False) / self.scale).round().clamp(self.qmin, self.qmax)
        return (current != baseline).float().mean()

def pair_metrics(x, y):
    """PSNR and Wang-style SSIM: 11x11 Gaussian sigma=1.5, valid crop, population covariance."""
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 11:
        raise ValueError("SSIM requires paired NCHW images at least 11x11")
    image01(x, "metric image")
    image01(y, "metric reference")
    mse = (x - y).square().flatten(1).mean(1)
    psnr = -10 * mse.clamp_min(1e-12).log10()
    # FP64 statistics avoid cancellation in nearly constant patches; model execution stays FP32.
    x, y = x.double(), y.double()
    axis = torch.arange(-5, 6, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    def window(z):
        horizontal = kernel.reshape(1, 1, 1, 11).expand(z.shape[1], 1, 1, 11)
        vertical = kernel.reshape(1, 1, 11, 1).expand(z.shape[1], 1, 11, 1)
        return F.conv2d(F.conv2d(z, horizontal, groups=z.shape[1]), vertical, groups=z.shape[1])
    ux, uy = window(x), window(y)
    vx = window(x.square()) - ux.square()
    vy = window(y.square()) - uy.square()
    cov = window(x * y) - ux * uy
    ssim = (((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2)))
    return psnr, ssim.flatten(1).mean(1)


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(path, rows):
    """Scalar metrics only; per-image arrays remain in the accompanying JSON."""
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))} for row in rows]
    fields = list(dict.fromkeys(k for row in flat for k in row))
    tmp = Path(path).with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def select_names(vae, scope):
    selected = []
    for name, p in vae.decoder.named_parameters():
        if p.ndim < 2 or not name.endswith("weight"):
            continue
        if scope == "all" or name.startswith(("up_blocks.", "conv_out.")):
            selected.append(name)
    if not selected:
        raise ValueError("No VAE decoder weights selected")
    return selected


def materialize(decoder, names, grids):
    with torch.no_grad():
        for name, grid in zip(names, grids):
            decoder.get_parameter(name).copy_(grid(False))


@torch.no_grad()
def score(vae, latents, refs, strength, args, folder=None):
    psnr, ssim, errors = [], [], []
    texture = []
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    if not latents or len(latents) != len(refs):
        raise ValueError("Need nonempty, paired latents/references")
    device = next(vae.parameters()).device
    def evaluate(chunk):
        z = torch.cat([pair[0] for pair in chunk]).to(device, torch.float32)
        ref = torch.cat([pair[1] for pair in chunk]).to(device)
        x = vae.decode(z, return_dict=False)[0]
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite decoder output")
        x = decoded01(x)
        p, s = pair_metrics(ref, x)
        mse = lambda v: v.square().flatten(1).mean(1)
        high, ref_high = x - blur(x), ref - blur(ref)
        energy = mse(ref_high)
        ratio = torch.where(energy > 1e-12, mse(high) / energy.clamp_min(1e-12), float("nan"))
        semantic_error = mse(semantic_lowpass(x) - semantic_lowpass(ref))
        metrics = torch.stack([p, s, mse(x - pseudo_target(ref, strength)),
                               mse(high - ref_high), ratio, semantic_error], 1).cpu().tolist()
        return metrics, x.cpu() if folder is not None else None
    for start, (metrics, pixels) in batches(list(zip(latents, refs)), evaluate, batch_size(getattr(args, "eval_batch_size", 0), device), "score"):
        for j, (p, s, error, high_error, ratio, semantic_error) in enumerate(metrics):
            psnr.append(p); ssim.append(s); errors.append(error)
            texture.append({"highpass_mse": high_error, "highpass_energy_ratio": ratio if math.isfinite(ratio) else None,
                            "semantic_lowpass_mse": semantic_error})
            if folder is not None:
                array = (pixels[j].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                Image.fromarray(array).save(folder / f"{start+j:04d}.png")
    p, s = torch.tensor(psnr), torch.tensor(ssim)
    feasible = bool(p.mean() >= args.min_psnr and s.mean() >= args.min_ssim
                    and p.min() >= args.min_image_psnr)
    if getattr(args, "quality_constraint", "off") == "dual":
        budget_violation = ((p < args.budget_psnr) | (s < args.budget_ssim)).float().mean().item()
        feasible = feasible and budget_violation <= args.budget_max_violation + 1e-7
    else:
        budget_violation = None
    return {"psnr": p.mean().item(), "ssim": s.mean().item(),
            "budget_violation_fraction": budget_violation,
            "psnr_p05": torch.quantile(p, .05).item(), "min_psnr": p.min().item(),
            "quality_violation_fraction": ((p < args.min_psnr) | (s < args.min_ssim)).float().mean().item(),
            "target_mse": sum(errors) / len(errors), "feasible": feasible,
            "semantic_lowpass_mse": sum(t["semantic_lowpass_mse"] for t in texture) / len(texture),
            "per_image_psnr": psnr, "per_image_ssim": ssim, "per_image_texture": texture}


def choose(rows, policy="constrained"):
    eligible = list(rows) if policy == "report" else [r for r in rows if r["feasible"]]
    return min(eligible, key=lambda r: (r.get("selection_objective", r["target_mse"]), -r["psnr"], r["candidate"])) if eligible else None


@torch.no_grad()
def natural_reconstruction_metrics(vae, latents, images, perceptual=None, perceptual_weight=0., eval_batch_size=1,
                                   residual=None, residual_weight=0., spectral_weight=0.):
    """Natural validation ranks reconstruction only; generated pairs supply quality gates."""
    if not latents or len(latents) != len(images):
        raise ValueError("Need nonempty paired natural latents and images")
    device = next(vae.parameters()).device
    errors, perceptual_errors, residual_errors, spectral_errors = [], [], [], []
    def evaluate(chunk):
        z = torch.cat([p[0] for p in chunk]).to(device, torch.float32)
        x = torch.cat([p[1] for p in chunk]).to(device)
        image01(x, "natural validation reference")
        prediction = vae.decode(z, return_dict=False)[0] / 2 + .5
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite natural reconstruction")
        error = (prediction.clamp(0, 1) - x).square().flatten(1).mean(1)
        lp = perceptual(prediction.clamp(0, 1) * 2 - 1, x * 2 - 1).reshape(len(chunk), -1).mean(1) if perceptual is not None else torch.zeros_like(error)
        rp = residual.loss(prediction.clamp(0, 1) - x) if residual is not None else torch.zeros_like(error)
        sp = spectral_loss(prediction.clamp(0, 1), x) if spectral_weight else torch.zeros_like(error)
        return torch.stack([error, lp, rp, sp], 1).cpu().tolist()
    for _, metrics in batches(list(zip(latents, images)), evaluate, eval_batch_size, "natural_validation"):
        for error, lp, rp, sp in metrics:
            errors.append(error)
            residual_errors.append(rp)
            spectral_errors.append(sp)
            if perceptual is not None:
                perceptual_errors.append(lp)
    mse = sum(errors) / len(errors)
    lpips_value = sum(perceptual_errors) / len(perceptual_errors) if perceptual_errors else 0.
    if not math.isfinite(lpips_value):
        raise ValueError("Nonfinite natural validation perceptual loss")
    residual_value = sum(residual_errors) / len(residual_errors)
    spectral_value = sum(spectral_errors) / len(spectral_errors)
    return {"natural_validation_mse": mse, "natural_validation_lpips": lpips_value if perceptual is not None else None,
            "natural_validation_residual": residual_value if residual is not None else None,
            "natural_validation_spectral": spectral_value if spectral_weight else None,
            "target_mse": mse, "selection_objective": mse + perceptual_weight * lpips_value + residual_weight * residual_value + spectral_weight * spectral_value}


@torch.no_grad()
def profile_layers(vae, names, latents, refs, bits, clip, args):
    """One-layer perturbation replay on TRAIN only, not an ownership profiler."""
    device = next(vae.parameters()).device
    rows = []
    for name in names:
        weight = vae.decoder.get_parameter(name)
        original = weight.detach().clone()
        errors, gains = [], []
        try:
            weight.copy_(RoundingGrid(weight, bits, clip).to(device)(False))
            def evaluate(chunk):
                z = torch.cat([p[0] for p in chunk]).to(device)
                ref = torch.cat([p[1] for p in chunk]).to(device)
                image = decoded01(vae.decode(z, return_dict=False)[0])
                target = pseudo_target(ref, args.strength)
                mse = lambda v: v.square().flatten(1).mean(1)
                return torch.stack([mse(image - ref), mse(ref - target) - mse(image - target)], 1).cpu().tolist()
            for _, metrics in batches(list(zip(latents, refs)), evaluate, batch_size(getattr(args, "eval_batch_size", 0), device), "profile"):
                for error, gain in metrics:
                    errors.append(error); gains.append(gain)
        finally:
            weight.copy_(original)
        if not all(math.isfinite(x) for x in errors + gains):
            raise ValueError(f"Nonfinite layer profile: {name}")
        # Identical bootstrap sample indices across layers preserve paired calibration images.
        rng = np.random.default_rng(args.seed)
        indices = rng.integers(len(errors), size=(args.profile_bootstrap, len(errors)))
        priorities = np.asarray(gains)[indices].sum(1) / np.maximum(np.asarray(errors)[indices].sum(1), 1e-12)
        low, high = np.quantile(priorities, [.025, .975]) if len(errors) > 1 else (None, None)
        rows.append({"name": name, "output_mse": sum(errors) / len(latents),
                     "sensitivity_kind": "visual_proxy_not_ownership",
                     "proxy_gain": sum(gains) / len(latents),
                     "priority": sum(gains) / max(sum(errors), 1e-12),
                     "priority_ci95_low": None if low is None else float(low),
                     "priority_ci95_high": None if high is None else float(high),
                     "per_image_output_mse": errors, "per_image_proxy_gain": gains})
    return sorted(rows, key=lambda r: (-r["priority"], r["name"]))


def optimize_branch(vae, names, train_z, train_ref, search_z, search_ref,
                    args, bits, clip, method, profile, folder, diagnostics_root=None,
                    natural_search=None, preserve_data=None, perceptual=None, residual=None,
                    heavy_folder=None, teacher_targets=None):
    """Independent initialization, train-only updates, search-only checkpoint choice."""
    device = next(vae.parameters()).device
    block_branch = method == "block_reconstruction"
    block_groups = {}
    if block_branch:
        for index, name in enumerate(names):
            parts = name.split(".")
            prefix = ".".join(parts[:2]) if parts[0] == "up_blocks" else parts[0]
            # A bare Conv2d in CPU tests is itself the block.
            prefix = "" if prefix == "weight" else prefix
            block_groups.setdefault(prefix, []).append(index)
        if args.steps < len(block_groups):
            raise ValueError("Block reconstruction requires at least one step per block")
    pristine = {k: v.detach().clone() for k, v in vae.decoder.state_dict().items()}
    qat_branch = method in QAT_METHODS
    float_branch = method in FLOAT_METHODS
    gan_branch = method in ("natural_gan_qat", "natural_gan_finetune")
    spectral_weight = args.spectral_weight if method == "natural_spectral" else 0.
    grids = nn.ModuleList([FloatWeight(vae.decoder.get_parameter(k)) if float_branch else RoundingGrid(vae.decoder.get_parameter(k), bits, clip,
                           learn_scale=method in ("rounding_scale", "natural_rounding_scale", "natural_qat_scale"),
                           learn_code_offsets=qat_branch) for k in names]).to(device)
    steering_layers = getattr(args, '_steering_layers', None) if method.startswith('natural_') and not float_branch else None
    if steering_layers is not None:
        unknown = set(steering_layers) - set(names)
        if unknown or not steering_layers:
            raise ValueError(f'Invalid steering layer names: {sorted(unknown)}')
        for name, grid in zip(names, grids):
            if name not in steering_layers:
                grid.requires_grad_(False)  # Still quantized at the same bitwidth; frozen at RTN.
    rows, updates, best = [], [], None
    best_state = None
    natural = method.startswith("natural_")
    teacher_branch = method == "natural_teacher_rounding"
    if teacher_branch and (teacher_targets is None or preserve_data is None
            or len(teacher_targets['train']) != len(preserve_data[0])
            or len(teacher_targets['search']) != len(search_z)):
        raise ValueError("Teacher branch requires aligned frozen TRAIN/SEARCH targets")
    residual_branch = method in RESIDUAL_METHODS
    if residual_branch and residual is None:
        raise ValueError("Residual branch requires a frozen TRAIN-only basis")
    residual_weight = args.residual_weight if residual_branch else 0.
    if natural and (natural_search is None or preserve_data is None):
        raise ValueError("Natural branch requires independent natural validation and generated preservation data")
    perceptual_weight = getattr(args, "natural_perceptual_weight", 0.) if natural else 0.
    if perceptual_weight and perceptual is None:
        raise ValueError("Natural perceptual objective requires a frozen LPIPS network")
    policy = getattr(args, "quality_policy", "report")
    strength = 0. if method in ("reconstruction", "block_reconstruction") or natural else args.strength
    folder.mkdir(parents=True, exist_ok=False)
    diagnostics_root = folder if diagnostics_root is None else diagnostics_root
    thresholds = {"min_psnr": args.min_psnr, "min_ssim": args.min_ssim,
                  "min_image_psnr": args.min_image_psnr}
    budget = QualityBudget(args.budget_psnr, args.budget_ssim, args.dual_lr) if args.quality_constraint == "dual" and natural else None

    def evaluate(step, source):
        nonlocal best, best_state
        try:
            materialize(vae.decoder, names, grids)
            metrics = score(vae, search_z, search_ref, strength, args)
            if natural:
                # Gate on generated pairs, rank on disjoint natural validation reconstruction.
                metrics["generated_reference_mse"] = metrics["target_mse"]
                metrics.update(natural_reconstruction_metrics(vae, *natural_search, perceptual, perceptual_weight,
                    batch_size(getattr(args, "eval_batch_size", 0), device), residual if residual_branch else None, residual_weight, spectral_weight))
                if args.natural_preservation == "lowpass" and not float_branch:
                    metrics["selection_objective"] += (args.preserve_weight * metrics["semantic_lowpass_mse"])
                elif qat_branch and not residual_branch:
                    metrics["selection_objective"] += (args.qat_semantic_preserve_weight *
                                                        metrics["semantic_lowpass_mse"])
                if float_branch:
                    metrics["selection_objective"] += args.ft_preserve_weight * metrics["semantic_lowpass_mse"]
                if teacher_branch:
                    from wmq_teacher import teacher_mse
                    metrics['teacher_search_mse'] = teacher_mse(vae, search_z, teacher_targets['search'], args.eval_batch_size)
                    metrics['selection_objective'] += args.teacher_weight * metrics['teacher_search_mse']
                if qat_branch:
                    count = sum(g.source.numel() for g in grids)
                    stats = torch.stack([torch.stack([g.code_change_fraction() * g.source.numel(),
                        g.code_offset.square().sum(), (g.code_offset.abs() >= 1).sum()]) for g in grids]).sum(0) / count
                    changed, trust, multicell = stats.cpu().tolist()
                    metrics.update(code_change_fraction_vs_rtn=changed, code_offset_rms=math.sqrt(trust),
                                   code_offset_ge_one_fraction=multicell)
            metrics.setdefault("selection_objective", metrics["target_mse"])
        finally:
            vae.decoder.load_state_dict(pristine)
        row = {"candidate": f"{method}_w{bits}_c{clip}_step{step}", "source": source,
               "method": method, "bits": bits, "clip": clip, "step": step, **metrics}
        rows.append(row)
        if not row["feasible"]:
            quality_warning(diagnostics_root, "search_candidate", "Candidate failed quality gate",
                            method=method, metrics=row, thresholds=thresholds)
        winner = choose(rows, policy=policy)
        # Retain a diagnosed failed control if no candidate is feasible; never call it successful.
        if winner is None:
            winner = min(rows, key=lambda r: (-r["psnr"], r["candidate"]))
        if best is None or winner["candidate"] != best["candidate"]:
            if winner is not row:
                raise RuntimeError("Selection snapshot lost")
            best, best_state = dict(winner), {k: v.detach().cpu().clone() for k, v in grids.state_dict().items()}
        save_json(folder / "search.json", rows)
        save_csv(folder / "search.csv", rows)
        print({k: v for k, v in row.items() if not k.startswith("per_image")}, flush=True)

    evaluate(0, "marked_fp32_fallback" if float_branch else ("fixed_rtn" if method == "fixed_ptq" else "rtn_fallback"))
    order_rng = torch.Generator().manual_seed(args.seed)
    params = [p for p in grids.parameters() if p.requires_grad]
    base_lr = args.ft_lr if float_branch else (args.qat_lr if qat_branch else args.lr)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(params, lr=base_lr, weight_decay=args.weight_decay) if method not in ("fixed_ptq", "sensitivity") else None
    discriminator = None
    if gan_branch:
        # Isolate initialization from branch ordering and the data sampler.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            discriminator = NaturalDiscriminator().to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr)
    disc_valid_updates = 0
    train_image_forwards = 0
    lr_backoff = 1.
    ranked = [names.index(r["name"]) for r in profile]
    valid_updates = 0
    backward_attempts = 0
    train_evaluations = 0
    if method == "sensitivity":
        materialize(vae.decoder, names, grids)
        training_best = score(vae, train_z, train_ref, strength, args)
        train_evaluations += 1
        vae.decoder.load_state_dict(pristine)
    total = (0 if method == "fixed_ptq" else
             ((args.ft_steps or args.steps) if float_branch else ((args.qat_steps or args.steps) if qat_branch else args.steps)))
    for step in range(1, total + 1):
        if optimizer is not None and (qat_branch or float_branch or args.lr_schedule == "warmup_cosine"):
            for group in optimizer.param_groups:
                group["lr"] = scheduled_lr(base_lr, step, total, args.warmup_steps) * lr_backoff
        applied, reason = True, None
        if method == "sensitivity":
            index = ranked[(step - 1) % len(ranked)]
            grid = grids[index]
            old = grid.log_scale.detach().clone()
            # Coordinate proposals change scale for one measured layer, keeping all layers low-bit.
            direction = -1 if ((step - 1) // len(ranked)) % 2 == 0 else 1
            with torch.no_grad():
                grid.log_scale.add_(direction * .05).clamp_(math.log(.8), math.log(1.25))
            try:
                materialize(vae.decoder, names, grids)
                trial = score(vae, train_z, train_ref, strength, args)
                train_evaluations += 1
            finally:
                vae.decoder.load_state_dict(pristine)
            # Enter feasibility first; never trade a feasible training point for an infeasible one.
            rank = (lambda m: (m["target_mse"],)) if policy == "report" else (lambda m: (not m["feasible"], m["target_mse"] if m["feasible"] else -m["psnr"]))
            applied = rank(trial) < rank(training_best)
            if applied:
                training_best = trial
            else:
                with torch.no_grad():
                    grid.log_scale.copy_(old)
                reason = "no_train_improvement"
        else:
            active_params = params
            block_name = None
            if block_branch:
                block_name = list(block_groups)[min((step - 1) * len(block_groups) // total, len(block_groups) - 1)]
                for index, grid in enumerate(grids):
                    grid.alpha.requires_grad_(index in block_groups[block_name])
                active_params = [p for p in grids.parameters() if p.requires_grad]
            indices = []
            for offset in range(args.train_batch_size):
                position = ((step - 1) * args.train_batch_size + offset) % len(train_z)
                if position == 0:
                    order = torch.randperm(len(train_z), generator=order_rng).tolist()
                indices.append(order[position])
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                hidden = vae.post_quant_conv(torch.cat([train_z[i] for i in indices]).to(device))
            weights = {k: q() for k, q in zip(names, grids)}
            if block_branch:
                captured = []
                handle = vae.decoder.get_submodule(block_name).register_forward_hook(lambda module, inputs, output: captured.append(output))
                try:
                    with torch.no_grad():
                        vae.decoder(hidden)
                    target_activation = captured.pop().detach()
                    raw = functional_call(vae.decoder, weights, (hidden,)) / 2 + .5
                    block_loss = F.mse_loss(captured.pop(), target_activation)
                finally:
                    handle.remove()
                train_evaluations += 1
                train_image_forwards += len(indices)
            else:
                raw = functional_call(vae.decoder, weights, (hidden,)) / 2 + .5
            ref = torch.cat([train_ref[i] for i in indices]).to(device)
            image01(ref, "training reference")
            target_loss = F.mse_loss(raw, pseudo_target(ref, strength))
            if block_branch:
                target_loss = block_loss
            # Natural branches replace this term with generated-image preservation.
            preserve = raw.new_zeros(()) if natural or block_branch else F.mse_loss(blur(raw), blur(ref))
            perceptual_loss = raw.new_zeros(())
            residual_loss = raw.new_zeros(())
            trust_loss = raw.new_zeros(())
            spectrum = spectral_loss(raw, ref).mean() if spectral_weight else raw.new_zeros(())
            adversarial = raw.new_zeros(())
            distillation = raw.new_zeros(())
            disc_loss, disc_applied = None, None
            if discriminator is not None:
                disc_loss, disc_applied = discriminator_update(discriminator, disc_optimizer, ref, raw.clamp(0, 1))
                disc_valid_updates += int(disc_applied)
                adversarial = F.softplus(-discriminator(raw.clamp(0, 1))).mean()
            if natural:
                if residual_branch:
                    residual_loss = residual.loss(raw - ref).mean()
                if perceptual is not None:
                    perceptual_loss = perceptual(raw.clamp(0, 1) * 2 - 1, ref * 2 - 1).mean()
                if float_branch and args.ft_preserve_weight == 0 and budget is None:
                    generated = preserve_hidden = None
                    preserve = raw.new_zeros(())
                else:
                    js = [((step - 1) * args.train_batch_size + j) % len(preserve_data[0]) for j in range(args.train_batch_size)]
                    with torch.no_grad():
                        preserve_hidden = vae.post_quant_conv(torch.cat([preserve_data[0][j] for j in js]).to(device))
                    generated = functional_call(vae.decoder, weights, (preserve_hidden,)) / 2 + .5
                    preserve_ref = image01(torch.cat([preserve_data[1][j] for j in js]).to(device), "preservation reference")
                    if args.natural_preservation == "lowpass" or (qat_branch and not residual_branch) or float_branch:
                        preserve = F.mse_loss(semantic_lowpass(generated), semantic_lowpass(preserve_ref))
                    else:
                        preserve = (residual.preservation(generated - preserve_ref, args.residual_preservation)
                                    if residual_branch else F.mse_loss(generated, preserve_ref))
                    train_evaluations += 1
                    train_image_forwards += len(js)
                    if teacher_branch:
                        teacher_ref = torch.cat([teacher_targets['train'][j] for j in js]).to(device)
                        distillation = F.mse_loss(generated.clamp(0, 1), teacher_ref)
                if qat_branch:
                    trust_loss = sum(g.code_offset.square().sum() for g in grids) / sum(g.code_offset.numel() for g in grids)
            preserve_weight = (args.preserve_weight if natural and not float_branch and args.natural_preservation == "lowpass" else args.ft_preserve_weight if float_branch else
                               (args.qat_semantic_preserve_weight if qat_branch and not residual_branch else args.preserve_weight))
            loss = (target_loss + perceptual_weight * perceptual_loss + preserve_weight * preserve +
                    residual_weight * residual_loss + args.qat_trust_weight * trust_loss +
                    spectral_weight * spectrum + (args.adversarial_weight * adversarial if gan_branch else 0.) +
                    args.teacher_weight * distillation)
            budget_penalty = raw.new_zeros(())
            budget_violations = raw.new_zeros(2)
            if budget is not None:
                _, preservation_ssim = pair_metrics(generated.clamp(0, 1), preserve_ref)
                budget_penalty, budget_violations = budget.terms(generated, preserve_ref, preservation_ssim)
                loss = loss + budget_penalty
            gradient_diagnostics = {}
            if args.gradient_diagnostics_every and step % args.gradient_diagnostics_every == 0:
                # Diagnostics of attacker losses only, never owner gradients.
                pieces = {"reconstruction": target_loss + perceptual_weight * perceptual_loss,
                          "teacher": args.teacher_weight * distillation,
                          "residual": residual_weight * residual_loss,
                          "preservation": preserve_weight * preserve + budget_penalty}
                for label, component in pieces.items():
                    if component.requires_grad:
                        grads = torch.autograd.grad(component, active_params, retain_graph=True, allow_unused=True)
                        squares = [g.detach().double().square().sum() for g in grads if g is not None]
                        diagnostic_norm = math.sqrt(float(torch.stack(squares).sum())) if squares else 0.
                        gradient_diagnostics[label + "_gradient_norm"] = diagnostic_norm if math.isfinite(diagnostic_norm) else None
            train_evaluations += 1
            train_image_forwards += len(indices)
            applied = bool(torch.isfinite(loss))
            grad_norm = None
            if applied:
                backward_attempts += 1
                loss.backward()
                applied = all(p.grad is not None for p in active_params) and all_finite(p.grad for p in active_params)
                if applied:
                    norm = nn.utils.clip_grad_norm_(active_params, 1.)
                    applied = bool(torch.isfinite(norm))
                    if applied:
                        grad_norm = norm.detach()
                    else:
                        reason = "nonfinite_gradient_norm"
                else:
                    reason = "nonfinite_or_missing_gradient"
            else:
                reason = "nonfinite_loss"
            if applied:
                backup = [p.detach().clone() for p in params]
                optimizer.step()
                applied = all_finite(params)
                if not applied:
                    with torch.no_grad():
                        for p, old in zip(params, backup):
                            p.copy_(old)
                    reason = "nonfinite_parameter_rollback"
                del backup
            if not applied:
                optimizer.state.clear()
                lr_backoff *= .5
                for group in optimizer.param_groups:
                    group["lr"] *= .5
            elif budget is not None:
                budget.update(budget_violations)
            with torch.no_grad():
                for grid in grids:
                    if float_branch:
                        continue
                    if grid.alpha.requires_grad:
                        grid.alpha.clamp_(-12, 12)
                    if grid.log_scale.requires_grad:
                        grid.log_scale.clamp_(math.log(.8), math.log(1.25))
                    if grid.code_offset.requires_grad:
                        grid.code_offset.clamp_(-args.qat_max_code_shift, args.qat_max_code_shift)
            scalars = torch.stack([target_loss.detach(), perceptual_loss.detach(), preserve.detach(), loss.detach(),
                                   residual_loss.detach(), trust_loss.detach(), spectrum.detach(), adversarial.detach(),
                                   grad_norm if grad_norm is not None else raw.new_tensor(float("nan"))]).cpu().tolist()
            updates.append({"step": step, "applied": applied, "reason": reason,
                            **gradient_diagnostics,
                            "teacher_mse": float(distillation.detach()) if teacher_branch and torch.isfinite(distillation) else None,
                            "budget_penalty": float(budget_penalty.detach()) if torch.isfinite(budget_penalty) else None,
                            "dual_mse": budget.multipliers[0] if budget else None,
                            "dual_ssim": budget.multipliers[1] if budget else None,
                            "block": block_name,
                            "discriminator_loss": disc_loss, "discriminator_applied": disc_applied,
                            **{k: v if math.isfinite(v) else None for k, v in zip(
                                ["reconstruction_mse", "perceptual_loss", "preservation_mse", "loss", "residual_loss", "trust_loss",
                                 "spectral_loss", "adversarial_loss", "grad_norm"], scalars)},
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            del weights, raw, ref, loss, target_loss, preserve, perceptual_loss, residual_loss, trust_loss, spectrum, adversarial
            if natural:
                del generated, preserve_hidden
        if method == "sensitivity":
            updates.append({"step": step, "applied": applied, "reason": reason})
        if not applied and method != "sensitivity":
            quality_warning(diagnostics_root, "optimizer_update", "Invalid update skipped or rolled back",
                            method=method, metrics=updates[-1], action="skip_update_and_continue")
        valid_updates += int(applied)
        if step % getattr(args, "log_every", 10) == 0 or step == total or (not applied and reason != "no_train_improvement"):
            save_json(folder / "updates.json", {"attempted": step, "valid_updates": valid_updates,
                      "skipped_or_rejected": step - valid_updates, "rows": updates})
            save_csv(folder / "updates.csv", updates)
        if step % args.eval_every == 0 or step == total:
            evaluate(step, method if valid_updates else ("marked_fp32_fallback" if float_branch else "rtn_fallback"))
    if args.evaluate_final and total:
        grids.final_state = {k: v.detach().cpu().clone() for k, v in grids.state_dict().items()}
    grids.load_state_dict(best_state)
    selected = {**best, "search_feasible": best["feasible"], "valid_updates": valid_updates,
                "steering_layers": steering_layers,
                "steering_threat_model": 'owner_informed_development_not_blind' if steering_layers is not None else None,
                "reconstruction_blocks": list(block_groups),
                "optimizer": args.optimizer, "weight_decay": args.weight_decay,
                "training_batch_size": args.train_batch_size, "discriminator_valid_updates": disc_valid_updates,
                "training_pool_images": len(train_z),
                "training_examples_seen": total * args.train_batch_size if optimizer is not None else 0,
                "training_epoch_equivalents": total * args.train_batch_size / len(train_z) if optimizer is not None else 0.,
                "selected_at_budget_boundary": best['step'] == total and total > 0,
                "adversarial_weight": args.adversarial_weight if gan_branch else 0., "spectral_weight": spectral_weight,
                "selection_uses_discriminator": False, "paper_reproduction": False,
                "additional_objectives": (["patch_logistic_adversarial"] if gan_branch else []) +
                                         (["multiscale_natural_log_spectrum"] if spectral_weight else []) +
                                         (["sequential_block_activation_reconstruction"] if block_branch else []),
                "gradient_updates": valid_updates if optimizer is not None else 0,
                "teacher_dependency": teacher_targets['provenance'] if teacher_branch else None,
                "teacher_weight": args.teacher_weight if teacher_branch else 0.,
                "teacher_search_image_forwards": len(rows) * len(search_z) if teacher_branch else 0,
                "objective": "generated_teacher_distillation_plus_natural_reconstruction" if teacher_branch else ("natural_residual_projection" if residual_branch else ("quantization_constrained_natural_purification" if qat_branch else ("unpaired_natural_reconstruction" if natural else ("reconstruction" if method in ("reconstruction", "block_reconstruction") else "blind_smoothing_proxy")))),
                "residual_weight": residual_weight,
                "quality_constraint": args.quality_constraint,
                "budget_psnr": args.budget_psnr, "budget_ssim": args.budget_ssim,
                "residual_loss_scale": residual.loss_scale if residual_branch else None,
                "preservation_mode": "lowpass_8x" if args.natural_preservation == "lowpass" and natural else (args.residual_preservation if residual_branch else ("lowpass_8x" if qat_branch or float_branch else "legacy")),
                "parameter_space": "unrestricted_fp32_decoder" if float_branch else "quantized_decoder_weights",
                "learning_rate_peak": base_lr, "lr_schedule": "warmup_cosine" if qat_branch or float_branch else args.lr_schedule,
                "objective_strength": strength,
                "quality_policy": policy, "perceptual_weight": perceptual_weight,
                "selected_source": best["source"], "attempted_updates": total,
                "train_evaluations": train_evaluations, "search_evaluations": len(rows),
                "train_image_forwards": train_evaluations * len(train_z) if method == "sensitivity" else train_image_forwards,
                "discriminator_image_forwards": total * args.train_batch_size * 3 if gan_branch else 0,
                "search_image_forwards": len(rows) * len(search_z), "backward_attempts": backward_attempts,
                "natural_validation_image_forwards": len(rows) * len(natural_search[0]) if natural else 0,
                "optimized_dofs": {"fixed_ptq": [], "sensitivity": ["layer_scale"], "rounding": ["rounding"],
                    "rounding_scale": ["rounding", "per_channel_scale"], "reconstruction": ["rounding"],
                    "block_reconstruction": ["sequential_block_rounding"],
                    "natural_rounding": ["rounding"], "natural_residual": ["rounding"],
                    "natural_teacher_rounding": ["rounding"],
                    "natural_random_subspace": ["rounding"], "natural_frequency_subspace": ["rounding"],
                    "natural_contrastive_subspace": ["rounding"],
                    "natural_rounding_scale": ["rounding", "per_channel_scale"],
                    "natural_spectral": ["rounding"],
                    "natural_qat_scale": ["bounded_multi_cell_code_offsets", "per_channel_scale"],
                    "natural_gan_qat": ["bounded_multi_cell_code_offsets"],
                    "natural_gan_finetune": ["all_decoder_parameters_including_bias_and_norm"],
                    "natural_residual_qat": ["bounded_multi_cell_code_offsets"],
                    "natural_full_finetune": ["all_decoder_parameters_including_bias_and_norm"],
                    "natural_qat_purification": ["bounded_multi_cell_code_offsets"]}[method],
                "status": "selected" if best["feasible"] else ("selected_quality_failed" if policy == "report" else "no_feasible_candidate")}
    save_json(folder / "selection.json", selected)
    if discriminator is not None:
        from safetensors.torch import save_file
        discriminator_folder = folder if heavy_folder is None else Path(heavy_folder)
        discriminator_folder.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in discriminator.state_dict().items()},
                  str(discriminator_folder / "discriminator_final.safetensors"))
        save_json(folder / "discriminator.json", {"architecture": "wmq_objectives.NaturalDiscriminator",
            "training_split": "natural_train_only", "checkpoint": "final_not_selected_generator_step",
            "selection_uses_discriminator": False, "valid_updates": disc_valid_updates,
            "paper_reproduction": False})
    if not selected["search_feasible"]:
        quality_warning(diagnostics_root, "branch_quality", "Selected candidate failed quality thresholds; retaining diagnostic result",
                        method=method, metrics=selected, thresholds=thresholds,
                        action="continue_with_diagnostic_fallback")
    return grids, selected, rows


@torch.no_grad()
def export_quantizer(vae, names, grids, folder):
    """Save auditable integer codes/scales beside the dequantized VAE artifact."""
    from safetensors.torch import save_file
    tensors, spec = {}, {}
    for name, grid in zip(names, grids):
        scale = grid.scale.detach()
        weight = grid(False)
        codes = (weight / scale).round()
        if (not torch.isfinite(weight).all() or not torch.isfinite(scale).all()
                or (scale <= 0).any() or (codes < grid.qmin).any() or (codes > grid.qmax).any()
                or not torch.allclose(weight, codes * scale, atol=1e-7, rtol=1e-6)):
            raise ValueError(f"Invalid quantizer export: {name}")
        if not torch.equal(vae.decoder.get_parameter(name), weight):
            raise ValueError(f"Materialized weight differs from hard quantizer: {name}")
        tensors[name + ".codes"] = codes.to(torch.int8).cpu().contiguous()
        tensors[name + ".scale"] = scale.cpu().contiguous()
        spec[name] = {"bits": grid.bits, "zero_point": 0, "qmin": grid.qmin, "qmax": grid.qmax}
    save_file(tensors, str(folder / "quantizer.safetensors"))
    save_json(folder / "quantizer.json", {"scheme": "signed_per_output_channel_zero_point_0_full_range", "layers": spec,
              "execution": "simulated PTQ; dequantized FP32 weights, not a certified backend kernel"})


def export_finetune_rtn(vae, names, bits, clip, folder):
    """Quantize selected weights of an already fine-tuned decoder; keep its learned bias/norm."""
    from safetensors.torch import save_file
    before = {k: v.detach().cpu().clone() for k, v in vae.decoder.state_dict().items()}
    folder.mkdir(parents=True, exist_ok=False)
    try:
        grids = nn.ModuleList([RoundingGrid(vae.decoder.get_parameter(k), bits, clip) for k in names])
        materialize(vae.decoder, names, grids)
        export_quantizer(vae, names, grids, folder)
        # Full state is needed: nonquantized bias/norm also changed during FT.
        save_file({k: v.detach().cpu().contiguous() for k, v in vae.decoder.state_dict().items()},
                  str(folder / "decoder_fp32.safetensors"))
        vae.save_pretrained(folder / "vae", safe_serialization=True)
        save_json(folder / "finetune.json", {"format": "finetune_then_rtn", "bits": bits,
            "quantized_names": names, "remaining_parameters": "fine-tuned FP32 bias/norm or excluded weights",
            "threat_model": "unrestricted_finetune_then_quantize_not_quantizer_only"})
    finally:
        vae.decoder.load_state_dict(before)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Diffusers pipeline already containing the marked VAE")
    p.add_argument("--prompts", required=True, help="UTF-8 text file; one unique prompt per line")
    p.add_argument("--evaluation-stage", choices=["development", "final"], default="development")
    p.add_argument("--development-manifests", nargs="*", default=[])
    p.add_argument("--output", required=True, help="New directory, never overwrite an experiment")
    p.add_argument("--image-output", help="Separate NEW image directory; default output_artifacts/images/<run>")
    p.add_argument("--artifact-output", help="Separate NEW checkpoint directory; default output_artifacts/checkpoints/<run>")
    p.add_argument("--gen-batch-size", type=int, default=0, help="Inference batch; 0 starts at 8 and halves on CUDA OOM")
    p.add_argument("--eval-batch-size", type=int, default=0, help="VAE/metric batch; 0 starts at 32 and halves on CUDA OOM")
    p.add_argument("--train-batch-size", type=int, default=1, help="Actual generator batch; reduce on OOM, never silently alter budget")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--lr-schedule", choices=["constant", "warmup_cosine"], default="constant")
    p.add_argument("--data-cache", choices=["auto", "cpu"], default="auto")
    p.add_argument("--cache-max-gib", type=float, default=16., help="Maximum total GPU data cache; preserve at least half total VRAM free when promoting")
    p.add_argument("--log-every", type=int, default=50, help="Flush accumulated update rows every N steps, on failure, and at branch completion")
    p.add_argument("--train-n", type=int, default=32)
    p.add_argument("--search-n", type=int, default=20)
    p.add_argument("--test-n", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000, help="Update/proposal budget per optimized branch and grid cell")
    p.add_argument("--qat-steps", type=int, help="Independent QAT update budget; default --steps")
    p.add_argument("--ft-steps", type=int, help="Independent FP32 update budget; default --steps")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--sampling-steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--qat-lr", type=float, default=.001, help="Code-offset LR, independent of sigmoid rounding LR")
    p.add_argument("--ft-lr", type=float, default=1e-5, help="FP32 decoder control LR in weight units")
    p.add_argument("--ft-preserve-weight", type=float, default=0., help="Optional generated low-pass preservation in FP32 control; 0 is natural reconstruction only")
    p.add_argument("--warmup-steps", type=int, default=10, help="QAT/FP32 linear warmup followed by cosine decay")
    p.add_argument("--bits", type=int, nargs="+", default=[4])
    p.add_argument("--clips", type=float, nargs="+", default=[1.])
    p.add_argument("--methods", nargs="+", choices=["fixed_ptq", "sensitivity",
                   "rounding", "rounding_scale", "reconstruction", "block_reconstruction"],
                   default=["fixed_ptq", "reconstruction", "block_reconstruction"])
    p.add_argument("--profile-n", type=int, default=16, help="Train-only samples for measured layer sensitivity")
    p.add_argument("--profile-bootstrap", type=int, default=1000, help="CPU paired resamples for layer priority uncertainty")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU is for small diagnostic pipelines")
    p.add_argument("--cuda-math", choices=["strict", "tf32"], default="strict",
                   help="tf32 accelerates FP32 convolutions/matmuls on supported CUDA GPUs; recorded in manifest")
    p.add_argument("--scope", choices=["all", "late"], default="all")
    p.add_argument('--owner-informed-steering-plan',
                   help='Explicit DEVELOPMENT-only layer mask from owner diagnostics; changes threat model, never claimed blind')
    p.add_argument("--strength", type=float, default=.25, help="Fixed blind smoothing hypothesis; 0 = reconstruction control")
    p.add_argument("--min-psnr", type=float, default=25.)
    p.add_argument("--min-image-psnr", type=float, default=22.)
    p.add_argument("--min-ssim", type=float, default=.9)
    p.add_argument("--preserve-weight", type=float, default=2.)
    p.add_argument("--natural-images", help="Optional directory of unpaired natural non-watermarked images")
    p.add_argument("--natural-train-n", type=int, help="Natural TRAIN count independent of prompt count; default --train-n")
    p.add_argument("--natural-search-n", type=int, help="Natural SEARCH count independent of prompt count; default --search-n")
    p.add_argument("--natural-methods", nargs="+", choices=NATURAL_METHODS,
                   default=list(NATURAL_METHODS[:10]), help="Natural controls and exploratory objectives; FP32 controls outside bitwidth grid")
    p.add_argument("--exclude-method-bits", nargs="+", default=[], metavar="METHOD:BITS",
                   help="Do not declare matching quantized branches, independent of clip (for example natural_residual:8)")
    p.add_argument("--natural-resolution", type=int, choices=[256, 512], default=512)
    p.add_argument("--natural-preservation", choices=["legacy", "lowpass"], default="legacy",
                   help="lowpass matches preservation form across natural quantized branches for residual ablations")
    p.add_argument("--spectral-weight", type=float, default=.1)
    p.add_argument("--adversarial-weight", type=float, default=.1)
    p.add_argument("--discriminator-lr", type=float, default=.001)
    p.add_argument("--residual-patch", type=int, choices=[4, 8, 16], default=8)
    p.add_argument("--residual-rank", type=int, default=8)
    p.add_argument("--residual-patches-per-image", type=int, default=256)
    p.add_argument("--residual-weight", type=float, default=1., help="Extra projected natural reconstruction loss; 0 for ablation")
    p.add_argument("--subspace-seeds", type=int, nargs="+", default=[1701, 1702, 1703])
    p.add_argument("--subspace-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--subspace-ridge", type=float, default=.01)
    p.add_argument("--finetune-rtn-bits", type=int, nargs="*", default=[], help="Derive RTN controls from the same selected FP32 decoder; outside quantizer-only threat model")
    p.add_argument("--teacher-weight", type=float, default=1., help="Generated TRAIN/SEARCH teacher MSE coefficient")
    p.add_argument("--teacher-source", choices=["purified", "marked"], default="purified", help="SEARCH-selected FP32 control, or original marked decoder for ablation")
    p.add_argument("--quality-constraint", choices=["off", "dual"], default="off")
    p.add_argument("--budget-psnr", type=float, default=30.)
    p.add_argument("--budget-ssim", type=float, default=.9)
    p.add_argument("--budget-max-violation", type=float, default=.1)
    p.add_argument("--dual-lr", type=float, default=.01)
    p.add_argument("--gradient-diagnostics-every", type=int, default=0)
    p.add_argument("--residual-preservation", choices=["full", "orthogonal"], default="orthogonal",
                   help="Residual branch only: protect complement plus 0.1 full RGB MSE, or full RGB control")
    p.add_argument("--natural-perceptual-weight", type=float, default=.1,
                   help="LPIPS coefficient ONLY for natural branches; 0 disables the external perceptual prior")
    p.add_argument("--qat-semantic-preserve-weight", type=float, default=2.,
                   help="QAT purification weight for low-frequency generated-image consistency")
    p.add_argument("--qat-trust-weight", type=float, default=.01,
                   help="QAT purification L2 penalty on code-space displacement")
    p.add_argument("--qat-max-code-shift", type=float, default=2.,
                   help="Maximum continuous shadow displacement in quantization-code units")
    p.add_argument("--branch-error-policy", choices=["report", "raise"], default="report",
                   help="Record failed branches and evaluate completed ones; setup errors still fail")
    p.add_argument("--evaluate-final", action=argparse.BooleanOptionalAction, default=False,
                   help="Also freeze/evaluate the last training state if search selected an earlier state; never choose using owner metrics")
    p.add_argument("--quality-policy", choices=["report", "constrained"], default="report",
                   help="report: choose by objective and record quality failures; constrained: prefer feasible candidates; neither stops on a failed gate")
    return p


def main():
    args = parser().parse_args()
    if not math.isfinite(args.teacher_weight) or args.teacher_weight <= 0:
        raise ValueError("--teacher-weight must be finite and positive")
    uses_teacher = bool(args.natural_images and "natural_teacher_rounding" in args.natural_methods)
    if uses_teacher and args.teacher_source == 'purified' and 'natural_full_finetune' not in args.natural_methods:
        raise ValueError("Purified teacher requires natural_full_finetune in --natural-methods")
    QualityBudget(args.budget_psnr, args.budget_ssim, args.dual_lr)
    if (not 0 <= args.budget_max_violation <= 1 or args.gradient_diagnostics_every < 0
            or not math.isfinite(args.subspace_ridge) or args.subspace_ridge <= 0
            or any(s < 0 for s in args.subspace_seeds) or len(set(args.subspace_seeds)) != len(args.subspace_seeds)
            or any(b not in (2, 3, 4, 6, 8) for b in args.finetune_rtn_bits)
            or len(set(args.finetune_rtn_bits)) != len(args.finetune_rtn_bits)):
        raise ValueError("Invalid scientific ablation configuration")
    if args.finetune_rtn_bits and (not args.natural_images or "natural_full_finetune" not in args.natural_methods):
        raise ValueError("--finetune-rtn-bits requires natural images and natural_full_finetune")
    EVENTS.clear()
    BATCH_LIMITS.clear()
    if args.train_batch_size < 1 or any(not math.isfinite(v) or v < 0 for v in
            (args.weight_decay, args.spectral_weight, args.adversarial_weight)) or not math.isfinite(args.discriminator_lr) or args.discriminator_lr <= 0:
        raise ValueError("Invalid training batch/objective/discriminator configuration")
    if min(args.gen_batch_size, args.eval_batch_size, args.cache_max_gib) < 0 or args.log_every < 1 or not math.isfinite(args.cache_max_gib):
        raise ValueError("Invalid runtime batch/cache/log configuration")
    if min(args.train_n, args.search_n, args.test_n, args.steps, args.eval_every, args.sampling_steps, args.profile_n, args.profile_bootstrap) < 1:
        raise ValueError("Sample counts and steps must be positive")
    if any(v is not None and v < 1 for v in (args.natural_train_n, args.natural_search_n, args.qat_steps, args.ft_steps)):
        raise ValueError("Natural split counts and branch step budgets must be positive")
    if (args.warmup_steps < 0 or min(args.qat_lr, args.ft_lr) <= 0 or args.ft_preserve_weight < 0
            or not all(math.isfinite(v) for v in (args.qat_lr, args.ft_lr, args.ft_preserve_weight))):
        raise ValueError("Invalid QAT/FP32 optimizer configuration")
    if (not 0 <= args.strength <= 1 or not 0 < args.min_ssim <= 1
            or args.lr <= 0 or args.preserve_weight < 0 or args.natural_perceptual_weight < 0 or args.guidance <= 1
            or args.qat_semantic_preserve_weight < 0 or args.qat_trust_weight < 0 or args.qat_max_code_shift <= 0
            or not all(math.isfinite(v) for v in [args.strength, args.min_ssim, args.lr,
                args.preserve_weight, args.natural_perceptual_weight, args.qat_semantic_preserve_weight,
                args.qat_trust_weight, args.qat_max_code_shift, args.guidance, args.min_psnr, args.min_image_psnr, *args.clips])):
        raise ValueError("Invalid numeric arguments")
    if any(b not in (2, 3, 4, 6, 8) for b in args.bits) or any(not 0 < c <= 1 for c in args.clips):
        raise ValueError("Invalid grid")
    excluded_method_bits = parse_method_bit_exclusions(args.exclude_method_bits)
    if (not 1 <= args.residual_rank <= 3 * (args.residual_patch ** 2 - 1) or args.residual_patches_per_image < 1
            or not math.isfinite(args.residual_weight) or args.residual_weight < 0):
        raise ValueError("Invalid residual configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    model = Path(args.model).resolve()
    if not (model / "model_index.json").is_file():
        raise ValueError("Supply a local, already fingerprinted Diffusers pipeline")
    prompts = [s.strip() for s in Path(args.prompts).read_text(encoding="utf-8").splitlines() if s.strip()]
    n = args.train_n + args.search_n + args.test_n
    if len(prompts) < n or len(set(prompts[:n])) != n:
        raise ValueError(f"Need {n} distinct prompts for disjoint train/search/test splits")
    prompts = prompts[:n]
    protocol_audit = audit_prompt_protocol(prompts, args.train_n + args.search_n, args.evaluation_stage, args.development_manifests)
    if args.evaluation_stage == "final" and args.owner_informed_steering_plan:
        raise ValueError("Owner-informed steering cannot be declared blind final evaluation")
    natural_manifest = natural_image_manifest(args.natural_images, args.natural_train_n or args.train_n,
        args.natural_search_n or args.search_n, args.seed, args.natural_resolution) if args.natural_images else None
    print(f"Training protocol: batch={args.train_batch_size}; rounding steps={args.steps}; "
          f"QAT steps={args.qat_steps or args.steps}; FP32 steps={args.ft_steps or args.steps}; "
          f"optimizer={args.optimizer}; FP32 LR={args.ft_lr}; "
          f"natural train/search={len(natural_manifest['train']) if natural_manifest else 0}/"
          f"{len(natural_manifest['search']) if natural_manifest else 0}; "
          f"natural resolution={args.natural_resolution}; preservation={args.natural_preservation}", flush=True)
    out = Path(args.output).resolve()
    output_parent = out.parent.parent if out.parent.name == "output_attack" else out.parent
    image_out = Path(args.image_output).resolve() if args.image_output else output_parent / "output_artifacts" / "images" / out.name
    artifact_out = Path(args.artifact_output).resolve() if args.artifact_output else output_parent / "output_artifacts" / "checkpoints" / out.name
    if out == model or model in out.parents:
        raise ValueError("Output must be outside the input model directory")
    if image_out == model or model in image_out.parents:
        raise ValueError("Image output must be outside the input model directory")
    if artifact_out == model or model in artifact_out.parents:
        raise ValueError("Artifact output must be outside the input model directory")
    roots = {"report": out, "image": image_out, "artifact": artifact_out}
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name < right_name and (left == right or left in right.parents or right in left.parents):
                raise ValueError(f"{left_name.title()} and {right_name} directories must be separate, not nested")
    if image_out.exists():
        raise FileExistsError(f"Image output already exists: {image_out}")
    if artifact_out.exists():
        raise FileExistsError(f"Artifact output already exists: {artifact_out}")
    out.mkdir(parents=True, exist_ok=False)
    image_out.mkdir(parents=True, exist_ok=False)
    artifact_out.mkdir(parents=True, exist_ok=False)
    try:
        image_location = Path(os.path.relpath(image_out, out)).as_posix()
    except ValueError:  # Separate Windows drives cannot have a relative path.
        image_location = str(image_out)
    try:
        artifact_location = Path(os.path.relpath(artifact_out, out)).as_posix()
    except ValueError:
        artifact_location = str(artifact_out)
    print(f"Report directory: {out}\nImage directory: {image_out}\nCheckpoint directory: {artifact_out}", flush=True)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # Both profiles are explicit and recorded; neither promises cross-GPU bitwise identity.
    use_tf32 = args.device == "cuda" and args.cuda_math == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(str(model), local_files_only=True,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                safety_checker=None, requires_safety_checker=False)
    # Reload VAE directly in FP32 (casting an already-rounded FP16 VAE is insufficient).
    from diffusers import AutoencoderKL
    pipe.vae = AutoencoderKL.from_pretrained(str(model / "vae"), local_files_only=True,
                                            torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(args.device)
    memory_before_batches = None
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info(args.device)
        memory_before_batches = {"free_gib": free / 2 ** 30, "total_gib": total / 2 ** 30}
    args.gen_batch_size = batch_size(args.gen_batch_size, args.device, cap=8)
    args.eval_batch_size = batch_size(args.eval_batch_size, args.device, cap=32)
    data_cache = DataCache(args.device, args.data_cache, args.cache_max_gib)
    print(f"Inference batch={args.gen_batch_size}; evaluation batch={args.eval_batch_size}; data cache={args.data_cache}", flush=True)
    vae = pipe.vae.eval().requires_grad_(False)
    names = select_names(vae, args.scope)
    steering_plan = None
    if args.owner_informed_steering_plan:
        plan_path = Path(args.owner_informed_steering_plan)
        steering_plan = json.loads(plan_path.read_text())
        layers = steering_plan.get('layers')
        if (steering_plan.get('source_split') != 'victim_test' or steering_plan.get('uses_owner_information') is not True
                or not isinstance(layers, list) or not layers or any(not isinstance(n, str) for n in layers)
                or len(set(layers)) != len(layers) or set(layers) - set(names)):
            raise ValueError('Invalid owner-informed development steering plan or layer names')
        args._steering_layers = layers
        steering_plan = {**steering_plan, 'plan_sha256': sha(plan_path)}
    pristine = {k: v.detach().cpu().clone() for k, v in vae.decoder.state_dict().items()}
    script = Path(__file__).resolve()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script.parent,
                                 capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        commit = None  # A copied source folder need not have git installed.
    manifest = {"schema_version": 2, "threat_model": "marked_model_only_no_detector_no_key_no_clean_model",
                "method": "pilot_blind_quantization_ladder_not_OS_MQ", "args": vars(args),
                "ownership_loss": None, "surrogate_transfer_evaluated": False,
                "git_commit": commit, "script_sha256": sha(script),
                "helper_sources_sha256": {name: sha(script.parent / name) for name in
                    ("wmq_runtime.py", "wmq_diagnostics.py", "wmq_residual.py", "wmq_objectives.py", "wmq_science.py", "wmq_teacher.py")},
                "model_files_sha256": {str(p.relative_to(model)): sha(p) for p in sorted(model.rglob("*"))
                      if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")},
                "prompts": prompts, "seeds": list(range(args.seed, args.seed + n)),
                "target_names": names, "changed_parameters": sum(vae.decoder.get_parameter(k).numel() for k in names),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "reproducibility": {"python": platform.python_version(), "platform": platform.platform(),
                    "cudnn": torch.backends.cudnn.version(), "tf32_matmul": use_tf32, "tf32_cudnn": use_tf32,
                    "cudnn_benchmark": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cuda_memory_before_auto_batch": memory_before_batches,
                    "warning": "Seeds and TF32 settings do not guarantee bitwise reproducibility across GPUs or library versions."},
                "quality_metric_definition": {"ssim": "Gaussian 11x11 sigma=1.5; population covariance; valid crop; RGB channel mean; data_range=1",
                    "ssim_statistics_dtype": "float64", "psnr": "per-image RGB MSE; data_range=1; cap 120dB"},
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "inference": f"FP32 VAE; {'FP16' if args.device == 'cuda' else 'FP32'} UNet; simulated low-bit weights",
                "sampling_scheduler": dict(pipe.scheduler.config),
                "watermark_success": "unknown; requires independent owner evaluation",
                "quality_gate": "PSNR/SSIM only; not equivalent to legacy LPIPS gate",
                "quality_policy": args.quality_policy,
                "qat_purification": {"hard_forward": True, "backward": "straight_through_estimator",
                    "export": "integer codes at requested bitwidth and fixed per-output-channel scale",
                    "owner_feedback": False, "preservation": "8x low-pass generated-reference consistency"},
                "natural_perceptual_prior": "frozen LPIPS AlexNet v0.1; public pretrained features; natural branches only" if natural_manifest and args.natural_perceptual_weight else None}
    plan = []
    for bits in dict.fromkeys(args.bits):
        for clip in dict.fromkeys(args.clips):
            for method in dict.fromkeys([*args.methods, *(args.natural_methods if natural_manifest else [])]):
                if method in FLOAT_METHODS:
                    continue  # No duplicate FP32 control for every bitwidth/clip.
                if (method, bits) in excluded_method_bits:
                    continue
                branch_id = f"{method}_w{bits}_c{clip}"
                plan.append({"label": branch_id + "_test", "method": method,
                             "comparison_group": f"w{bits}_c{clip}_{args.scope}",
                             "bits": bits, "clip": clip, "role": "candidate",
                             "threat_model": "marked_model_plus_unpaired_natural_images" if method.startswith("natural_") else "marked_model_only",
                             "artifact": f"branches/{branch_id}"})
    for method in FLOAT_METHODS:
        if not natural_manifest or method not in args.natural_methods:
            continue
        plan.append({"label": f"{method}_fp32_test", "method": method,
                     "comparison_group": "fp32_unrestricted_decoder_control", "bits": 32, "clip": 1.,
                     "role": "finetune_control", "artifact_format": "decoder_fp32",
                     "threat_model": "marked_model_plus_unpaired_natural_images_unrestricted_finetune",
                     "artifact": f"branches/{method}_fp32"})
    manifest["reference_label"] = "marked_reference_test"
    manifest["evaluation_protocol"] = protocol_audit
    manifest['teacher_protocol'] = {'enabled': uses_teacher, 'source': args.teacher_source,
        'selection': 'natural SEARCH objective and generated SEARCH quality; no owner feedback',
        'student_source': 'original marked decoder; rounding only; original bias/norm frozen',
        'targets': 'generated TRAIN and SEARCH only; no TEST targets during optimization',
        'auxiliary_finetuning_allowed': uses_teacher and args.teacher_source == 'purified',
        'teacher_watermark_status': 'unknown until owner evaluation after all selections freeze',
        'compute': 'teacher training is shared with FP32 control; include teacher cost once'}
    manifest["derived_control_protocol"] = {"finetune_rtn_bits": args.finetune_rtn_bits,
        "source": "search-selected natural_full_finetune", "additional_training": False,
        "owner_feedback": False, "clip": 1., "scope": args.scope}
    # Random seeds change only the null basis, never prompts/minibatch ordering.
    expanded_plan = []
    for branch in plan:
        if branch["method"] == "natural_random_subspace":
            for seed in args.subspace_seeds:
                expanded_plan.append({**branch, "subspace_seed": seed,
                    "label": branch["label"].removesuffix("_test") + f"_rseed{seed}_test",
                    "artifact": branch["artifact"] + f"_rseed{seed}"})
        else:
            expanded_plan.append(branch)
    plan = expanded_plan
    manifest["image_root"] = image_location
    manifest["artifact_root"] = artifact_location
    manifest["test_branches"] = [{"label": "marked_reference_test", "role": "reference",
                                  "method": "marked_reference"},
                                 {"label": "pseudo_target_test", "role": "diagnostic", "method": "direct_smoothing",
                                  "strength": args.strength}, *plan]
    manifest["natural_dataset"] = natural_manifest
    manifest["excluded_method_bits"] = [f"{method}:{bits}" for method, bits in sorted(excluded_method_bits)]
    if natural_manifest:
        manifest["threat_model"] = "separate_model_only_and_model_plus_unpaired_natural_branches"
    manifest['owner_informed_steering'] = steering_plan
    if steering_plan is not None:
        if natural_manifest is None:
            raise ValueError('Owner-informed steering requires natural branches')
        manifest['threat_model'] = 'includes_owner_informed_DEVELOPMENT_not_blind'
        manifest['method'] = 'owner_informed_layer_mask_development_ladder'
        for branch in plan:
            if branch['method'].startswith('natural_') and branch['method'] not in FLOAT_METHODS:
                branch['threat_model'] = 'owner_informed_development_not_blind'
                branch['comparison_group'] += '_owner_informed'
    manifest["sensitivity_kind"] = "visual_proxy_only_not_ownership_sensitivity"
    manifest["proxy_limitations"] = ["Smoothing may retain fingerprint", "Proxy improvement may erase natural texture",
                                      "Owner metrics are evaluation-only after every branch is frozen"]
    manifest["comparison_note"] = "Compare only within comparison_group; optimization costs reported, not equalized."
    manifest["endpoint_policy"] = "Evaluate fixed final step in addition to search selection when different" if args.evaluate_final else "Search-selected checkpoint only"
    save_json(out / "manifest.json", manifest)
    @torch.no_grad()
    def cache(indices):
        # Test data are not generated until the chosen checkpoint is exported.
        pipe.unet.to(args.device)
        pipe.text_encoder.to(args.device)
        latents = []
        def generate(chunk):
            # Recreate each generator on OOM retry: never reuse an advanced RNG.
            generators = [torch.Generator(device=args.device).manual_seed(args.seed + i) for i in chunk]
            prompt_batch = [prompts[i] for i in chunk]
            z = pipe(prompt_batch if len(chunk) > 1 else prompt_batch[0],
                     generator=generators if len(chunk) > 1 else generators[0], num_inference_steps=args.sampling_steps,
                     guidance_scale=args.guidance, height=512, width=512, output_type="latent").images
            return (z.float() / float(vae.config.scaling_factor)).cpu()
        for start, z in batches(list(indices), generate, args.gen_batch_size, "generate"):
            latents.extend(z.split(1))
            print(f"cached latents {start + len(z)}/{len(indices)}", flush=True)
        pipe.unet.to("cpu")
        pipe.text_encoder.to("cpu")
        if args.device == "cuda":
            torch.cuda.empty_cache()
        references = []
        def decode(chunk):
            return decoded01(vae.decode(torch.cat(chunk).to(args.device), return_dict=False)[0]).cpu()
        for _, x in batches(latents, decode, args.eval_batch_size, "reference_decode"):
            references.extend(x.split(1))
            print(f"decoded references {len(references)}/{len(latents)}", flush=True)
        return latents, references

    a, b = args.train_n, args.train_n + args.search_n
    zs, refs = cache(range(b))
    zs = data_cache.promote(zs, "generated_train_search_latents")
    refs = data_cache.promote(refs, "generated_train_search_references")
    search_z, search_ref = zs[a:b], refs[a:b]
    natural_train, natural_search, perceptual = None, None, None
    residual, residual_info = None, None
    residual_variants = {}
    # Controls first, then teacher/student, then residual ablations. Owner evaluation
    # still waits for every branch to freeze; no TEST-driven scheduling.
    plan.sort(key=lambda branch: (0 if not branch['method'].startswith('natural_') else
        (1 if uses_teacher and args.teacher_source == 'purified' and branch['method'] == 'natural_full_finetune' else
         2 if branch['method'] in ('natural_rounding', 'natural_teacher_rounding') else 3), branch['bits']))
    teacher_targets = None
    rows, selections, frozen_branches = [], {}, {}
    profiles = {}
    costs = {}
    branch_failures = []
    endpoint_plan = []
    declared_plan = list(plan)
    run_status = {"phase": "branch_search", "declared_branches": [b["label"] for b in declared_plan],
                  "completed_branches": [], "failed_branches": [], "current_branch": None}
    save_json(out / "run_status.json", run_status)
    for branch in plan:
        try:
            run_status["current_branch"] = branch["label"]
            save_json(out / "run_status.json", run_status)
            if args.device == "cuda":
                torch.cuda.synchronize()
            branch_started = time.monotonic()
            bits, clip, method = branch["bits"], branch["clip"], branch["method"]
            float_branch = method in FLOAT_METHODS
            branch_names = [name for name, _ in vae.decoder.named_parameters()] if float_branch else names
            vae.decoder.load_state_dict(pristine)
            group = branch["comparison_group"]
            if method == "sensitivity" and group not in profiles:
                count = min(a, args.profile_n)
                profiles[group] = profile_layers(vae, names, zs[:count], refs[:count], bits, clip, args)
                save_json(out / f"profile_{group}.json",
                          {"split": "train", "n": count, "ownership_signal": False, "layers": profiles[group],
                           "bootstrap_resamples": args.profile_bootstrap, "bootstrap_seed": args.seed,
                           "uncertainty": "Pointwise percentile intervals conditional on calibration pool; not simultaneous ranking confidence; sorting uses point estimates."})
                if count < 16:
                    quality_warning(out, "profile_sample_size", "Small calibration profile; priorities may be unstable",
                                    metrics={"actual_n": count, "requested_n": args.profile_n}, action="continue")
            folder = out / branch["artifact"]
            artifact_folder = artifact_out / branch["artifact"]
            natural = method.startswith("natural_")
            if natural and natural_train is None:
                natural_train = cache_natural(vae, natural_manifest["train"], args.eval_batch_size, args.natural_resolution)
                natural_search = cache_natural(vae, natural_manifest["search"], args.eval_batch_size, args.natural_resolution)
                natural_train = tuple(data_cache.promote(v, f"natural_train_{i}") for i, v in enumerate(natural_train))
                natural_search = tuple(data_cache.promote(v, f"natural_search_{i}") for i, v in enumerate(natural_search))
                if args.natural_perceptual_weight:
                    import lpips
                    perceptual = lpips.LPIPS(net="alex", version="0.1").to(args.device).eval().requires_grad_(False)
            train_z, train_ref = natural_train if natural else (zs[:a], refs[:a])
            if method == 'natural_teacher_rounding' and teacher_targets is None:
                teacher_cache_started = time.monotonic()
                if args.teacher_source == 'marked':
                    teacher_images = refs
                    provenance = {'source': 'marked', 'label': 'marked_reference', 'additional_image_forwards': 0}
                else:
                    from wmq_teacher import cache_teacher_targets
                    teacher_label = 'natural_full_finetune_fp32_test'
                    if teacher_label not in frozen_branches:
                        raise RuntimeError('Selected purification teacher unavailable; student cannot run')
                    relative = 'branches/natural_full_finetune_fp32/decoder_fp32.safetensors'
                    expected = frozen_branches[teacher_label]['files_sha256'][relative]
                    teacher_images = cache_teacher_targets(vae, artifact_out / relative, expected, zs, args.eval_batch_size)
                    provenance = {'source': 'purified', 'label': teacher_label,
                        'selected_step': selections[teacher_label]['step'], 'checkpoint_sha256': expected,
                        'search_feasible': selections[teacher_label]['search_feasible'],
                        'additional_image_forwards': len(zs), 'owner_feedback': False}
                from wmq_teacher import target_signal
                provenance['target_signal'] = {
                    'train': target_signal(teacher_images[:a], refs[:a]),
                    'search': target_signal(teacher_images[a:b], refs[a:b])}
                if provenance['target_signal']['search']['near_identity']:
                    quality_warning(out, 'teacher_target', 'Teacher SEARCH targets are numerically near the marked reference',
                        metrics=provenance['target_signal']['search'], action='continue_with_diagnostic')
                teacher_images = data_cache.promote(teacher_images, 'teacher_train_search_targets') if args.teacher_source == 'purified' else teacher_images
                provenance['target_cache_seconds'] = time.monotonic() - teacher_cache_started
                save_json(out / 'teacher_target_diagnostics.json', provenance)
                print(f"Teacher source={args.teacher_source}; selected_step={provenance.get('selected_step')}; "
                      f"SEARCH reference MSE={provenance['target_signal']['search']['reference_mse']:.6g}; "
                      f"near_identity={provenance['target_signal']['search']['near_identity']}", flush=True)
                teacher_targets = {'train': teacher_images[:a], 'search': teacher_images[a:b], 'provenance': provenance}
            if method == 'natural_teacher_rounding':
                branch['teacher_dependency'] = teacher_targets['provenance']
            if method in RESIDUAL_METHODS and residual is None:
                from wmq_residual import fit_residual_subspace
                residual, residual_info = fit_residual_subspace(vae, *natural_train, args.residual_patch,
                    args.residual_rank, args.residual_patches_per_image, args.seed)
            branch_residual, branch_residual_info = None, None
            if method in RESIDUAL_METHODS:
                from wmq_residual import subspace_variant
                kind = {"natural_random_subspace": "random", "natural_frequency_subspace": "frequency",
                        "natural_contrastive_subspace": "contrastive"}.get(method, "pca")
                variant_key = (kind, branch.get("subspace_seed", 0))
                if variant_key not in residual_variants:
                    residual_variants[variant_key] = subspace_variant(residual, residual_info, kind,
                        branch.get("subspace_seed", 0), args.subspace_normalize, args.subspace_ridge)
                branch_residual, branch_residual_info = residual_variants[variant_key]
            grids, selected, branch_rows = optimize_branch(
                vae, branch_names, train_z, train_ref, search_z, search_ref, args,
                bits, clip, method, profiles.get(group, []), folder, diagnostics_root=out,
                natural_search=natural_search if natural else None,
                preserve_data=(zs[:a], refs[:a]) if natural else None,
                perceptual=perceptual if natural else None,
                residual=branch_residual,
                heavy_folder=artifact_folder,
                teacher_targets=teacher_targets if method == 'natural_teacher_rounding' else None)
            rows.extend(branch_rows)
            selections[branch["label"]] = selected
            materialize(vae.decoder, branch_names, grids)
            # No bias, norm, or unselected decoder weight may change.
            for name, value in vae.decoder.state_dict().items():
                if name not in branch_names and not torch.equal(value.cpu(), pristine[name]):
                    raise ValueError(f"Unexpected parameter change: {name}")
            artifact_folder.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(artifact_folder / "vae", safe_serialization=True)
            if float_branch:
                from safetensors.torch import save_file
                save_file({k: v.detach().cpu().contiguous() for k, v in vae.decoder.state_dict().items()},
                          str(artifact_folder / "decoder_fp32.safetensors"))
                save_json(artifact_folder / "finetune.json", {"format": "decoder_fp32", "quantized": False,
                    "changed_parameter_names": branch_names, "paper_reproduction": False,
                    "objective": "natural MSE + LPIPS; optional low-pass preservation",
                    "adversarial": method == "natural_gan_finetune",
                    "discriminator": "custom patch logistic discriminator, not HiDDeN" if method == "natural_gan_finetune" else None})
            else:
                export_quantizer(vae, names, grids, artifact_folder)
            if method in RESIDUAL_METHODS:
                from safetensors.torch import save_file
                save_file({"basis": branch_residual.basis.detach().cpu().contiguous()}, str(artifact_folder / "residual_basis.safetensors"))
                save_json(folder / "residual_calibration.json", branch_residual_info)
            frozen_branches[branch["label"]] = {
                "selected": selected,
                "files_sha256": {p.relative_to(artifact_out).as_posix(): sha(p)
                                 for p in sorted(artifact_folder.rglob("*")) if p.is_file()},
                "diagnostic_files_sha256": {p.relative_to(out).as_posix(): sha(p)
                                            for p in sorted(folder.rglob("*")) if p.is_file()}}
            if method == "natural_full_finetune":
                for derived_bits in args.finetune_rtn_bits:
                    derived_started = time.monotonic()
                    derived_id = f"natural_finetune_rtn_w{derived_bits}_c1.0"
                    derived_branch = {"label": derived_id + "_test", "method": "natural_finetune_rtn",
                        "bits": derived_bits, "clip": 1., "artifact": "branches/" + derived_id,
                        "artifact_format": "decoder_fp32", "parameter_space": "finetune_then_rtn",
                        "role": "finetune_control", "shares_training_with": branch["label"],
                        "comparison_group": f"finetune_then_w{derived_bits}_{args.scope}",
                        "threat_model": "marked_model_plus_unpaired_natural_images_unrestricted_finetune"}
                    destination = artifact_out / derived_branch["artifact"]
                    export_finetune_rtn(vae, names, derived_bits, 1., destination)
                    from safetensors.torch import load_file
                    vae.decoder.load_state_dict(load_file(str(destination / "decoder_fp32.safetensors"), device=args.device))
                    derived_quality = score(vae, search_z, search_ref, 0., args)
                    derived_quality.update(natural_reconstruction_metrics(vae, *natural_search, perceptual,
                        args.natural_perceptual_weight, args.eval_batch_size))
                    derived_selected = {**selected, **derived_quality, "method": "natural_finetune_rtn",
                        "candidate": derived_id + f"_step{selected['step']}",
                        "bits": derived_bits, "parameter_space": "finetune_then_rtn",
                        "search_feasible": derived_quality["feasible"], "shares_training_with": branch["label"],
                        "selected_source": "RTN_of_search_selected_FP32", "additional_training_updates": 0,
                        "status": "selected" if derived_quality["feasible"] else "selected_quality_failed"}
                    derived_folder = out / derived_branch["artifact"]
                    derived_folder.mkdir(parents=True, exist_ok=False)
                    save_json(derived_folder / "selection.json", derived_selected)
                    selections[derived_branch["label"]] = derived_selected
                    frozen_branches[derived_branch["label"]] = {"selected": derived_selected,
                        "files_sha256": {p.relative_to(artifact_out).as_posix(): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()},
                        "diagnostic_files_sha256": {p.relative_to(out).as_posix(): sha(p) for p in sorted(derived_folder.rglob("*")) if p.is_file()}}
                    costs[derived_branch["label"]] = {"shares_training_with": branch["label"],
                        "post_training_export_and_validation_seconds": time.monotonic() - derived_started}
                    endpoint_plan.append(derived_branch)
                    materialize(vae.decoder, branch_names, grids)
            if args.evaluate_final and selected['step'] != selected['attempted_updates']:
                # Endpoint existence depends only on the predeclared training schedule.
                # Both artifacts are frozen before any held-out latent/owner feedback.
                final_branch = {**branch, 'label': branch['label'].removesuffix('_test') + '_final_test',
                    'artifact': branch['artifact'] + '_final', 'role': 'endpoint_control',
                    'shares_training_with': branch['label']}
                final_selected = {**selected, **branch_rows[-1], 'checkpoint_policy': 'fixed_final_step',
                    'selected_at_budget_boundary': True,
                    'search_feasible': branch_rows[-1]['feasible'], 'shares_training_with': branch['label'],
                    'selected_source': 'fixed_final_step',
                    'status': 'selected' if branch_rows[-1]['feasible'] else 'selected_quality_failed'}
                grids.load_state_dict(grids.final_state)
                materialize(vae.decoder, branch_names, grids)
                final_artifact = artifact_out / final_branch['artifact']
                final_artifact.mkdir(parents=True, exist_ok=False)
                vae.save_pretrained(final_artifact / 'vae', safe_serialization=True)
                if float_branch:
                    from safetensors.torch import save_file
                    save_file({k: v.detach().cpu().contiguous() for k, v in vae.decoder.state_dict().items()},
                              str(final_artifact / 'decoder_fp32.safetensors'))
                    save_json(final_artifact / 'finetune.json', {'format': 'decoder_fp32', 'quantized': False,
                        'checkpoint_policy': 'fixed_final_step', 'shares_training_with': branch['label']})
                else:
                    export_quantizer(vae, names, grids, final_artifact)
                final_folder = out / final_branch['artifact']
                final_folder.mkdir(parents=True, exist_ok=False)
                save_json(final_folder / 'selection.json', final_selected)
                selections[final_branch['label']] = final_selected
                frozen_branches[final_branch['label']] = {'selected': final_selected,
                    'files_sha256': {p.relative_to(artifact_out).as_posix(): sha(p) for p in sorted(final_artifact.rglob('*')) if p.is_file()},
                    'diagnostic_files_sha256': {p.relative_to(out).as_posix(): sha(p) for p in sorted(final_folder.rglob('*')) if p.is_file()}}
                endpoint_plan.append(final_branch)
            del grids
            vae.decoder.load_state_dict(pristine)
            save_json(out / "search.json", rows)
            save_csv(out / "search.csv", rows)
            if args.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            costs[branch["label"]] = {"search_and_export_seconds_including_profile": time.monotonic() - branch_started,
                                     "profile_image_forwards": len(names) * min(a, args.profile_n) if method == "sensitivity" else 0}
            run_status["completed_branches"].append(branch["label"])
            run_status["current_branch"] = None
            run_status["branch_compute"] = costs
            save_json(out / "run_status.json", run_status)
        except Exception as error:
            if args.branch_error_policy == "raise":
                raise
            import traceback
            failure = {"label": branch["label"], "method": branch["method"],
                       "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            branch_failures.append(failure)
            run_status["failed_branches"].append(failure)
            run_status["current_branch"] = None
            save_json(out / "run_status.json", run_status)
            selections.pop(branch["label"], None)
            frozen_branches.pop(branch["label"], None)
            for endpoint in list(endpoint_plan):
                if endpoint['shares_training_with'] == branch['label']:
                    selections.pop(endpoint['label'], None)
                    frozen_branches.pop(endpoint['label'], None)
                    endpoint_plan.remove(endpoint)
            save_json(out / "branch_failures.json", branch_failures)
            print(f"FAILED {branch['label']}: {failure['error']}; continuing other branches", flush=True)
            # A partial artifact is retained for diagnosis but excluded from evaluation.
            vae.decoder.load_state_dict(pristine)
            if "grids" in locals():
                del grids
            if args.device == "cuda":
                torch.cuda.empty_cache()
    plan = [branch for branch in declared_plan if branch["label"] in frozen_branches] + endpoint_plan
    manifest["declared_branches"] = declared_plan
    manifest["branch_failures"] = branch_failures
    manifest["test_branches"] = [branch for branch in manifest["test_branches"]
                                 if branch["role"] in ("reference", "diagnostic") or branch["label"] in frozen_branches] + endpoint_plan
    save_json(out / "manifest.json", manifest)

    # Freeze ALL branches and the declared protocol before the first test latent.
    frozen = {"schema_version": 2, "phase": "before_test_generation",
              "branches": frozen_branches, "manifest_sha256": sha(out / "manifest.json"),
              "search_sha256": sha(out / "search.json"),
              "profile_sha256": {p.name: sha(p) for p in sorted(out.glob("profile_*.json"))}}
    save_json(out / "selection_frozen.json", frozen)
    run_status.update(phase="held_out_test", current_branch=None)
    save_json(out / "run_status.json", run_status)
    test_z, test_ref = cache(range(b, n))
    test_z = data_cache.promote(test_z, "test_latents")
    test_ref = data_cache.promote(test_ref, "test_references")
    reference_quality = score(vae, test_z, test_ref, args.strength, args, image_out / "marked_reference_test")
    quality = {"marked_reference_test": reference_quality}
    quality["pseudo_target_test"] = save_pseudo_diagnostic(test_ref, args.strength, args, image_out / "pseudo_target_test")
    from safetensors.torch import load_file
    for branch in plan:
        vae.decoder.load_state_dict(pristine)
        print(f"Test evaluation: {branch['label']} ({len(test_z)} images)", flush=True)
        if branch.get("artifact_format") == "decoder_fp32":
            tensors = load_file(str(artifact_out / branch["artifact"] / "decoder_fp32.safetensors"), device=args.device)
            vae.decoder.load_state_dict(tensors, strict=True)
        else:
            tensors = load_file(str(artifact_out / branch["artifact"] / "quantizer.safetensors"), device=args.device)
            with torch.no_grad():
                for name in names:
                    vae.decoder.get_parameter(name).copy_(tensors[name + ".codes"].float() * tensors[name + ".scale"])
        quality[branch["label"]] = score(vae, test_z, test_ref, args.strength, args, image_out / branch["label"])
        if not quality[branch["label"]]["feasible"]:
            quality_warning(out, "test_quality", "Branch failed held-out quality gate; results retained",
                            method=branch["label"], metrics=quality[branch["label"]],
                            thresholds={"min_psnr": args.min_psnr, "min_ssim": args.min_ssim,
                                        "min_image_psnr": args.min_image_psnr})
        del tensors
    complete = all(s["search_feasible"] and quality[label]["feasible"] for label, s in selections.items())
    save_csv(out / "quality_summary.csv", [{"method": label,
        "quality_policy": args.quality_policy, "test_quality_valid": metrics["feasible"],
        "search_quality_valid": selections.get(label, {}).get("search_feasible"),
        "selection_status": selections.get(label, {}).get("status"), **metrics} for label, metrics in quality.items()])
    save_json(out / "report.json", {"schema_version": 2, "selections": selections,
              "branch_compute": costs, "branch_failures": branch_failures,
              "runtime": {"gen_batch_size": args.gen_batch_size, "eval_batch_size": args.eval_batch_size,
                          "training_batch_size": args.train_batch_size, "learned_batch_limits": dict(BATCH_LIMITS),
                          "cache": data_cache.stats, "events": list(EVENTS), "log_every": args.log_every},
              "branch_quality": quality, "status": "branch_failures" if branch_failures else ("complete" if complete else "quality_failures"),
              "watermark_metrics": None, "watermark_success": "unknown",
              "test_used_for_selection": False, "selection_frozen_sha256": sha(out / "selection_frozen.json"),
              "image_root": image_location,
              "artifact_root": artifact_location,
              "test_images_sha256": {p.relative_to(image_out).as_posix(): sha(p)
                   for branch in manifest["test_branches"] for p in sorted((image_out / branch["label"]).glob("*.png"))},
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
              "note": "Failed controls are retained and flagged; no watermark removal or OS-MQ claim."})
    run_status.update(phase="attack_complete_owner_evaluation_pending")
    save_json(out / "run_status.json", run_status)


if __name__ == "__main__":
    main()

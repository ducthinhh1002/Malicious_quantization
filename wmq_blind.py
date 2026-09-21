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
from wmq_runtime import batches, batch_size, DataCache, all_finite, EVENTS, image01, decoded01


def blur(x):
    """Fixed Gaussian [1,4,6,4,1]/16, reflection padding, no learned prior."""
    k = x.new_tensor([1, 4, 6, 4, 1]) / 16
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, 5, 5)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel,
                    groups=x.shape[1])


def pseudo_target(reference, strength):
    # Both inputs and targets retain unknown watermark content. Never call clean.
    image01(reference, "pseudo-target reference")
    return reference.lerp(blur(reference), strength)


def natural_image_manifest(directory, train_n, search_n, seed):
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
            "preprocessing": "EXIF transpose, RGB, bicubic fit center crop 512x512",
            "assumption": "User-supplied natural unwatermarked images; not detector-certified",
            "paired_with_generated_images": False}


@torch.no_grad()
def cache_natural(vae, entries, eval_batch_size=1):
    from PIL import Image, ImageOps
    device = next(vae.parameters()).device
    latents, images = [], []
    for entry in entries:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Natural image changed after manifest creation")
        with Image.open(entry["path"]) as im:
            im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (512, 512),
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


class RoundingGrid(nn.Module):
    """Hard rounding, with optional bounded per-channel scale optimization.

    Forward uses hard integers with a sigmoid straight-through gradient. Export
    uses exactly the same grid; unconstrained full-precision weights never train.
    """
    def __init__(self, weight, bits, clip=1.0, learn_scale=False):
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
        self.alpha = nn.Parameter(torch.logit(frac.clamp(1e-4, 1 - 1e-4)))

    @property
    def scale(self):
        return self.base_scale * self.log_scale.clamp(math.log(.8), math.log(1.25)).exp()

    def forward(self, differentiable=True):
        soft = self.alpha.sigmoid()
        hard = (self.alpha >= 0).to(soft.dtype)
        rounding = hard + (soft - soft.detach()) if differentiable else hard
        scale = self.scale
        floor = (self.source / scale).clamp(self.qmin, self.qmax).detach().floor()
        return (floor + rounding).clamp(self.qmin, self.qmax) * scale

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
        metrics = torch.stack([p, s, mse(x - pseudo_target(ref, strength)), mse(high - ref_high), ratio], 1).cpu().tolist()
        return metrics, x.cpu() if folder is not None else None
    for start, (metrics, pixels) in batches(list(zip(latents, refs)), evaluate, batch_size(getattr(args, "eval_batch_size", 0), device), "score"):
        for j, (p, s, error, high_error, ratio) in enumerate(metrics):
            psnr.append(p); ssim.append(s); errors.append(error)
            texture.append({"highpass_mse": high_error, "highpass_energy_ratio": ratio if math.isfinite(ratio) else None})
            if folder is not None:
                array = (pixels[j].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                Image.fromarray(array).save(folder / f"{start+j:04d}.png")
    p, s = torch.tensor(psnr), torch.tensor(ssim)
    feasible = bool(p.mean() >= args.min_psnr and s.mean() >= args.min_ssim
                    and p.min() >= args.min_image_psnr)
    return {"psnr": p.mean().item(), "ssim": s.mean().item(),
            "psnr_p05": torch.quantile(p, .05).item(), "min_psnr": p.min().item(),
            "quality_violation_fraction": ((p < args.min_psnr) | (s < args.min_ssim)).float().mean().item(),
            "target_mse": sum(errors) / len(errors), "feasible": feasible,
            "per_image_psnr": psnr, "per_image_ssim": ssim, "per_image_texture": texture}


def choose(rows, policy="constrained"):
    eligible = list(rows) if policy == "report" else [r for r in rows if r["feasible"]]
    return min(eligible, key=lambda r: (r.get("selection_objective", r["target_mse"]), -r["psnr"], r["candidate"])) if eligible else None


@torch.no_grad()
def natural_reconstruction_metrics(vae, latents, images, perceptual=None, perceptual_weight=0., eval_batch_size=1,
                                   residual=None, residual_weight=0.):
    """Natural validation ranks reconstruction only; generated pairs supply quality gates."""
    if not latents or len(latents) != len(images):
        raise ValueError("Need nonempty paired natural latents and images")
    device = next(vae.parameters()).device
    errors, perceptual_errors, residual_errors = [], [], []
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
        return torch.stack([error, lp, rp], 1).cpu().tolist()
    for _, metrics in batches(list(zip(latents, images)), evaluate, eval_batch_size, "natural_validation"):
        for error, lp, rp in metrics:
            errors.append(error)
            residual_errors.append(rp)
            if perceptual is not None:
                perceptual_errors.append(lp)
    mse = sum(errors) / len(errors)
    lpips_value = sum(perceptual_errors) / len(perceptual_errors) if perceptual_errors else 0.
    if not math.isfinite(lpips_value):
        raise ValueError("Nonfinite natural validation perceptual loss")
    residual_value = sum(residual_errors) / len(residual_errors)
    return {"natural_validation_mse": mse, "natural_validation_lpips": lpips_value if perceptual is not None else None,
            "natural_validation_residual": residual_value if residual is not None else None,
            "target_mse": mse, "selection_objective": mse + perceptual_weight * lpips_value + residual_weight * residual_value}


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
                    natural_search=None, preserve_data=None, perceptual=None, residual=None):
    """Independent initialization, train-only updates, search-only checkpoint choice."""
    device = next(vae.parameters()).device
    pristine = {k: v.detach().clone() for k, v in vae.decoder.state_dict().items()}
    grids = nn.ModuleList([RoundingGrid(vae.decoder.get_parameter(k), bits, clip,
                           learn_scale=method in ("rounding_scale", "natural_rounding_scale")) for k in names]).to(device)
    rows, updates, best = [], [], None
    best_state = None
    natural = method.startswith("natural_")
    residual_branch = method == "natural_residual"
    if residual_branch and residual is None:
        raise ValueError("Residual branch requires a frozen TRAIN-only basis")
    residual_weight = args.residual_weight if residual_branch else 0.
    if natural and (natural_search is None or preserve_data is None):
        raise ValueError("Natural branch requires independent natural validation and generated preservation data")
    perceptual_weight = getattr(args, "natural_perceptual_weight", 0.) if natural else 0.
    if perceptual_weight and perceptual is None:
        raise ValueError("Natural perceptual objective requires a frozen LPIPS network")
    policy = getattr(args, "quality_policy", "report")
    strength = 0. if method == "reconstruction" or natural else args.strength
    folder.mkdir(parents=True, exist_ok=False)
    diagnostics_root = folder if diagnostics_root is None else diagnostics_root
    thresholds = {"min_psnr": args.min_psnr, "min_ssim": args.min_ssim,
                  "min_image_psnr": args.min_image_psnr}

    def evaluate(step, source):
        nonlocal best, best_state
        try:
            materialize(vae.decoder, names, grids)
            metrics = score(vae, search_z, search_ref, strength, args)
            if natural:
                # Gate on generated pairs, rank on disjoint natural validation reconstruction.
                metrics["generated_reference_mse"] = metrics["target_mse"]
                metrics.update(natural_reconstruction_metrics(vae, *natural_search, perceptual, perceptual_weight,
                    batch_size(getattr(args, "eval_batch_size", 0), device), residual if residual_branch else None, residual_weight))
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

    evaluate(0, "fixed_rtn" if method == "fixed_ptq" else "rtn_fallback")
    order_rng = torch.Generator().manual_seed(args.seed)
    params = [p for p in grids.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr) if method in ("rounding", "rounding_scale", "reconstruction", "natural_rounding", "natural_rounding_scale", "natural_residual") else None
    ranked = [names.index(r["name"]) for r in profile]
    valid_updates = 0
    backward_attempts = 0
    train_evaluations = 0
    if method == "sensitivity":
        materialize(vae.decoder, names, grids)
        training_best = score(vae, train_z, train_ref, strength, args)
        train_evaluations += 1
        vae.decoder.load_state_dict(pristine)
    total = 0 if method == "fixed_ptq" else args.steps
    for step in range(1, total + 1):
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
            if (step - 1) % len(train_z) == 0:
                order = torch.randperm(len(train_z), generator=order_rng).tolist()
            i = order[(step - 1) % len(train_z)]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                hidden = vae.post_quant_conv(train_z[i].to(device))
            weights = {k: q() for k, q in zip(names, grids)}
            raw = functional_call(vae.decoder, weights, (hidden,)) / 2 + .5
            ref = train_ref[i].to(device)
            image01(ref, "training reference")
            target_loss = F.mse_loss(raw, pseudo_target(ref, strength))
            preserve = F.mse_loss(blur(raw), blur(ref))
            perceptual_loss = raw.new_zeros(())
            residual_loss = raw.new_zeros(())
            if natural:
                if residual_branch:
                    residual_loss = residual.loss(raw - ref).mean()
                if perceptual is not None:
                    perceptual_loss = perceptual(raw.clamp(0, 1) * 2 - 1, ref * 2 - 1).mean()
                j = (step - 1) % len(preserve_data[0])
                with torch.no_grad():
                    preserve_hidden = vae.post_quant_conv(preserve_data[0][j].to(device))
                generated = functional_call(vae.decoder, weights, (preserve_hidden,)) / 2 + .5
                preserve_ref = image01(preserve_data[1][j].to(device), "preservation reference")
                preserve = (residual.preservation(generated - preserve_ref, args.residual_preservation)
                            if residual_branch else F.mse_loss(generated, preserve_ref))
                train_evaluations += 1
            loss = target_loss + perceptual_weight * perceptual_loss + args.preserve_weight * preserve + residual_weight * residual_loss
            train_evaluations += 1
            applied = bool(torch.isfinite(loss))
            grad_norm = None
            if applied:
                backward_attempts += 1
                loss.backward()
                applied = all(p.grad is not None for p in params) and all_finite(p.grad for p in params)
                if applied:
                    norm = nn.utils.clip_grad_norm_(params, 1.)
                    applied = bool(torch.isfinite(norm))
                    if applied:
                        grad_norm = norm.item()
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
                for group in optimizer.param_groups:
                    group["lr"] *= .5
            with torch.no_grad():
                for grid in grids:
                    grid.alpha.clamp_(-12, 12)
                    grid.log_scale.clamp_(math.log(.8), math.log(1.25))
            scalars = torch.stack([target_loss.detach(), perceptual_loss.detach(), preserve.detach(), loss.detach(), residual_loss.detach()]).cpu().tolist()
            updates.append({"step": step, "applied": applied, "reason": reason,
                            **{k: v if math.isfinite(v) else None for k, v in zip(
                                ["reconstruction_mse", "perceptual_loss", "preservation_mse", "loss", "residual_loss"], scalars)},
                            "grad_norm": grad_norm})
            del weights, raw, ref, loss, target_loss, preserve, perceptual_loss, residual_loss
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
            evaluate(step, method if valid_updates else "rtn_fallback")
    grids.load_state_dict(best_state)
    selected = {**best, "search_feasible": best["feasible"], "valid_updates": valid_updates,
                "gradient_updates": valid_updates if optimizer is not None else 0,
                "objective": "natural_residual_projection" if residual_branch else ("unpaired_natural_reconstruction" if natural else ("reconstruction" if method == "reconstruction" else "blind_smoothing_proxy")),
                "residual_weight": residual_weight,
                "preservation_mode": args.residual_preservation if residual_branch else "legacy",
                "objective_strength": strength,
                "quality_policy": policy, "perceptual_weight": perceptual_weight,
                "selected_source": best["source"], "attempted_updates": total,
                "train_evaluations": train_evaluations, "search_evaluations": len(rows),
                "train_image_forwards": train_evaluations * len(train_z) if method == "sensitivity" else train_evaluations,
                "search_image_forwards": len(rows) * len(search_z), "backward_attempts": backward_attempts,
                "natural_validation_image_forwards": len(rows) * len(natural_search[0]) if natural else 0,
                "optimized_dofs": {"fixed_ptq": [], "sensitivity": ["layer_scale"], "rounding": ["rounding"],
                    "rounding_scale": ["rounding", "per_channel_scale"], "reconstruction": ["rounding"],
                    "natural_rounding": ["rounding"], "natural_residual": ["rounding"], "natural_rounding_scale": ["rounding", "per_channel_scale"]}[method],
                "status": "selected" if best["feasible"] else ("selected_quality_failed" if policy == "report" else "no_feasible_candidate")}
    save_json(folder / "selection.json", selected)
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Diffusers pipeline already containing the marked VAE")
    p.add_argument("--prompts", required=True, help="UTF-8 text file; one unique prompt per line")
    p.add_argument("--output", required=True, help="New directory, never overwrite an experiment")
    p.add_argument("--image-output", help="Separate NEW image directory; default output_artifacts/images/<run>")
    p.add_argument("--artifact-output", help="Separate NEW checkpoint directory; default output_artifacts/checkpoints/<run>")
    p.add_argument("--gen-batch-size", type=int, default=0, help="Inference batch; 0 chooses from free VRAM, capped at 8")
    p.add_argument("--eval-batch-size", type=int, default=0, help="VAE/metric batch; 0 chooses from free VRAM, capped at 8")
    p.add_argument("--data-cache", choices=["auto", "cpu"], default="auto")
    p.add_argument("--cache-max-gib", type=float, default=4., help="Maximum total GPU data cache; preserve at least half total VRAM free when promoting")
    p.add_argument("--log-every", type=int, default=10, help="Flush accumulated update rows every N steps, on failure, and at branch completion")
    p.add_argument("--train-n", type=int, default=32)
    p.add_argument("--search-n", type=int, default=20)
    p.add_argument("--test-n", type=int, default=100)
    p.add_argument("--steps", type=int, default=100, help="Update/proposal budget per optimized branch and grid cell")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--sampling-steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--bits", type=int, nargs="+", default=[4])
    p.add_argument("--clips", type=float, nargs="+", default=[1.])
    p.add_argument("--methods", nargs="+", choices=["fixed_ptq", "sensitivity",
                   "rounding", "rounding_scale", "reconstruction"],
                   default=["fixed_ptq", "rounding_scale", "reconstruction"])
    p.add_argument("--profile-n", type=int, default=16, help="Train-only samples for measured layer sensitivity")
    p.add_argument("--profile-bootstrap", type=int, default=1000, help="CPU paired resamples for layer priority uncertainty")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU is for small diagnostic pipelines")
    p.add_argument("--scope", choices=["all", "late"], default="all")
    p.add_argument("--strength", type=float, default=.25, help="Fixed blind smoothing hypothesis; 0 = reconstruction control")
    p.add_argument("--min-psnr", type=float, default=25.)
    p.add_argument("--min-image-psnr", type=float, default=22.)
    p.add_argument("--min-ssim", type=float, default=.9)
    p.add_argument("--preserve-weight", type=float, default=2.)
    p.add_argument("--natural-images", help="Optional directory of unpaired natural non-watermarked images")
    p.add_argument("--natural-methods", nargs="+", choices=["natural_rounding", "natural_rounding_scale", "natural_residual"],
                   default=["natural_rounding", "natural_residual"], help="Branches added when natural images are supplied")
    p.add_argument("--residual-patch", type=int, choices=[4, 8, 16], default=8)
    p.add_argument("--residual-rank", type=int, default=8)
    p.add_argument("--residual-patches-per-image", type=int, default=256)
    p.add_argument("--residual-weight", type=float, default=1., help="Extra projected natural reconstruction loss; 0 for ablation")
    p.add_argument("--residual-preservation", choices=["full", "orthogonal"], default="orthogonal",
                   help="Residual branch only: protect complement plus 0.1 full RGB MSE, or full RGB control")
    p.add_argument("--natural-perceptual-weight", type=float, default=.1,
                   help="LPIPS coefficient ONLY for natural branches; 0 disables the external perceptual prior")
    p.add_argument("--quality-policy", choices=["report", "constrained"], default="report",
                   help="report: choose by objective and record quality failures; constrained: prefer feasible candidates; neither stops on a failed gate")
    return p


def main():
    args = parser().parse_args()
    EVENTS.clear()
    if min(args.gen_batch_size, args.eval_batch_size, args.cache_max_gib) < 0 or args.log_every < 1 or not math.isfinite(args.cache_max_gib):
        raise ValueError("Invalid runtime batch/cache/log configuration")
    if min(args.train_n, args.search_n, args.test_n, args.steps, args.eval_every, args.sampling_steps, args.profile_n, args.profile_bootstrap) < 1:
        raise ValueError("Sample counts and steps must be positive")
    if (not 0 <= args.strength <= 1 or not 0 < args.min_ssim <= 1
            or args.lr <= 0 or args.preserve_weight < 0 or args.natural_perceptual_weight < 0 or args.guidance <= 1
            or not all(math.isfinite(v) for v in [args.strength, args.min_ssim, args.lr,
                args.preserve_weight, args.natural_perceptual_weight, args.guidance, args.min_psnr, args.min_image_psnr, *args.clips])):
        raise ValueError("Invalid numeric arguments")
    if any(b not in (2, 3, 4, 6, 8) for b in args.bits) or any(not 0 < c <= 1 for c in args.clips):
        raise ValueError("Invalid grid")
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
    natural_manifest = natural_image_manifest(args.natural_images, args.train_n, args.search_n, args.seed) if args.natural_images else None
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
    # Accuracy audit profile, not a promise of bitwise reproducibility across GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    args.gen_batch_size = batch_size(args.gen_batch_size, args.device)
    args.eval_batch_size = batch_size(args.eval_batch_size, args.device)
    data_cache = DataCache(args.device, args.data_cache, args.cache_max_gib)
    print(f"Inference batch={args.gen_batch_size}; evaluation batch={args.eval_batch_size}; data cache={args.data_cache}", flush=True)
    vae = pipe.vae.eval().requires_grad_(False)
    names = select_names(vae, args.scope)
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
                    ("wmq_runtime.py", "wmq_diagnostics.py", "wmq_residual.py")},
                "model_files_sha256": {str(p.relative_to(model)): sha(p) for p in sorted(model.rglob("*"))
                      if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")},
                "prompts": prompts, "seeds": list(range(args.seed, args.seed + n)),
                "target_names": names, "changed_parameters": sum(vae.decoder.get_parameter(k).numel() for k in names),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "reproducibility": {"python": platform.python_version(), "platform": platform.platform(),
                    "cudnn": torch.backends.cudnn.version(), "tf32_matmul": False, "tf32_cudnn": False,
                    "cudnn_benchmark": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "warning": "Seeds and TF32 settings do not guarantee bitwise reproducibility across GPUs or library versions."},
                "quality_metric_definition": {"ssim": "Gaussian 11x11 sigma=1.5; population covariance; valid crop; RGB channel mean; data_range=1",
                    "ssim_statistics_dtype": "float64", "psnr": "per-image RGB MSE; data_range=1; cap 120dB"},
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "inference": f"FP32 VAE; {'FP16' if args.device == 'cuda' else 'FP32'} UNet; simulated low-bit weights",
                "sampling_scheduler": dict(pipe.scheduler.config),
                "watermark_success": "unknown; requires independent owner evaluation",
                "quality_gate": "PSNR/SSIM only; not equivalent to legacy LPIPS gate",
                "quality_policy": args.quality_policy,
                "natural_perceptual_prior": "frozen LPIPS AlexNet v0.1; public pretrained features; natural branches only" if natural_manifest and args.natural_perceptual_weight else None}
    plan = []
    for bits in dict.fromkeys(args.bits):
        for clip in dict.fromkeys(args.clips):
            for method in dict.fromkeys([*args.methods, *(args.natural_methods if natural_manifest else [])]):
                branch_id = f"{method}_w{bits}_c{clip}"
                plan.append({"label": branch_id + "_test", "method": method,
                             "comparison_group": f"w{bits}_c{clip}_{args.scope}",
                             "bits": bits, "clip": clip, "role": "candidate",
                             "threat_model": "marked_model_plus_unpaired_natural_images" if method.startswith("natural_") else "marked_model_only",
                             "artifact": f"branches/{branch_id}"})
    manifest["reference_label"] = "marked_reference_test"
    manifest["image_root"] = image_location
    manifest["artifact_root"] = artifact_location
    manifest["test_branches"] = [{"label": "marked_reference_test", "role": "reference",
                                  "method": "marked_reference"},
                                 {"label": "pseudo_target_test", "role": "diagnostic", "method": "direct_smoothing",
                                  "strength": args.strength}, *plan]
    manifest["natural_dataset"] = natural_manifest
    if natural_manifest:
        manifest["threat_model"] = "separate_model_only_and_model_plus_unpaired_natural_branches"
    manifest["sensitivity_kind"] = "visual_proxy_only_not_ownership_sensitivity"
    manifest["proxy_limitations"] = ["Smoothing may retain fingerprint", "Proxy improvement may erase natural texture",
                                      "Owner metrics are evaluation-only after every branch is frozen"]
    manifest["comparison_note"] = "Compare only within comparison_group; optimization costs reported, not equalized."
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
    # Complete every model-only branch before loading external training images/network.
    plan.sort(key=lambda branch: branch["method"].startswith("natural_"))
    rows, selections, frozen_branches = [], {}, {}
    profiles = {}
    costs = {}
    for branch in plan:
        if args.device == "cuda":
            torch.cuda.synchronize()
        branch_started = time.monotonic()
        bits, clip, method = branch["bits"], branch["clip"], branch["method"]
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
            natural_train = cache_natural(vae, natural_manifest["train"], args.eval_batch_size)
            natural_search = cache_natural(vae, natural_manifest["search"], args.eval_batch_size)
            natural_train = tuple(data_cache.promote(v, f"natural_train_{i}") for i, v in enumerate(natural_train))
            natural_search = tuple(data_cache.promote(v, f"natural_search_{i}") for i, v in enumerate(natural_search))
            if args.natural_perceptual_weight:
                import lpips
                perceptual = lpips.LPIPS(net="alex", version="0.1").to(args.device).eval().requires_grad_(False)
        train_z, train_ref = natural_train if natural else (zs[:a], refs[:a])
        if method == "natural_residual" and residual is None:
            from wmq_residual import fit_residual_subspace
            residual, residual_info = fit_residual_subspace(vae, *natural_train, args.residual_patch,
                args.residual_rank, args.residual_patches_per_image, args.seed)
        grids, selected, branch_rows = optimize_branch(
            vae, names, train_z, train_ref, search_z, search_ref, args,
            bits, clip, method, profiles.get(group, []), folder, diagnostics_root=out,
            natural_search=natural_search if natural else None,
            preserve_data=(zs[:a], refs[:a]) if natural else None,
            perceptual=perceptual if natural else None, residual=residual if method == "natural_residual" else None)
        rows.extend(branch_rows)
        selections[branch["label"]] = selected
        materialize(vae.decoder, names, grids)
        # No bias, norm, or unselected decoder weight may change.
        for name, value in vae.decoder.state_dict().items():
            if name not in names and not torch.equal(value.cpu(), pristine[name]):
                raise ValueError(f"Unexpected parameter change: {name}")
        artifact_folder.mkdir(parents=True, exist_ok=False)
        vae.save_pretrained(artifact_folder / "vae", safe_serialization=True)
        export_quantizer(vae, names, grids, artifact_folder)
        if method == "natural_residual":
            from safetensors.torch import save_file
            save_file({"basis": residual.basis.detach().cpu().contiguous()}, str(artifact_folder / "residual_basis.safetensors"))
            save_json(folder / "residual_calibration.json", residual_info)
        frozen_branches[branch["label"]] = {
            "selected": selected,
            "files_sha256": {p.relative_to(artifact_out).as_posix(): sha(p)
                             for p in sorted(artifact_folder.rglob("*")) if p.is_file()},
            "diagnostic_files_sha256": {p.relative_to(out).as_posix(): sha(p)
                                        for p in sorted(folder.rglob("*")) if p.is_file()}}
        del grids
        vae.decoder.load_state_dict(pristine)
        save_json(out / "search.json", rows)
        save_csv(out / "search.csv", rows)
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        costs[branch["label"]] = {"search_and_export_seconds_including_profile": time.monotonic() - branch_started,
                                 "profile_image_forwards": len(names) * min(a, args.profile_n) if method == "sensitivity" else 0}

    # Freeze ALL branches and the declared protocol before the first test latent.
    frozen = {"schema_version": 2, "phase": "before_test_generation",
              "branches": frozen_branches, "manifest_sha256": sha(out / "manifest.json"),
              "search_sha256": sha(out / "search.json"),
              "profile_sha256": {p.name: sha(p) for p in sorted(out.glob("profile_*.json"))}}
    save_json(out / "selection_frozen.json", frozen)
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
              "branch_compute": costs,
              "runtime": {"gen_batch_size": args.gen_batch_size, "eval_batch_size": args.eval_batch_size,
                          "training_batch_size": 1, "cache": data_cache.stats, "events": list(EVENTS), "log_every": args.log_every},
              "branch_quality": quality, "status": "complete" if complete else "quality_failures",
              "watermark_metrics": None, "watermark_success": "unknown",
              "test_used_for_selection": False, "selection_frozen_sha256": sha(out / "selection_frozen.json"),
              "image_root": image_location,
              "artifact_root": artifact_location,
              "test_images_sha256": {p.relative_to(image_out).as_posix(): sha(p)
                   for branch in manifest["test_branches"] for p in sorted((image_out / branch["label"]).glob("*.png"))},
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
              "note": "Failed controls are retained and flagged; no watermark removal or OS-MQ claim."})


if __name__ == "__main__":
    main()

"""Model-only VAE quantization experiment; no watermark detector or secret inputs.

The suppression target is a hypothesis, not a watermark oracle. All models and
calibration images used by the attacker come from the supplied marked pipeline.
"""
import argparse
import hashlib
import json
import math
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


def blur(x):
    """Fixed Gaussian [1,4,6,4,1]/16, reflection padding, no learned prior."""
    k = x.new_tensor([1, 4, 6, 4, 1]) / 16
    kernel = (k[:, None] * k[None, :]).expand(x.shape[1], 1, 5, 5)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel,
                    groups=x.shape[1])


def pseudo_target(reference, strength):
    # Both inputs and targets retain unknown watermark content. Never call clean.
    return reference.lerp(blur(reference), strength)


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

    @torch.no_grad()
    def randomize(self, generator):
        value = (self.source / self.scale).clamp(self.qmin, self.qmax)
        fraction = value - value.floor()
        draws = torch.rand(fraction.shape, device=fraction.device, generator=generator)
        self.alpha.copy_(torch.where(draws < fraction, 8., -8.))


def pair_metrics(x, y):
    """PSNR and Wang-style SSIM: 11x11 Gaussian sigma=1.5, valid crop, population covariance."""
    if x.shape != y.shape or x.ndim != 4 or min(x.shape[-2:]) < 11:
        raise ValueError("SSIM requires paired NCHW images at least 11x11")
    x, y = x.float(), y.float()
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
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    if not latents or len(latents) != len(refs):
        raise ValueError("Need nonempty, paired latents/references")
    device = next(vae.parameters()).device
    for i, (z, ref) in enumerate(zip(latents, refs)):
        x = vae.decode(z.to(device, torch.float32), return_dict=False)[0]
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite decoder output")
        x = (x / 2 + .5).clamp(0, 1)
        ref = ref.to(x.device)
        p, s = pair_metrics(ref, x)
        psnr.append(p.item())
        ssim.append(s.item())
        errors.append(F.mse_loss(x, pseudo_target(ref, strength)).item())
        if folder is not None:
            array = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            Image.fromarray(array).save(folder / f"{i:04d}.png")
    p, s = torch.tensor(psnr), torch.tensor(ssim)
    feasible = bool(p.mean() >= args.min_psnr and s.mean() >= args.min_ssim
                    and p.min() >= args.min_image_psnr)
    return {"psnr": p.mean().item(), "ssim": s.mean().item(),
            "psnr_p05": torch.quantile(p, .05).item(), "min_psnr": p.min().item(),
            "quality_violation_fraction": ((p < args.min_psnr) | (s < args.min_ssim)).float().mean().item(),
            "target_mse": sum(errors) / len(errors), "feasible": feasible,
            "per_image_psnr": psnr, "per_image_ssim": ssim}


def choose(rows):
    eligible = [r for r in rows if r["feasible"]]
    return min(eligible, key=lambda r: (r["target_mse"], -r["psnr"], r["candidate"])) if eligible else None


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
            for z, ref in zip(latents, refs):
                ref = ref.to(device)
                image = (vae.decode(z.to(device), return_dict=False)[0] / 2 + .5).clamp(0, 1)
                target = pseudo_target(ref, args.strength)
                errors.append(F.mse_loss(image, ref).item())
                gains.append((F.mse_loss(ref, target) - F.mse_loss(image, target)).item())
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
                     "proxy_gain": sum(gains) / len(latents),
                     "priority": sum(gains) / max(sum(errors), 1e-12),
                     "priority_ci95_low": None if low is None else float(low),
                     "priority_ci95_high": None if high is None else float(high),
                     "per_image_output_mse": errors, "per_image_proxy_gain": gains})
    return sorted(rows, key=lambda r: (-r["priority"], r["name"]))


def optimize_branch(vae, names, train_z, train_ref, search_z, search_ref,
                    args, bits, clip, method, profile, folder, diagnostics_root=None):
    """Independent initialization, train-only updates, search-only checkpoint choice."""
    device = next(vae.parameters()).device
    pristine = {k: v.detach().clone() for k, v in vae.decoder.state_dict().items()}
    grids = nn.ModuleList([RoundingGrid(vae.decoder.get_parameter(k), bits, clip,
                           learn_scale=method == "rounding_scale") for k in names]).to(device)
    rows, updates, best = [], [], None
    best_state = None
    strength = 0. if method == "reconstruction" else args.strength
    folder.mkdir(parents=True, exist_ok=False)
    diagnostics_root = folder if diagnostics_root is None else diagnostics_root
    thresholds = {"min_psnr": args.min_psnr, "min_ssim": args.min_ssim,
                  "min_image_psnr": args.min_image_psnr}

    def evaluate(step, source):
        nonlocal best, best_state
        try:
            materialize(vae.decoder, names, grids)
            metrics = score(vae, search_z, search_ref, strength, args)
        finally:
            vae.decoder.load_state_dict(pristine)
        row = {"candidate": f"{method}_w{bits}_c{clip}_step{step}", "source": source,
               "method": method, "bits": bits, "clip": clip, "step": step, **metrics}
        rows.append(row)
        if not row["feasible"]:
            quality_warning(diagnostics_root, "search_candidate", "Candidate failed quality gate",
                            method=method, metrics=row, thresholds=thresholds)
        winner = choose(rows)
        # Retain a diagnosed failed control if no candidate is feasible; never call it successful.
        if winner is None:
            winner = min(rows, key=lambda r: (-r["psnr"], r["candidate"]))
        if best is None or winner["candidate"] != best["candidate"]:
            if winner is not row:
                raise RuntimeError("Selection snapshot lost")
            best, best_state = dict(winner), {k: v.detach().cpu().clone() for k, v in grids.state_dict().items()}
        save_json(folder / "search.json", rows)
        print({k: v for k, v in row.items() if not k.startswith("per_image")}, flush=True)

    evaluate(0, "fixed_rtn" if method == "fixed_ptq" else "rtn_fallback")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    order_rng = torch.Generator().manual_seed(args.seed)
    params = [p for p in grids.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr) if method in ("rounding", "rounding_scale", "reconstruction") else None
    ranked = [names.index(r["name"]) for r in profile]
    valid_updates = 0
    backward_attempts = 0
    train_evaluations = 0
    if method == "sensitivity":
        materialize(vae.decoder, names, grids)
        training_best = score(vae, train_z, train_ref, strength, args)
        train_evaluations += 1
        vae.decoder.load_state_dict(pristine)
    total = 0 if method == "fixed_ptq" else (math.ceil(args.steps / args.eval_every)
                                             if method == "random_rounding" else args.steps)
    for step in range(1, total + 1):
        applied, reason = True, None
        if method == "random_rounding":
            for grid in grids:
                grid.randomize(rng)
            evaluate(step, method)
        elif method == "sensitivity":
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
            rank = lambda m: (not m["feasible"], m["target_mse"] if m["feasible"] else -m["psnr"])
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
            target_loss = F.mse_loss(raw, pseudo_target(ref, strength))
            preserve = F.mse_loss(blur(raw), blur(ref))
            loss = target_loss + args.preserve_weight * preserve
            train_evaluations += 1
            applied = bool(torch.isfinite(loss))
            grad_norm = None
            if applied:
                backward_attempts += 1
                loss.backward()
                applied = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
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
                applied = all(bool(torch.isfinite(p).all()) for p in params)
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
            updates.append({"step": step, "applied": applied, "reason": reason,
                            "loss": loss.item() if torch.isfinite(loss) else None, "grad_norm": grad_norm})
            del weights, raw, ref, loss, target_loss, preserve
        if method in ("random_rounding", "sensitivity"):
            updates.append({"step": step, "applied": applied, "reason": reason})
        if not applied and method not in ("random_rounding", "sensitivity"):
            quality_warning(diagnostics_root, "optimizer_update", "Invalid update skipped or rolled back",
                            method=method, metrics=updates[-1], action="skip_update_and_continue")
        valid_updates += int(applied)
        save_json(folder / "updates.json", {"attempted": step, "valid_updates": valid_updates,
                  "skipped_or_rejected": step - valid_updates, "rows": updates})
        if method != "random_rounding" and (step % args.eval_every == 0 or step == total):
            evaluate(step, method if valid_updates else "rtn_fallback")
    grids.load_state_dict(best_state)
    selected = {**best, "search_feasible": best["feasible"], "valid_updates": valid_updates,
                "gradient_updates": valid_updates if optimizer is not None else 0,
                "objective": "reconstruction" if method == "reconstruction" else "blind_smoothing_proxy",
                "objective_strength": strength,
                "selected_source": best["source"], "attempted_updates": total,
                "train_evaluations": train_evaluations, "search_evaluations": len(rows),
                "train_image_forwards": train_evaluations * len(train_z) if method == "sensitivity" else train_evaluations,
                "search_image_forwards": len(rows) * len(search_z), "backward_attempts": backward_attempts,
                "optimized_dofs": {"fixed_ptq": [], "random_rounding": ["stochastic_rounding"],
                    "sensitivity": ["layer_scale"], "rounding": ["rounding"],
                    "rounding_scale": ["rounding", "per_channel_scale"], "reconstruction": ["rounding"]}[method],
                "status": "selected" if best["feasible"] else "no_feasible_candidate"}
    save_json(folder / "selection.json", selected)
    if not selected["search_feasible"]:
        quality_warning(diagnostics_root, "branch_fallback", "No feasible candidate; retaining diagnostic result",
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
    p.add_argument("--methods", nargs="+", choices=["fixed_ptq", "random_rounding", "sensitivity",
                   "rounding", "rounding_scale", "reconstruction"],
                   default=["fixed_ptq", "random_rounding", "sensitivity", "rounding", "rounding_scale"])
    p.add_argument("--profile-n", type=int, default=16, help="Train-only samples for measured layer sensitivity")
    p.add_argument("--profile-bootstrap", type=int, default=1000, help="CPU paired resamples for layer priority uncertainty")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU is for small diagnostic pipelines")
    p.add_argument("--scope", choices=["all", "late"], default="all")
    p.add_argument("--strength", type=float, default=.25, help="Fixed blind smoothing hypothesis; 0 = reconstruction control")
    p.add_argument("--min-psnr", type=float, default=25.)
    p.add_argument("--min-image-psnr", type=float, default=22.)
    p.add_argument("--min-ssim", type=float, default=.9)
    p.add_argument("--preserve-weight", type=float, default=2.)
    return p


def main():
    args = parser().parse_args()
    if min(args.train_n, args.search_n, args.test_n, args.steps, args.eval_every, args.sampling_steps, args.profile_n, args.profile_bootstrap) < 1:
        raise ValueError("Sample counts and steps must be positive")
    if (not 0 <= args.strength <= 1 or not 0 < args.min_ssim <= 1
            or args.lr <= 0 or args.preserve_weight < 0 or args.guidance <= 1
            or not all(math.isfinite(v) for v in [args.strength, args.min_ssim, args.lr,
                args.preserve_weight, args.guidance, args.min_psnr, args.min_image_psnr, *args.clips])):
        raise ValueError("Invalid numeric arguments")
    if any(b not in (2, 3, 4, 6, 8) for b in args.bits) or any(not 0 < c <= 1 for c in args.clips):
        raise ValueError("Invalid grid")
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
    out = Path(args.output).resolve()
    if out == model or model in out.parents:
        raise ValueError("Output must be outside the input model directory")
    out.mkdir(parents=True, exist_ok=False)
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
                "method": "pilot_model_only_ladder_not_OS_MQ", "args": vars(args),
                "ownership_loss": None, "surrogate_transfer_evaluated": False,
                "git_commit": commit, "script_sha256": sha(script),
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
                "quality_gate": "PSNR/SSIM only; not equivalent to legacy LPIPS gate"}
    plan = []
    for bits in dict.fromkeys(args.bits):
        for clip in dict.fromkeys(args.clips):
            for method in dict.fromkeys(args.methods):
                branch_id = f"{method}_w{bits}_c{clip}"
                plan.append({"label": branch_id + "_test", "method": method,
                             "comparison_group": f"w{bits}_c{clip}_{args.scope}",
                             "bits": bits, "clip": clip, "role": "candidate",
                             "artifact": f"branches/{branch_id}"})
    manifest["reference_label"] = "marked_reference_test"
    manifest["test_branches"] = [{"label": "marked_reference_test", "role": "reference",
                                  "method": "marked_reference"}, *plan]
    manifest["comparison_note"] = "Compare only within comparison_group; optimization costs reported, not equalized."
    save_json(out / "manifest.json", manifest)
    @torch.no_grad()
    def cache(indices):
        # Test data are not generated until the chosen checkpoint is exported.
        pipe.unet.to(args.device)
        pipe.text_encoder.to(args.device)
        latents = []
        for i in indices:
            gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
            z = pipe(prompts[i], generator=gen, num_inference_steps=args.sampling_steps,
                     guidance_scale=args.guidance, height=512, width=512, output_type="latent").images
            latents.append((z.float() / float(vae.config.scaling_factor)).cpu())
            print(f"latent {i + 1}/{n}", flush=True)
        pipe.unet.to("cpu")
        pipe.text_encoder.to("cpu")
        if args.device == "cuda":
            torch.cuda.empty_cache()
        references = [(vae.decode(z.to(args.device), return_dict=False)[0] / 2 + .5).clamp(0, 1).cpu()
                      for z in latents]
        return latents, references

    a, b = args.train_n, args.train_n + args.search_n
    zs, refs = cache(range(b))
    search_z, search_ref = zs[a:b], refs[a:b]
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
        grids, selected, branch_rows = optimize_branch(
            vae, names, zs[:a], refs[:a], search_z, search_ref, args,
            bits, clip, method, profiles.get(group, []), folder, diagnostics_root=out)
        rows.extend(branch_rows)
        selections[branch["label"]] = selected
        materialize(vae.decoder, names, grids)
        # No bias, norm, or unselected decoder weight may change.
        for name, value in vae.decoder.state_dict().items():
            if name not in names and not torch.equal(value.cpu(), pristine[name]):
                raise ValueError(f"Unexpected parameter change: {name}")
        vae.save_pretrained(folder / "vae", safe_serialization=True)
        export_quantizer(vae, names, grids, folder)
        frozen_branches[branch["label"]] = {
            "selected": selected,
            "files_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}}
        del grids
        vae.decoder.load_state_dict(pristine)
        save_json(out / "search.json", rows)
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
    reference_quality = score(vae, test_z, test_ref, args.strength, args, out / "marked_reference_test")
    quality = {"marked_reference_test": reference_quality}
    from safetensors.torch import load_file
    for branch in plan:
        vae.decoder.load_state_dict(pristine)
        tensors = load_file(str(out / branch["artifact"] / "quantizer.safetensors"), device=args.device)
        with torch.no_grad():
            for name in names:
                vae.decoder.get_parameter(name).copy_(tensors[name + ".codes"].float() * tensors[name + ".scale"])
        quality[branch["label"]] = score(vae, test_z, test_ref, args.strength, args, out / branch["label"])
        if not quality[branch["label"]]["feasible"]:
            quality_warning(out, "test_quality", "Branch failed held-out quality gate; results retained",
                            method=branch["label"], metrics=quality[branch["label"]],
                            thresholds={"min_psnr": args.min_psnr, "min_ssim": args.min_ssim,
                                        "min_image_psnr": args.min_image_psnr})
        del tensors
    complete = all(s["search_feasible"] and quality[label]["feasible"] for label, s in selections.items())
    save_json(out / "report.json", {"schema_version": 2, "selections": selections,
              "branch_compute": costs,
              "branch_quality": quality, "status": "complete" if complete else "quality_failures",
              "watermark_metrics": None, "watermark_success": "unknown",
              "test_used_for_selection": False, "selection_frozen_sha256": sha(out / "selection_frozen.json"),
              "test_images_sha256": {str(p.relative_to(out)): sha(p)
                   for branch in manifest["test_branches"] for p in sorted((out / branch["label"]).glob("*.png"))},
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if args.device == "cuda" else None,
              "note": "Failed controls are retained and flagged; no watermark removal or OS-MQ claim."})


if __name__ == "__main__":
    main()

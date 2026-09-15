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

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


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
    """One trainable rounding decision per weight; frozen per-channel scales.

    Forward uses hard integers with a sigmoid straight-through gradient. Export
    uses exactly the same grid; unconstrained full-precision weights never train.
    """
    def __init__(self, weight, bits, clip=1.0):
        super().__init__()
        if bits not in (2, 3, 4, 6, 8) or not 0 < clip <= 1:
            raise ValueError("Invalid bitwidth/clip")
        w = weight.detach().float()
        if not torch.isfinite(w).all():
            raise ValueError("Nonfinite source weights")
        self.qmax = 2 ** (bits - 1) - 1
        scale = w.flatten(1).abs().amax(1).clamp_min(1e-8) * clip / self.qmax
        scale = scale.reshape([-1] + [1] * (w.ndim - 1))
        value = (w / scale).clamp(-self.qmax, self.qmax)
        self.register_buffer("scale", scale)
        self.register_buffer("floor", value.floor())
        frac = value - value.floor()
        self.alpha = nn.Parameter(torch.logit(frac.clamp(1e-4, 1 - 1e-4)))

    def forward(self, differentiable=True):
        soft = self.alpha.sigmoid()
        hard = (self.alpha >= 0).to(soft.dtype)
        rounding = hard + (soft - soft.detach()) if differentiable else hard
        return (self.floor + rounding).clamp(-self.qmax, self.qmax) * self.scale


def pair_metrics(x, y):
    """Per-image PSNR and local SSIM, float32 [0,1]; no pretrained metric."""
    x, y = x.float(), y.float()
    mse = (x - y).square().flatten(1).mean(1)
    psnr = -10 * mse.clamp_min(1e-12).log10()
    ux, uy = blur(x), blur(y)
    vx = (blur(x.square()) - ux.square()).clamp_min(0)
    vy = (blur(y.square()) - uy.square()).clamp_min(0)
    cov = blur(x * y) - ux * uy
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
    for i, (z, ref) in enumerate(zip(latents, refs)):
        x = vae.decode(z.to("cuda", torch.float32), return_dict=False)[0]
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local Diffusers pipeline already containing the marked VAE")
    p.add_argument("--prompts", required=True, help="UTF-8 text file; one unique prompt per line")
    p.add_argument("--output", required=True, help="New directory, never overwrite an experiment")
    p.add_argument("--train-n", type=int, default=32)
    p.add_argument("--search-n", type=int, default=20)
    p.add_argument("--test-n", type=int, default=100)
    p.add_argument("--steps", type=int, default=100, help="Gradient steps per candidate")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--sampling-steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    p.add_argument("--clips", type=float, nargs="+", default=[1., .9])
    p.add_argument("--scope", choices=["all", "late"], default="all")
    p.add_argument("--strength", type=float, default=.25, help="Fixed blind smoothing hypothesis; 0 = reconstruction control")
    p.add_argument("--min-psnr", type=float, default=25.)
    p.add_argument("--min-image-psnr", type=float, default=22.)
    p.add_argument("--min-ssim", type=float, default=.9)
    p.add_argument("--preserve-weight", type=float, default=2.)
    return p


def main():
    args = parser().parse_args()
    if min(args.train_n, args.search_n, args.test_n, args.steps, args.eval_every, args.sampling_steps) < 1:
        raise ValueError("Sample counts and steps must be positive")
    if (not 0 <= args.strength <= 1 or not 0 < args.min_ssim <= 1
            or args.lr <= 0 or args.preserve_weight < 0 or args.guidance <= 1
            or not all(math.isfinite(v) for v in [args.strength, args.min_ssim, args.lr,
                args.preserve_weight, args.guidance, args.min_psnr, args.min_image_psnr, *args.clips])):
        raise ValueError("Invalid numeric arguments")
    if any(b not in (2, 3, 4, 6, 8) for b in args.bits) or any(not 0 < c <= 1 for c in args.clips):
        raise ValueError("Invalid grid")
    if not torch.cuda.is_available():
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
    torch.cuda.manual_seed_all(args.seed)
    # Accuracy audit profile, not a promise of bitwise reproducibility across GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(str(model), local_files_only=True,
                torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    # Reload VAE directly in FP32 (casting an already-rounded FP16 VAE is insufficient).
    from diffusers import AutoencoderKL
    pipe.vae = AutoencoderKL.from_pretrained(str(model / "vae"), local_files_only=True,
                                            torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    vae = pipe.vae.eval().requires_grad_(False)
    names = select_names(vae, args.scope)
    pristine = {k: v.detach().cpu().clone() for k, v in vae.decoder.state_dict().items()}
    script = Path(__file__).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script.parent,
                             capture_output=True, text=True, check=False).stdout.strip()
    manifest = {"status": "running", "threat_model": "marked_model_only_no_detector_no_key_no_clean_model",
                "method": "blind_self_smoothing_adaptive_rounding_experimental", "args": vars(args),
                "git_commit": commit, "script_sha256": sha(script),
                "model_files_sha256": {str(p.relative_to(model)): sha(p) for p in sorted(model.rglob("*"))
                      if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")},
                "prompts": prompts, "seeds": list(range(args.seed, args.seed + n)),
                "target_names": names, "changed_parameters": sum(vae.decoder.get_parameter(k).numel() for k in names),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "inference": "FP32 VAE; FP16 UNet; simulated low-bit weights",
                "sampling_scheduler": dict(pipe.scheduler.config),
                "watermark_success": "unknown; requires independent owner evaluation",
                "quality_gate": "PSNR/SSIM only; not equivalent to legacy LPIPS gate"}
    save_json(out / "manifest.json", manifest)
    @torch.no_grad()
    def cache(indices):
        # Test data are not generated until the chosen checkpoint is exported.
        pipe.unet.to("cuda")
        pipe.text_encoder.to("cuda")
        latents = []
        for i in indices:
            gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
            z = pipe(prompts[i], generator=gen, num_inference_steps=args.sampling_steps,
                     guidance_scale=args.guidance, height=512, width=512, output_type="latent").images
            latents.append((z.float() / float(vae.config.scaling_factor)).cpu())
            print(f"latent {i + 1}/{n}", flush=True)
        pipe.unet.to("cpu")
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        references = [(vae.decode(z.to("cuda"), return_dict=False)[0] / 2 + .5).clamp(0, 1).cpu()
                      for z in latents]
        return latents, references

    a, b = args.train_n, args.train_n + args.search_n
    zs, refs = cache(range(b))
    search_z, search_ref = zs[a:b], refs[a:b]
    rows, best_states = [], None
    best_id = None

    def evaluate(candidate, grids, label, bits, clip, step):
        nonlocal best_states, best_id
        materialize(vae.decoder, names, grids)
        m = score(vae, search_z, search_ref, args.strength, args)
        row = {"candidate": candidate, "source": label, "bits": bits, "clip": clip, "step": step, **m}
        rows.append(row)
        winner = choose(rows)
        if winner is not None and winner["candidate"] != best_id:
            # This call's candidate must be the new winner, otherwise preserve prior snapshot.
            assert winner["candidate"] == candidate
            best_id = candidate
            best_states = {k: vae.decoder.get_parameter(k).detach().cpu().clone() for k in names}
            save_json(out / "selection.json", winner)
        save_json(out / "search.json", rows)
        print({k: v for k, v in row.items() if not k.startswith("per_image")}, flush=True)
        vae.decoder.load_state_dict(pristine)

    for bits in dict.fromkeys(args.bits):
        for clip in dict.fromkeys(args.clips):
            vae.decoder.load_state_dict(pristine)
            grids = nn.ModuleList([RoundingGrid(vae.decoder.get_parameter(k), bits, clip) for k in names]).to("cuda")
            evaluate(f"w{bits}_c{clip}_rtn", grids, "rtn", bits, clip, 0)
            optimizer = torch.optim.Adam(grids.parameters(), lr=args.lr)
            rng = torch.Generator().manual_seed(args.seed)
            order = torch.randperm(a, generator=rng).tolist()
            updates = []
            for step in range(1, args.steps + 1):
                if (step - 1) % a == 0:
                    order = torch.randperm(a, generator=rng).tolist()
                i = order[(step - 1) % a]
                optimizer.zero_grad(set_to_none=True)
                weights = {k: q() for k, q in zip(names, grids)}
                # Decode via functional_call: original weights stay frozen and unmodified.
                with torch.no_grad():
                    hidden = vae.post_quant_conv(zs[i].to("cuda"))
                prediction = functional_call(vae.decoder, weights, (hidden,))
                raw = prediction / 2 + .5
                ref = refs[i].to("cuda")
                target_loss = F.mse_loss(raw, pseudo_target(ref, args.strength))
                preserve = F.mse_loss(blur(raw), blur(ref))
                # Small penalties are optimization guidance. Hard search gates decide feasibility.
                loss = target_loss + args.preserve_weight * preserve
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; no candidate is exported")
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(grids.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates.append({"step": step, "target_mse": target_loss.item(),
                                "preserve_mse": preserve.item(), "grad_norm": grad_norm.item()})
                del weights, prediction, raw, loss, target_loss, preserve
                if step % args.eval_every == 0 or step == args.steps:
                    evaluate(f"w{bits}_c{clip}_step{step}", grids, "adaptive_rounding", bits, clip, step)
                    save_json(out / f"updates_w{bits}_c{clip}.json", updates)
            del optimizer, grids
            torch.cuda.empty_cache()
    if best_states is None:
        manifest["status"] = "no_feasible_candidate"
        save_json(out / "manifest.json", manifest)
        print("No candidate passed quality constraints. No attacked VAE exported.", flush=True)
        return
    # Freeze selection before decoding test images. Detector feedback never enters this program.
    winner = choose(rows)
    vae.decoder.load_state_dict(best_states, strict=False)
    vae.save_pretrained(out / "quantized_vae", safe_serialization=True)
    export_hashes = {p.name: sha(p) for p in (out / "quantized_vae").iterdir() if p.is_file()}
    save_json(out / "selection_frozen.json", {"selected": winner, "export_sha256": export_hashes,
              "phase": "before_test_generation", "search_sha256": sha(out / "search.json")})
    vae.decoder.load_state_dict(pristine)
    test_z, test_ref = cache(range(b, n))
    score(vae, test_z, test_ref, args.strength, args, out / "marked_reference_test")
    control_grids = nn.ModuleList([RoundingGrid(vae.decoder.get_parameter(k), winner["bits"],
                                 winner["clip"]) for k in names]).to("cuda")
    materialize(vae.decoder, names, control_grids)
    rtn_test = score(vae, test_z, test_ref, args.strength, args, out / "matched_rtn_test")
    del control_grids
    vae.decoder.load_state_dict(pristine)
    vae.decoder.load_state_dict(best_states, strict=False)
    test = score(vae, test_z, test_ref, args.strength, args, out / "attacked_test")
    save_json(out / "report.json", {"selected": winner, "test_quality": test,
              "status": "complete" if test["feasible"] else "heldout_quality_failed",
              "matched_rtn_test_quality": rtn_test,
              "watermark_metrics": None, "watermark_success": "unknown",
              "test_used_for_selection": False,
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
              "export_sha256": export_hashes,
              "note": "Archive candidate for audit even if held-out quality fails; do not claim successful removal."})
    manifest["status"] = "complete" if test["feasible"] else "heldout_quality_failed"
    save_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()

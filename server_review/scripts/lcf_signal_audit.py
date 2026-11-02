"""Read-only signal-path audit of a frozen teacher; no checkpoint selection."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.code))

import torch
import torch.nn.functional as F
import lpips
from new_watermark.learning_channel.runtime import read_json, storage, load_pipeline, PrefixCache, gpu_setup
from new_watermark.learning_channel.train import load_state
from new_watermark.learning_channel.sampling import trajectory, decode
from new_watermark.learning_channel.objective import rgb8
from new_watermark.learning_channel.carrier import key_bits
from new_watermark.learning_channel.reader import key_scores

config = read_json(args.run / "config.json")
gpu_setup(dict(config, vram_limit_gb=32))
pipe = load_pipeline(args.run)
carrier, reader = load_state(storage(args.run)["artifacts"] / "teacher_selected.pt", pipe, config)
reader.eval().requires_grad_(False)
perceptual = lpips.LPIPS(net="alex").cuda().eval().requires_grad_(False)
cache = PrefixCache(args.run, pipe, max_gpu_gb=0)
prompts = read_json(args.run / "prompts.json")["teacher_train"][:4]
codes = key_bits(config["train_keys"][:2], config["bits"], device="cuda")
original = carrier.max_strength
records = []

with torch.no_grad():
    for row in prompts:
        data = cache.batch([row, row])
        def generate(multiplier):
            carrier.max_strength = original * multiplier
            z = trajectory(pipe.unet, pipe.scheduler.config, data["noise"], data["text"],
                config["sampling_steps"], carrier=carrier, codes=codes, tail_steps=config["tail_steps"],
                prefix=data["prefix"], checkpointed=False)
            return decode(pipe.vae, z)
        clean = generate(0.)
        clean8 = rgb8(clean)
        clean_small = F.interpolate(clean8, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        clean_posterior = pipe.vae.encode(clean8 * 2 - 1).latent_dist
        clean_scores = key_scores(reader(clean8), codes).diag()
        for multiplier in (1., 4., 16., 64.):
            marked = generate(multiplier)
            marked8 = rgb8(marked)
            levels = ((marked8 - clean8) * 255).round().abs()
            small = F.interpolate(marked8, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
            posterior = pipe.vae.encode(marked8 * 2 - 1).latent_dist
            mse = (marked8 - clean8).square().flatten(1).mean(1)
            float_mse = (marked - clean).square().flatten(1).mean(1)
            resized_mse = (small - clean_small).square().flatten(1).mean(1)
            scores = key_scores(reader(marked8), codes).diag() - clean_scores
            perceptual_values = perceptual(marked8 * 2 - 1, clean8 * 2 - 1).flatten()
            for i, key in enumerate(config["train_keys"][:2]):
                records.append(dict(image_id=row["image_id"], key=key, multiplier=multiplier,
                    float_rgb_mse=float(float_mse[i]), rgb8_mse=float(mse[i]),
                    psnr=float(-10 * mse[i].clamp_min(1e-12).log10()), lpips=float(perceptual_values[i]),
                    changed_fraction=float((levels[i] > 0).float().mean()),
                    changed_one_level_fraction=float((levels[i] == 1).float().sum() / (levels[i] > 0).sum().clamp_min(1)),
                    resized_mse=float(resized_mse[i]), resize_energy_ratio=float(resized_mse[i] / mse[i].clamp_min(1e-30)),
                    posterior_mean_delta_rms=float((posterior.mean[i] - clean_posterior.mean[i]).square().mean().sqrt()),
                    posterior_std_rms=float(clean_posterior.std[i].square().mean().sqrt()),
                    reader_score_gain=float(scores[i])))
        print("Audited prompt", row["image_id"], flush=True)
carrier.max_strength = original
summary = []
for multiplier in (1., 4., 16., 64.):
    subset = [r for r in records if r["multiplier"] == multiplier]
    summary.append(dict(multiplier=multiplier, **{key: sum(r[key] for r in subset) / len(subset)
        for key in records[0] if key not in ("image_id", "key", "multiplier")}))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(dict(summary=summary, rows=records, prompts="four training prompts only",
    checkpoint_modified=False, amplitude_selected=False, not_a_detection_or_transfer_evaluation=True), indent=2))
print(json.dumps(summary, indent=2), flush=True)

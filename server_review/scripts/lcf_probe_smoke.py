"""Exercise multiple learning-proxy episodes/horizons with smoke weights only."""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code", type=Path, required=True)
parser.add_argument("--smoke-run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.code))

import torch
from diffusers import DDPMScheduler
from new_watermark.learning_channel.runtime import (read_json, write_json, storage,
    setup_storage, gpu_setup, load_pipeline, PrefixCache)
from new_watermark.learning_channel.train import load_state, meta_probe
from new_watermark.learning_channel.learner import attach_ordinary_lora
from new_watermark.learning_channel.carrier import key_bits

args.output.mkdir(parents=True, exist_ok=False)
config = dict(read_json(args.smoke_run / "config.json"), meta_probe_episodes=2,
    meta_probe_train_size=2, meta_probe_eval_size=2, meta_probe_horizons=[2, 4], gpu_cache_gb=0)
write_json(args.output / "config.json", config)
for name in ("assets.json", "prompts.json"):
    write_json(args.output / name, read_json(args.smoke_run / name))
setup_storage(args.output, "/root/logs/runtime/lcf_verification", "/root/logs/output_image/lcf_verification")
state = torch.load(storage(args.smoke_run)["artifacts"] / "teacher_selected.pt", map_location="cpu", weights_only=True)
state["config"] = config
path = storage(args.output)["artifacts"] / "smoke_weights.pt"
torch.save(state, path)
gpu_setup(config)
pipe = load_pipeline(args.output)
carrier, reader = load_state(path, pipe, config)
reader.eval().requires_grad_(False)
initial = attach_ordinary_lora(pipe.unet, config["design_rank"])
rows = meta_probe(args.output, pipe, PrefixCache(args.output, pipe), carrier, reader, initial,
    DDPMScheduler.from_config(pipe.scheduler.config),
    key_bits(config["meta_keys"], config["bits"], device="cuda"), [.05] * len(config["meta_keys"]))
assert len(rows) == 2
for row in rows:
    assert len(row["episodes"]) == 8
    assert {r["horizon"] for r in row["horizons"]} == {2, 4}
    assert all(torch.isfinite(torch.tensor(r["paired_effect"])) for r in row["episodes"])
write_json(args.output / "verification.json", dict(passed=True, scientific_result=False,
    purpose="Exercise disjoint multi-episode/two-horizon GPU paths with smoke weights",
    rows=rows, max_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3))
print("Multi-episode/horizon GPU smoke passed", flush=True)

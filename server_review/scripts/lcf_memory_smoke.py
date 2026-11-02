"""Non-scientific 512px/full-pilot-shape higher-order memory check.

Resize a tiny smoke carrier solely to exercise the production graph dimensions.
No checkpoint or score from this check is used by the scientific pilot.
"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code", type=Path, required=True)
parser.add_argument("--smoke-run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.code))

import torch
import torch.nn.functional as F
from new_watermark.learning_channel.runtime import (setup_storage, storage, read_json,
                                                    write_json, sha256)
from new_watermark.learning_channel.train import meta_train

args.output.mkdir(parents=True, exist_ok=False)
source = read_json(args.smoke_run / "config.json")
config = dict(source, resolution=512, sampling_steps=20, tail_steps=4,
              inner_steps=4, meta_sampling_steps=8, meta_steps=1,
              meta_evaluate_every=1, gpu_cache_gb=0)
write_json(args.output / "config.json", config)
for name in ("assets.json", "prompts.json", "source_manifest.json"):
    write_json(args.output / name, read_json(args.smoke_run / name))
setup_storage(args.output, "/root/logs/runtime/lcf_memory_checks", "/root/logs/output_image/lcf_memory_checks")
state = torch.load(storage(args.smoke_run)["artifacts"] / "teacher_selected.pt",
                   map_location="cpu", weights_only=True)
state["config"] = config
state["carrier"]["basis"] = F.interpolate(state["carrier"]["basis"], (8, 8), mode="bilinear", align_corners=False)
torch.save(state, storage(args.output)["artifacts"] / "teacher_selected.pt")
torch.cuda.reset_peak_memory_stats()
meta_train(args.output)
write_json(args.output / "memory_check.json", dict(scientific_result=False,
    purpose="full pilot-shaped owner hypergradient/VRAM smoke; weights excluded from pilot",
    passed=True, resolution=512, inner_steps=4, student_generation_steps=8,
    max_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
    max_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
    source_script_sha256=sha256(Path(__file__))))
print(json.dumps(read_json(args.output / "memory_check.json"), indent=2), flush=True)

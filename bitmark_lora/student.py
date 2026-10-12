"""Run upstream Diffusers trainers and audit saved adapters before inference."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import HERE, freeze, image_name, read_json, sha256, verify_vendor, write_json


def training_conditions(config):
    # Missing field denotes the original protocol, retained for analysis of old runs.
    return ("clean", "wm") if config.get("train_clean_controls", True) else ("wm",)


def branches(config):
    result = []
    for rank in config["ranks"]:
        for condition in training_conditions(config):
            result.append({"name": f"lora_r{rank}_{condition}", "kind": "lora", "rank": rank, "condition": condition})
    if not config["lora_only"]:
        for condition in training_conditions(config):
            result.append({"name": f"full_{condition}", "kind": "full", "condition": condition})
    if config["paper_full_ema"]:
        result.append({"name": "paper_full_wm_ema", "kind": "full", "condition": "wm", "ema": True})
    return result


def training_command(config, assets, run, branch):
    trainer = "train_text_to_image_lora.py" if branch["kind"] == "lora" else "train_text_to_image.py"
    cmd = [sys.executable, str(HERE / "vendor" / "diffusers" / trainer),
           "--pretrained_model_name_or_path", assets["student"],
           "--train_data_dir", str(run / "images" / f"teacher_{branch['condition']}_train"),
           "--output_dir", str(run / "students" / branch["name"]),
           "--resolution", "512", "--center_crop", "--train_batch_size", str(config["batch_size"]),
           "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
           "--num_train_epochs", str(config["epochs"]), "--learning_rate", str(config["learning_rate"]),
           "--mixed_precision", config["precision"], "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
           "--checkpointing_steps", "1000000000", "--report_to", "none", "--seed", str(config["seed"]),
           "--dataloader_num_workers", "0", "--max_grad_norm", "1.0"]
    if branch["kind"] == "lora":
        cmd += ["--rank", str(branch["rank"])]
    if branch.get("ema"):
        cmd += ["--use_ema"]
    return cmd


def checkpoint_hashes(directory):
    files = sorted(directory.rglob("*.safetensors"))
    if not files:
        raise RuntimeError(f"Training returned without a checkpoint: {directory}")
    return {str(p.relative_to(directory)): sha256(p) for p in files}


def train(run, name):
    import torch
    from safetensors.torch import load_file
    verify_vendor()
    config, assets = read_json(run / "config.json"), read_json(run / "assets.json")
    branch = next(b for b in branches(config) if b["name"] == name)
    out = run / "students" / name
    out.mkdir(parents=True, exist_ok=True)
    expected = read_json(run / "images" / f"teacher_{branch['condition']}_train" / "image_hashes.json")
    if len(expected) != config["train_n"]:
        raise RuntimeError("Training dataset incomplete")
    for entry in expected:
        path = run / "images" / f"teacher_{branch['condition']}_train" / entry["file_name"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Training image changed: {path}")
    command = training_command(config, assets, run, branch)
    freeze(out / "protocol.json", {"command": command, "branch": branch, "training_images": expected,
                                   "selection": "fixed final epoch; no owner scores or test samples used"})
    if (out / "complete.json").exists():
        if read_json(out / "complete.json")["checkpoints"] != checkpoint_hashes(out):
            raise RuntimeError(f"Saved checkpoint changed: {out}")
        print(f"Reusing verified trained student: {name}", flush=True)
        return
    # A partial run restarts deterministically. No silent selection of an earlier checkpoint.
    subprocess.run(command, check=True, cwd=HERE)
    checkpoint = out / "pytorch_lora_weights.safetensors" if branch["kind"] == "lora" else out / "unet/diffusion_pytorch_model.safetensors"
    tensors = load_file(str(checkpoint))
    if not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise RuntimeError(f"Nonfinite trained weights: {checkpoint}")
    if branch["kind"] == "lora":
        learned = [v for k, v in tensors.items() if "lora_B" in k or "lora.up" in k]
        if not learned or not any(t.abs().max().item() > 0 for t in learned):
            raise RuntimeError("LoRA up-projections remained zero; no adaptation occurred")
    else:
        initial = Path(assets["student"]) / "unet/diffusion_pytorch_model.safetensors"
        initial_tensors = load_file(str(initial))
        if not any(not torch.equal(v, initial_tensors[k]) for k, v in tensors.items()):
            raise RuntimeError("Full UNet did not change during training")
    write_json(out / "complete.json", {"checkpoints": checkpoint_hashes(out), "finite_weights": True,
                                       "actual_updates_verified": True})


def generate(run, name):
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from teacher import valid_image
    config, assets = read_json(run / "config.json"), read_json(run / "assets.json")
    prompts = read_json(run / "prompts.json")["test"]
    pipe = StableDiffusionPipeline.from_pretrained(assets["student"], torch_dtype=torch.float16,
                                                  safety_checker=None, local_files_only=True)
    checkpoint = None
    if name != "unadapted":
        branch = next(b for b in branches(config) if b["name"] == name)
        directory = run / "students" / name
        checkpoint = read_json(directory / "complete.json")["checkpoints"]
        if checkpoint != checkpoint_hashes(directory):
            raise RuntimeError(f"Changed checkpoint before generation: {directory}")
        if branch["kind"] == "lora":
            pipe.load_lora_weights(str(directory))
        else:
            pipe.unet = UNet2DConditionModel.from_pretrained(directory / "unet", torch_dtype=torch.float16,
                                                            local_files_only=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    out = run / "images" / name
    out.mkdir(parents=True, exist_ok=True)
    freeze(out / "generation.json", {"checkpoint": checkpoint, "prompts_sha256": sha256(run / "prompts.json"),
           "seed": config["seed"], "steps": config["inference_steps"], "cfg": 7.5,
           "resolution": 512, "batch_size": config["inference_batch"], "scheduler": type(pipe.scheduler).__name__,
           "dtype": "float16", "device": "cuda"})
    for start in range(0, len(prompts), config["inference_batch"]):
        batch = prompts[start:start+config["inference_batch"]]
        paths = [out / image_name(start+j) for j in range(len(batch))]
        if not all(valid_image(p, (512, 512)) for p in paths):
            generators = [torch.Generator("cuda").manual_seed(config["seed"] * 1000000 + 500000 + start+j)
                          for j in range(len(batch))]
            with torch.inference_mode():
                images = pipe([r["text"] for r in batch], height=512, width=512, generator=generators,
                              num_inference_steps=config["inference_steps"], guidance_scale=7.5).images
            for image, path in zip(images, paths):
                temp = path.with_suffix(".partial.png")
                image.save(temp)
                temp.replace(path)
        print(f"Student {name}: {start+len(batch)}/{len(prompts)}", flush=True)
    freeze(out / "image_hashes.json", [{"file": p.name, "sha256": sha256(p)} for p in sorted(out.glob("sample_*.png"))])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=["train", "generate"], required=True)
    p.add_argument("--branch", required=True)
    a = p.parse_args()
    (train if a.stage == "train" else generate)(a.run.resolve(), a.branch)

#!/usr/bin/env bash
set -Eeuo pipefail

# Adaptive PTQ stress test for Stable Signature and AquaLoRA.
# Stable Signature: public Meta decoder + SD2.1-base (public mirror fallback).
# AquaLoRA: official ppft_trained assets + SD1.5.
#
# This uses simulated weight-only PTQ: weights are genuinely rounded to low-bit
# values, then inference runs with CUDA FP16 kernels. This makes the numerical
# experiment portable; it does not claim INT4 kernel speedup.

WMQ_ROOT="${WMQ_ROOT:-$PWD/wmq_runs}"
WMQ_VENV="${WMQ_VENV:-$WMQ_ROOT/venv}"
WMQ_OUTPUT="${WMQ_OUTPUT:-$WMQ_ROOT/output}"
WMQ_CACHE="${WMQ_CACHE:-$WMQ_ROOT/hf_cache}"
WMQ_PYTHON="${WMQ_PYTHON:-python3}"
WMQ_SKIP_INSTALL="${WMQ_SKIP_INSTALL:-0}"
export WMQ_ROOT WMQ_OUTPUT WMQ_CACHE
export HF_HOME="$WMQ_CACHE"
export TOKENIZERS_PARALLELISM=false
# May reduce fragmentation with PyTorch's native CUDA allocator. This cannot
# guarantee that an intrinsically too-large workload will fit in GPU memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

mkdir -p "$WMQ_ROOT" "$WMQ_OUTPUT" "$WMQ_CACHE"

if [[ "$WMQ_SKIP_INSTALL" != "1" ]]; then
  if [[ ! -x "$WMQ_VENV/bin/python" ]]; then
    "$WMQ_PYTHON" -m venv --system-site-packages "$WMQ_VENV"
  fi
  PY="$WMQ_VENV/bin/python"
  "$PY" -m pip install --upgrade pip wheel
  # Keep the server's CUDA-specific torch installation. Do not replace it here.
  "$PY" -m pip install \
    'diffusers==0.35.1' 'transformers==4.56.2' 'accelerate==1.10.1' \
    'huggingface-hub==0.34.4' 'safetensors==0.6.2' 'peft==0.17.1' \
    'lpips==0.1.4' 'scipy>=1.11,<2' 'pillow>=10' 'numpy>=1.24,<3'
else
  PY="$WMQ_PYTHON"
fi

"$PY" - <<'PY'
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from scipy.stats import binom


ROOT = Path(os.environ.get("WMQ_ROOT", Path.cwd() / "wmq_runs")).resolve()
OUT = Path(os.environ.get("WMQ_OUTPUT", ROOT / "output")).resolve()
CACHE = Path(os.environ.get("WMQ_CACHE", ROOT / "hf_cache")).resolve()
DEVICE = torch.device("cuda")
DTYPE = torch.float16

SS_MODEL = os.environ.get("SS_MODEL", "stabilityai/stable-diffusion-2-1-base")
SS_MODEL_FALLBACK = os.environ.get("SS_MODEL_FALLBACK", "sd2-community/stable-diffusion-2-1-base")
AQUA_MODEL = os.environ.get("AQUA_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
AQUA_REPO = "georgefen/AquaLoRA-Models"
AQUA_REV = os.environ.get("AQUA_REV", "98688b2a1e762339593ee8fe96ed13762f06b732")
AQUA_FOLDER = os.environ.get("AQUA_FOLDER", "ppft_trained")
SS_DECODER_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth"
SS_EXTRACTOR_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt"
SS_KEY = os.environ.get("SS_KEY", "111010110101000001010111010011010100010000100111")
AQUA_KEY = os.environ.get("AQUA_KEY", "110110010011010110001111010101010101011000100110")

SEARCH_N = int(os.environ.get("SEARCH_N", "20"))
TEST_N = int(os.environ.get("TEST_N", "100"))
STEPS = int(os.environ.get("STEPS", "25"))
GUIDANCE = float(os.environ.get("GUIDANCE", "7.0"))
HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
MIN_PSNR = float(os.environ.get("MIN_PSNR", "25.0"))
MAX_LPIPS = float(os.environ.get("MAX_LPIPS", "0.15"))
FPR = float(os.environ.get("FPR", "0.001"))
BASE_SEED = int(os.environ.get("BASE_SEED", "3407"))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "0"))  # 0 = evaluate every recipe
PROMPT_FILE = os.environ.get("PROMPT_FILE", "")

PROMPTS = [
    "a professional photograph of a red fox in a snowy forest",
    "a small sailboat at sunset, realistic photography",
    "an old stone bridge over a quiet river",
    "a bowl of fresh fruit on a wooden table, studio light",
    "a futuristic city street at night, cinematic photograph",
    "a watercolor painting of mountains and a lake",
    "a close-up portrait of a tabby cat",
    "a green bicycle leaning against a brick wall",
    "a lighthouse above a stormy sea",
    "a cozy reading room with warm afternoon light",
    "a field of sunflowers beneath a blue sky",
    "an astronaut walking through a tropical jungle",
    "a ceramic teapot beside two cups",
    "a macro photograph of a butterfly on a flower",
    "a train crossing a valley in autumn",
    "a detailed illustration of a medieval market",
    "a modern glass building reflected in a pond",
    "a dog running on a beach, action photograph",
    "a quiet street in a European village",
    "an aerial photograph of winding coastal roads",
    "a robot preparing dinner in a home kitchen",
    "a misty pine forest at dawn",
    "a colorful hot air balloon above farmland",
    "an antique camera and notebooks on a desk",
]


def experiment_prompts(required):
    """Load one prompt per line/JSON item, or deterministically build a stress set.

    For paper results, set PROMPT_FILE to a preregistered COCO/DrawBench prompt
    list. The built-in expansion is intended for executable preliminary tests.
    """
    if PROMPT_FILE:
        path = Path(PROMPT_FILE)
        if not path.is_file():
            fail(f"PROMPT_FILE does not exist: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("prompts", "captions", "data"):
                    if key in payload:
                        payload = payload[key]
                        break
            if not isinstance(payload, list):
                fail("PROMPT_FILE JSON must contain a list or prompts/captions/data list")
            loaded = []
            for item in payload:
                if isinstance(item, str):
                    loaded.append(item.strip())
                elif isinstance(item, dict):
                    value = item.get("prompt", item.get("caption", item.get("text", "")))
                    loaded.append(str(value).strip())
        else:
            loaded = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        loaded = [prompt for prompt in loaded if prompt]
        if len(loaded) < required:
            fail(f"PROMPT_FILE has {len(loaded)} prompts; {required} are required")
        return loaded[:required]

    subjects = [
        "a red fox", "a tabby cat", "a lighthouse", "an astronaut", "a steam train",
        "a glass building", "a ceramic teapot", "a sailboat", "a green bicycle",
        "a medieval market", "a butterfly", "a reading room", "a mountain village",
        "a friendly robot", "a bowl of fruit", "a stone bridge", "a field of flowers",
        "a coastal road", "a golden retriever", "an antique camera",
    ]
    settings = [
        "at dawn", "at sunset", "in soft studio light", "during a rainstorm",
        "beneath a clear blue sky", "in winter", "in warm afternoon light",
        "at night", "surrounded by mist", "beside a quiet river",
    ]
    styles = [
        "realistic photograph", "cinematic photograph", "detailed digital illustration",
        "watercolor painting", "documentary photograph",
    ]
    expanded = list(PROMPTS)
    for style in styles:
        for setting in settings:
            for subject in subjects:
                expanded.append(f"{style} of {subject} {setting}")
                if len(expanded) >= required:
                    return expanded
    fail(f"Unable to build {required} prompts")


def fail(message):
    raise RuntimeError(message)


def download_url(url, destination):
    import urllib.request
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url} -> {destination}", flush=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destination)
    return destination


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def build_pipeline(model_id, fallback_id=None):
    kwargs = dict(torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False,
                  cache_dir=str(CACHE), use_safetensors=True)
    candidates = [model_id] + ([fallback_id] if fallback_id and fallback_id != model_id else [])
    last_error = None
    pipe = None
    used_model = None
    for candidate in candidates:
        print(f"Loading {candidate}", flush=True)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(candidate, **kwargs)
            used_model = candidate
            break
        except OSError as error:
            last_error = error
            if candidate != candidates[-1]:
                print(f"Could not load {candidate}; trying the declared public mirror", flush=True)
    if pipe is None:
        raise last_error
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing("max")
    pipe.to(DEVICE)
    pipe._wmq_model_id = used_model
    return pipe


@torch.inference_mode()
def generate(pipe, prompts, seeds):
    images = []
    for prompt, seed in zip(prompts, seeds):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        image = pipe(prompt, height=HEIGHT, width=WIDTH,
                     num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                     generator=generator).images[0]
        images.append(image)
    return images


def pil_tensor(images, normalized=False):
    tensors = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
    result = torch.stack(tensors).to(DEVICE)
    return result * 2 - 1 if normalized else result


def image_metrics(clean, attacked, lpips_model):
    x = pil_tensor(clean)
    y = pil_tensor(attacked)
    mse_per = ((x - y) ** 2).flatten(1).mean(1)
    psnr = (-10.0 * torch.log10(mse_per.clamp_min(1e-12))).mean().item()
    with torch.inference_mode():
        lp = lpips_model(x * 2 - 1, y * 2 - 1).mean().item()
    return {"psnr": psnr, "lpips": lp}


def detection_threshold(nbits, fpr):
    # Smallest number of matching bits whose random-match upper tail <= FPR.
    for matches in range(nbits + 1):
        if float(binom.sf(matches - 1, nbits, 0.5)) <= fpr:
            return matches
    return nbits + 1


def summarize_bits(predictions, key, fpr=FPR):
    truth = torch.tensor([int(x) for x in key], dtype=torch.bool)
    matches = (predictions.cpu().bool() == truth[None, :]).sum(1).numpy()
    threshold = detection_threshold(len(key), fpr)
    pvalues = [float(binom.sf(int(m) - 1, len(key), 0.5)) for m in matches]
    bitacc = matches / len(key)
    return {
        "bit_accuracy": float(bitacc.mean()),
        "bit_accuracy_std": float(bitacc.std(ddof=1)) if len(bitacc) > 1 else 0.0,
        "tpr": float((matches >= threshold).mean()),
        "threshold_matches": int(threshold),
        "mean_pvalue": float(np.mean(pvalues)),
        "per_image_bit_accuracy": bitacc.tolist(),
        "per_image_pvalue": pvalues,
    }


@torch.inference_mode()
def decode_stable_signature(extractor, images):
    transform = TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    batch = transform(pil_tensor(images))
    logits = extractor(batch)
    return logits > 0


class AquaSecretDecoder(nn.Module):
    def __init__(self, output_size=48):
        super().__init__()
        from torchvision.models import efficientnet_b1
        self.model = efficientnet_b1(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,
                                             output_size * 2)
        self.output_size = output_size

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        return self.model(x).view(-1, self.output_size, 2)


@torch.inference_mode()
def decode_aqualora(decoder, images):
    logits = decoder(pil_tensor(images, normalized=True))
    return logits.argmax(-1).bool()


class MapperNet(nn.Module):
    def __init__(self, input_size=48, output_size=320):
        super().__init__()
        self.input_size = input_size
        self.bit_embeddings = nn.Embedding(input_size, output_size)

    def forward(self, x):
        positions = torch.arange(self.input_size, device=x.device)
        encoded = self.bit_embeddings(positions) * x[:, :, None]
        return encoded.sum(dim=1) / math.sqrt(self.input_size) + 1.0


def conditioned_aqualora(folder, key, scale=1.03):
    base = load_file(str(folder / "pytorch_lora_weights.safetensors"), device="cpu")
    mapper = MapperNet(len(key), 320)
    mapper.load_state_dict(torch.load(folder / "mapper.pt", map_location="cpu",
                                      weights_only=True))
    mapper.eval()
    bits = torch.tensor([[int(x) for x in key]], dtype=torch.float32)
    mapped = mapper(bits)[0]
    result = {}
    for name, tensor in base.items():
        if "unet" not in name:
            continue
        if name.endswith("up.weight"):
            result[name] = tensor
        elif name.endswith("down.weight"):
            if "proj_in" in name or "proj_out" in name:
                result[name] = tensor * mapped[:, None, None, None] * scale
            elif "attn" in name or "ff" in name:
                result[name] = mapped[:, None] * tensor * scale
    return result


def resolve_module(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def aqua_target_path(down_key):
    path = down_key[len("unet."): -len(".lora.down.weight")]
    replacements = {
        ".processor.to_q_lora": ".to_q",
        ".processor.to_k_lora": ".to_k",
        ".processor.to_v_lora": ".to_v",
        ".processor.to_out_lora": ".to_out.0",
        ".processor.net.0.proj_lora": ".net.0.proj",
        ".processor.net.2_lora": ".net.2",
    }
    for old, new in replacements.items():
        path = path.replace(old, new)
    return path


def fuse_aqualora(unet, lora):
    fused = 0
    unresolved = []
    for down_key in sorted(k for k in lora if k.endswith(".lora.down.weight")):
        up_key = down_key.replace(".lora.down.weight", ".lora.up.weight")
        if up_key not in lora:
            unresolved.append((down_key, "missing up tensor"))
            continue
        path = aqua_target_path(down_key)
        try:
            module = resolve_module(unet, path)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            unresolved.append((down_key, f"{path}: {error}"))
            continue
        down = lora[down_key].float()
        up = lora[up_key].float()
        if down.ndim == 2 and up.ndim == 2:
            delta = up @ down
        elif down.ndim == 4 and up.ndim == 4 and down.shape[2:] == (1, 1) and up.shape[2:] == (1, 1):
            delta = (up[:, :, 0, 0] @ down[:, :, 0, 0])[:, :, None, None]
        else:
            unresolved.append((down_key, f"unsupported shapes {tuple(down.shape)}, {tuple(up.shape)}"))
            continue
        if tuple(delta.shape) != tuple(module.weight.shape):
            unresolved.append((down_key, f"shape {tuple(delta.shape)} != {tuple(module.weight.shape)}"))
            continue
        module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        fused += 1
    if unresolved:
        preview = "\n".join(f"  {key}: {why}" for key, why in unresolved[:10])
        fail(f"AquaLoRA fusion was incomplete ({len(unresolved)} unresolved):\n{preview}")
    if fused < 10:
        fail(f"Only {fused} AquaLoRA modules were fused; expected many more")
    print(f"Fused AquaLoRA into {fused} UNet modules", flush=True)


def clone_state(module):
    return OrderedDict((name, tensor.detach().cpu().clone())
                       for name, tensor in module.state_dict().items())


def restore_state(module, state):
    module.load_state_dict(state, strict=True)


def group_name(parameter_name, component):
    if component == "vae":
        if parameter_name.startswith("decoder.mid_block"):
            return "mid"
        if parameter_name.startswith("decoder.up_blocks.0") or parameter_name.startswith("decoder.up_blocks.1"):
            return "up_early"
        if parameter_name.startswith("decoder.up_blocks.2") or parameter_name.startswith("decoder.up_blocks.3"):
            return "up_late"
        if parameter_name.startswith("decoder.conv") or parameter_name.startswith("post_quant_conv"):
            return "io"
        return "other"
    if parameter_name.startswith("down_blocks"):
        return "down"
    if parameter_name.startswith("mid_block"):
        return "mid"
    if parameter_name.startswith("up_blocks"):
        return "up"
    if "conv_in" in parameter_name or "conv_out" in parameter_name:
        return "io"
    return "other"


def quantize_tensor(weight, bits, clip):
    if not weight.is_floating_point() or weight.ndim < 2:
        return weight
    source = weight.float()
    flat = source.reshape(source.shape[0], -1)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * clip
    qmax = float(2 ** (bits - 1) - 1)
    scale = max_abs / qmax
    quantized = torch.round(flat.clamp(-max_abs, max_abs) / scale).clamp(-qmax, qmax) * scale
    return quantized.reshape_as(source).to(weight.dtype)


def apply_recipe(module, component, recipe):
    selected = set(recipe["groups"])
    changed = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not name.endswith("weight") or parameter.ndim < 2:
                continue
            # Stable Signature is carried by the VAE decoder. Its encoder is
            # unused during text-to-image generation and is outside this test.
            if component == "vae" and not (
                name.startswith("decoder.") or name.startswith("post_quant_conv.")
            ):
                continue
            if group_name(name, component) not in selected and "all" not in selected:
                continue
            parameter.copy_(quantize_tensor(parameter, recipe["bits"], recipe["clip"]))
            changed += parameter.numel()
    if changed == 0:
        fail(f"Recipe selected no parameters: {recipe}")
    return changed


def candidate_recipes(component):
    groups = ["mid", "up_early", "up_late", "io"] if component == "vae" else ["down", "mid", "up", "io"]
    recipes = []
    for bits in (8, 6, 4, 3, 2):
        for clip in (1.0, 0.75):
            recipes.append({"bits": bits, "clip": clip, "groups": ["all"]})
            for group in groups:
                recipes.append({"bits": bits, "clip": clip, "groups": [group]})
    if MAX_CANDIDATES > 0 and MAX_CANDIDATES < len(recipes):
        print(f"WARNING: MAX_CANDIDATES={MAX_CANDIDATES} truncates the full "
              f"{len(recipes)}-recipe registry. Set MAX_CANDIDATES=0 for all recipes.", flush=True)
        return recipes[:MAX_CANDIDATES]
    return recipes


def save_images(images, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(folder / f"{idx:04d}.png")


def run_adaptive_search(name, pipe, component, decoder, decode_fn, key, lpips_model):
    result_dir = OUT / name
    result_dir.mkdir(parents=True, exist_ok=True)
    target = pipe.vae if component == "vae" else pipe.unet
    pristine = clone_state(target)
    all_prompts = experiment_prompts(SEARCH_N + TEST_N)
    prompts_search = all_prompts[:SEARCH_N]
    prompts_test = all_prompts[SEARCH_N: SEARCH_N + TEST_N]
    search_seeds = [BASE_SEED + i for i in range(SEARCH_N)]
    test_seeds = [BASE_SEED + 10000 + i for i in range(TEST_N)]

    clean_search = generate(pipe, prompts_search, search_seeds)
    clean_test = generate(pipe, prompts_test, test_seeds)
    clean_bits = summarize_bits(decode_fn(decoder, clean_test), key)
    save_images(clean_test, result_dir / "clean_test")
    if clean_bits["bit_accuracy"] < 0.80:
        fail(f"{name} clean bit accuracy is only {clean_bits['bit_accuracy']:.3f}; "
             "watermark/checkpoint loading is invalid, so attack search was stopped")

    rows = []
    best = None
    for index, recipe in enumerate(candidate_recipes(component)):
        restore_state(target, pristine)
        changed = apply_recipe(target, component, recipe)
        attacked = generate(pipe, prompts_search, search_seeds)
        quality = image_metrics(clean_search, attacked, lpips_model)
        bits = summarize_bits(decode_fn(decoder, attacked), key)
        feasible = quality["psnr"] >= MIN_PSNR and quality["lpips"] <= MAX_LPIPS
        row = {
            "candidate": index, **recipe, "changed_parameters": changed,
            "search_bit_accuracy": bits["bit_accuracy"], "search_tpr": bits["tpr"],
            "search_psnr": quality["psnr"], "search_lpips": quality["lpips"],
            "feasible": feasible,
        }
        rows.append(row)
        print(f"{name} candidate {index:02d}: {row}", flush=True)
        rank = (bits["bit_accuracy"], bits["tpr"], -quality["psnr"], quality["lpips"])
        if feasible and (best is None or rank < best[0]):
            best = (rank, dict(recipe), index)
        del attacked
        cleanup()

    with open(result_dir / "search.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best is None:
        restore_state(target, pristine)
        fail(f"{name}: no candidate passed PSNR>={MIN_PSNR} and LPIPS<={MAX_LPIPS}")

    _, recipe, candidate_index = best
    restore_state(target, pristine)
    changed = apply_recipe(target, component, recipe)
    attacked_test = generate(pipe, prompts_test, test_seeds)
    attack_bits = summarize_bits(decode_fn(decoder, attacked_test), key)
    quality = image_metrics(clean_test, attacked_test, lpips_model)
    save_images(attacked_test, result_dir / "attacked_test")
    report = {
        "watermark": name,
        "model": getattr(pipe, "_wmq_model_id", SS_MODEL if name == "stable_signature" else AQUA_MODEL),
        "quantization": "simulated per-output-channel symmetric weight PTQ",
        "selected_candidate": candidate_index,
        "recipe": recipe,
        "changed_parameters": changed,
        "search_size": SEARCH_N,
        "test_size": TEST_N,
        "quality_constraints": {"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS},
        "clean": clean_bits,
        "attacked": attack_bits,
        "paired_quality": quality,
        "bit_accuracy_retained_percent": 100.0 * attack_bits["bit_accuracy"] / max(clean_bits["bit_accuracy"], 1e-12),
        "tpr_retained_percent": 100.0 * attack_bits["tpr"] / max(clean_bits["tpr"], 1e-12),
    }
    with open(result_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    restore_state(target, pristine)
    return report


def load_stable_signature():
    assets = ROOT / "assets"
    decoder_path = download_url(SS_DECODER_URL, assets / "sd2_decoder.pth")
    extractor_path = download_url(SS_EXTRACTOR_URL, assets / "dec_48b_whit.torchscript.pt")
    pipe = build_pipeline(SS_MODEL, SS_MODEL_FALLBACK)
    ldm_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    converted = convert_ldm_vae_checkpoint(ldm_state, dict(pipe.vae.config))
    decoder_keys = {k: v for k, v in converted.items()
                    if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
    loaded = pipe.vae.load_state_dict(decoder_keys, strict=False)
    unexpected = list(loaded.unexpected_keys)
    missing_decoder = [k for k in loaded.missing_keys
                       if k.startswith("decoder.") or k.startswith("post_quant_conv.")]
    if unexpected or missing_decoder or len(decoder_keys) < 100:
        fail(f"Stable Signature decoder conversion invalid: loaded={len(decoder_keys)}, "
             f"unexpected={unexpected[:5]}, missing_decoder={missing_decoder[:5]}")
    extractor = torch.jit.load(str(extractor_path), map_location=DEVICE).eval()
    print(f"Stable Signature assets SHA256: decoder={sha256(decoder_path)}, extractor={sha256(extractor_path)}")
    return pipe, extractor


def load_aqualora():
    folder = ROOT / "assets" / "aqualora_ppft"
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("mapper.pt", "msgdecoder.pt", "pytorch_lora_weights.safetensors"):
        cached = hf_hub_download(AQUA_REPO, f"{AQUA_FOLDER}/{filename}", revision=AQUA_REV,
                                 cache_dir=str(CACHE))
        destination = folder / filename
        if not destination.exists():
            import shutil
            shutil.copy2(cached, destination)
    pipe = build_pipeline(AQUA_MODEL)
    lora = conditioned_aqualora(folder, AQUA_KEY)
    fuse_aqualora(pipe.unet, lora)
    decoder = AquaSecretDecoder(len(AQUA_KEY))
    decoder.load_state_dict(torch.load(folder / "msgdecoder.pt", map_location="cpu",
                                       weights_only=True), strict=True)
    decoder.to(device=DEVICE, dtype=torch.float32).eval()
    return pipe, decoder


def main():
    if not torch.cuda.is_available():
        fail("CUDA is required, but torch.cuda.is_available() is False")
    if diffusers.__version__ != "0.35.1":
        fail(f"This script validates AquaLoRA module mapping only for diffusers==0.35.1; "
             f"found {diffusers.__version__}")
    if len(SS_KEY) != 48 or len(AQUA_KEY) != 48 or set(SS_KEY + AQUA_KEY) - {"0", "1"}:
        fail("SS_KEY and AQUA_KEY must each be 48 binary characters")
    if HEIGHT % 8 or WIDTH % 8:
        fail("HEIGHT and WIDTH must be divisible by 8")
    if SEARCH_N < 1 or TEST_N < 1:
        fail("SEARCH_N and TEST_N must be positive")
    torch.manual_seed(BASE_SEED)
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    print("NOTE: low-bit weights run through FP16 CUDA kernels; latency is not an INT4 benchmark.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()
    reports = []

    pipe, extractor = load_stable_signature()
    reports.append(run_adaptive_search("stable_signature", pipe, "vae", extractor,
                                       decode_stable_signature, SS_KEY, lpips_model))
    del pipe, extractor
    cleanup()

    pipe, decoder = load_aqualora()
    reports.append(run_adaptive_search("aqualora", pipe, "unet", decoder,
                                       decode_aqualora, AQUA_KEY, lpips_model))
    del pipe, decoder
    cleanup()

    summary = {
        "created_unix": time.time(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "settings": {
            "search_n": SEARCH_N, "test_n": TEST_N, "steps": STEPS,
            "guidance": GUIDANCE, "height": HEIGHT, "width": WIDTH,
            "fpr": FPR, "min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS,
        },
        "reports": reports,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    retention_rows = []
    for report in reports:
        retention_rows.append({
            "watermark": report["watermark"],
            "model": report["model"],
            "bits": report["recipe"]["bits"],
            "clip": report["recipe"]["clip"],
            "groups": "+".join(report["recipe"]["groups"]),
            "clean_bit_accuracy_percent": 100.0 * report["clean"]["bit_accuracy"],
            "attacked_bit_accuracy_percent": 100.0 * report["attacked"]["bit_accuracy"],
            "bit_accuracy_retained_percent": report["bit_accuracy_retained_percent"],
            "clean_tpr_percent": 100.0 * report["clean"]["tpr"],
            "attacked_tpr_percent": 100.0 * report["attacked"]["tpr"],
            "tpr_retained_percent": report["tpr_retained_percent"],
            "paired_psnr": report["paired_quality"]["psnr"],
            "paired_lpips": report["paired_quality"]["lpips"],
            "test_images": report["test_size"],
        })
    with open(OUT / "watermark_retention.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retention_rows[0].keys()))
        writer.writeheader()
        writer.writerows(retention_rows)
    print(f"Finished. Summary: {OUT / 'summary.json'}", flush=True)
    print(f"Plot-ready table: {OUT / 'watermark_retention.csv'}", flush=True)


if __name__ == "__main__":
    main()
PY

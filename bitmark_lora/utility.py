"""Utility audit after frozen training; never an input to checkpoint selection."""
import argparse
from pathlib import Path

from common import read_json, write_json
from student import branches

CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def main(run):
    import torch
    import torch_fidelity
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    c = read_json(run / "config.json")
    prompts = read_json(run / "prompts.json")["test"]
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=CLIP_REV).cuda().eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision=CLIP_REV)
    labels = ["unadapted"] + [b["name"] for b in branches(c)]
    ref = run / "images/teacher_clean_test"
    report = {"reference": str(ref), "clip_revision": CLIP_REV, "n": len(prompts),
              "clip_definition": "mean 2.5 * max(cos(image,text),0), same as SERUM utility script",
              "fid_warning": "Finite-sample FID is biased; compare only identical counts/reference. KID included.", "branches": {}}
    for label in labels:
        folder = run / "images" / label
        values = []
        files = sorted(folder.glob("sample_*.png"))
        if len(files) != len(prompts):
            raise ValueError(f"Incomplete utility images: {folder}")
        for start in range(0, len(files), 16):
            images = [Image.open(p).convert("RGB") for p in files[start:start+16]]
            inputs = processor(text=[x["text"] for x in prompts[start:start+16]], images=images,
                               return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
                value = (outputs.image_embeds * outputs.text_embeds).sum(-1).clamp_min(0) * 2.5
            values += value.cpu().tolist()
        print(f"Utility {label}: CLIP done; computing FID/KID", flush=True)
        metrics = torch_fidelity.calculate_metrics(input1=str(folder), input2=str(ref), cuda=True,
            isc=False, fid=True, kid=len(files) >= 2, kid_subsets=100,
            kid_subset_size=min(1000, len(files)), rng_seed=c["seed"], verbose=True)
        report["branches"][label] = {"clip_score": sum(values)/len(values), "clip_per_image": values, **metrics}
        write_json(run / "result/utility.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run.resolve())

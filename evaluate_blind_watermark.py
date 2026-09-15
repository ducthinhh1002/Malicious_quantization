"""Owner-only evaluation AFTER model-only selection. Never import from attacker."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binom


def detection_threshold(bits, fpr):
    if bits < 1 or not 0 < fpr < 1:
        raise ValueError("Require positive key length and 0 < FPR < 1")
    for k in range(bits + 1):
        if binom.sf(k - 1, bits, .5) <= fpr:
            return k
    raise ValueError(f"Unattainable detection threshold: {bits}-bit key cannot achieve FPR={fpr:g} "
                     f"even with every bit correct (minimum theoretical FPR=2^-{bits}). "
                     "Use a longer key or a less stringent FPR; no evaluation was performed.")


def verify_frozen_run(root, report):
    frozen = json.loads((root / "selection_frozen.json").read_text(encoding="utf-8"))
    if (frozen.get("phase") != "before_test_generation" or frozen["selected"] != report["selected"]
            or frozen["export_sha256"] != report["export_sha256"]):
        raise ValueError("Frozen selection and final report disagree")
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    actual = {p.name: digest(p) for p in (root / "quantized_vae").iterdir() if p.is_file()}
    if actual != frozen["export_sha256"] or digest(root / "search.json") != frozen["search_sha256"]:
        raise ValueError("Checkpoint or search log changed after selection freeze")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--extractor", required=True, help="Owner's Stable Signature TorchScript extractor")
    p.add_argument("--key", required=True, help="Owner's binary secret")
    p.add_argument("--fpr", type=float, default=.001)
    p.add_argument("--lpips", action="store_true", help="Owner-only independent quality metric; may download AlexNet")
    args = p.parse_args()
    if not args.key or set(args.key) - {"0", "1"} or not 0 < args.fpr < 1:
        raise ValueError("Invalid key/FPR")
    threshold = detection_threshold(len(args.key), args.fpr)
    root = Path(args.run)
    report = json.loads((root / "report.json").read_text())
    if report["test_used_for_selection"]:
        raise ValueError("Test leakage")
    verify_frozen_run(root, report)
    manifest = json.loads((root / "manifest.json").read_text())
    n = manifest["args"]["test_n"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.jit.load(args.extractor, map_location=device).eval()
    key = torch.tensor([int(b) for b in args.key], device=device).bool()
    metric = None
    if args.lpips:
        import lpips
        metric = lpips.LPIPS(net="alex").to(device).eval()
    def tensor(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
    rows = []
    with torch.inference_mode():
        for label in ("marked_reference_test", "matched_rtn_test", "attacked_test"):
            paths = sorted((root / label).glob("*.png"))
            if len(paths) != n:
                raise ValueError(f"Wrong image count in {label}")
            accuracies, detected, perceptual = [], [], []
            for i, path in enumerate(paths):
                if path.name != f"{i:04d}.png":
                    raise ValueError("Unexpected paired image name")
                x = tensor(path)
                mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
                std = x.new_tensor([.229, .224, .225])[None, :, None, None]
                logits = net((x - mean) / std)
                if logits.numel() != len(key) or not torch.isfinite(logits).all():
                    raise ValueError("Extractor output invalid")
                matches = ((logits.reshape(-1) > 0) == key).sum().item()
                accuracies.append(matches / len(key))
                detected.append(matches >= threshold)
                if metric is not None:
                    reference = tensor(root / "marked_reference_test" / path.name)
                    perceptual.append(metric(x * 2 - 1, reference * 2 - 1).item())
            rows.append({"method": label, "bit_accuracy": float(np.mean(accuracies)),
                         "tpr": float(np.mean(detected)), "n": n,
                         "threshold_matches": threshold,
                         "lpips": float(np.mean(perceptual)) if perceptual else None,
                         "per_image_bit_accuracy": accuracies, "per_image_detected": detected})
    clean_detected = np.array(rows[0]["per_image_detected"], dtype=bool)
    for row in rows:
        flags = np.array(row["per_image_detected"], dtype=bool)
        row["survival_on_originally_detected"] = float(flags[clean_detected].mean()) if clean_detected.any() else None
        row["bit_accuracy_retained_percent"] = 100 * row["bit_accuracy"] / rows[0]["bit_accuracy"] if rows[0]["bit_accuracy"] else None
        row["tpr_retained_percent"] = 100 * row["tpr"] / rows[0]["tpr"] if rows[0]["tpr"] else None
    payload = {"selection_already_frozen": True, "fpr_target": args.fpr,
               "fpr_calibration": "theoretical fair-independent-bit null, not empirical FPR",
               "test_quality_valid": report["test_quality"]["feasible"], "rows": rows,
               "warning": "These metrics must not be fed back into the attacker selection in the same blind experiment."}
    path = root / "owner_evaluation.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    fields = [k for k in rows[0] if not k.startswith("per_image")]
    with (root / "watermark_retention.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([{k: v for k, v in r.items() if k in fields} for r in rows], indent=2))


if __name__ == "__main__":
    main()

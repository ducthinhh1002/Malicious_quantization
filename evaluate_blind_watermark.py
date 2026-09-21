"""Owner-only evaluation AFTER model-only selection. Never import from attacker."""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binom, binomtest
from wmq_runtime import batches, batch_size, EVENTS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes, n):
    if n == 0:
        return (None, None)
    ci = binomtest(int(successes), int(n)).proportion_ci(confidence_level=.95, method="wilson")
    return float(ci.low), float(ci.high)


def paired_drop_interval(reference, attacked, repeats=2000, seed=3407):
    """Resample paired images, never individual bits or independent treatment arms."""
    delta = np.asarray(reference, dtype=float) - np.asarray(attacked, dtype=float)
    if delta.ndim != 1 or not len(delta) or len(reference) != len(attacked):
        raise ValueError("Need nonempty paired detection arrays")
    rng = np.random.default_rng(seed)
    estimates = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repeats)]
    return tuple(float(v) for v in np.quantile(estimates, [.025, .975]))


def evaluation_branches(manifest):
    branches = manifest.get("test_branches")
    if branches is None and manifest.get("schema_version", 1) == 1:
        # Compatibility only: all new experiments must declare their branches.
        branches = [{"label": name} for name in
                    ("marked_reference_test", "matched_rtn_test", "attacked_test")]
    if not isinstance(branches, list) or not branches:
        raise ValueError("Manifest must declare nonempty test_branches")
    labels = [b.get("label") for b in branches if isinstance(b, dict)]
    if (len(labels) != len(branches) or any(not isinstance(s, str) or not s
            or s in (".", "..") or "/" in s or "\\" in s for s in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Invalid or duplicate branch labels")
    reference = manifest.get("reference_label", "marked_reference_test")
    if reference not in labels:
        raise ValueError("Reference branch missing from manifest")
    return branches, reference


def detection_threshold(bits, fpr, detector="double"):
    if bits < 1 or not 0 < fpr < 1:
        raise ValueError("Require positive key length and 0 < FPR < 1")
    if detector not in ("single", "double"):
        raise ValueError("Unknown detector")
    factor = 2 if detector == "double" else 1
    for k in range(bits // 2 + 1 if detector == "double" else 0, bits + 1):
        if factor * binom.sf(k - 1, bits, .5) <= fpr:
            return k
    raise ValueError(f"Unattainable detection threshold: {bits}-bit key cannot achieve FPR={fpr:g} "
                     f"(minimum theoretical FPR={factor * 2. ** -bits:g}, detector={detector}). "
                     "Use a longer key or a less stringent FPR; no evaluation was performed.")


def resolve_image_root(root, manifest, override=None):
    """Old runs keep images inside the report directory; new runs declare a separate root."""
    if override is not None:
        return Path(override).resolve()
    return (Path(root) / manifest.get("image_root", ".")).resolve()


def verify_frozen_run(root, report, image_root=None):
    root = Path(root).resolve()
    frozen = json.loads((root / "selection_frozen.json").read_text(encoding="utf-8"))
    if frozen.get("phase") != "before_test_generation":
        raise ValueError("Frozen selection and final report disagree")
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    if frozen.get("schema_version") == 2:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        branches, reference = evaluation_branches(manifest)
        expected = {b["label"] for b in branches if b["label"] != reference and b.get("role") != "diagnostic"}
        if (digest(root / "manifest.json") != frozen["manifest_sha256"]
                or digest(root / "search.json") != frozen["search_sha256"]
                or digest(root / "selection_frozen.json") != report["selection_frozen_sha256"]
                or set(frozen["branches"]) != expected or set(report["selections"]) != expected
                or set(report["branch_quality"]) != {b["label"] for b in branches}):
            raise ValueError("Protocol, search or branch set changed after selection freeze")
        for branch in branches:
            label = branch["label"]
            if label == reference or branch.get("role") == "diagnostic":
                continue
            entry = frozen["branches"][label]
            folder = (root / branch["artifact"]).resolve()
            if not folder.is_relative_to(root.resolve()) or folder == root.resolve():
                raise ValueError("Artifact path escapes run directory")
            actual = {str(p.relative_to(root)): digest(p) for p in folder.rglob("*") if p.is_file()}
            if entry["selected"] != report["selections"][label] or actual != entry["files_sha256"]:
                raise ValueError(f"Branch {label} changed after selection freeze")
        profiles = {p.name: digest(p) for p in root.glob("profile_*.json")}
        if profiles != frozen["profile_sha256"]:
            raise ValueError("Layer profile changed after selection freeze")
        images = resolve_image_root(root, manifest, image_root)
        if not images.is_dir():
            raise FileNotFoundError(f"Image directory missing: {images}. Supply --image-root if images were moved.")
        actual_images = {(p.relative_to(images).as_posix() if "image_root" in manifest else str(p.relative_to(images))): digest(p) for b in branches
                         for p in (images / b["label"]).glob("*.png")}
        if actual_images != report["test_images_sha256"]:
            raise ValueError("Test images changed after run completion")
        return
    if frozen["selected"] != report["selected"] or frozen["export_sha256"] != report["export_sha256"]:
        raise ValueError("Frozen selection and final report disagree")
    actual = {p.name: digest(p) for p in (root / "quantized_vae").iterdir() if p.is_file()}
    if actual != frozen["export_sha256"] or digest(root / "search.json") != frozen["search_sha256"]:
        raise ValueError("Checkpoint or search log changed after selection freeze")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--image-root", help="Override image location after relocation; image hashes must still match")
    p.add_argument("--batch-size", type=int, default=0, help="0 = automatic from free VRAM; CUDA OOM halves the batch")
    p.add_argument("--extractor", required=True, help="Owner's Stable Signature TorchScript extractor")
    p.add_argument("--expected-extractor-sha256", help="Refuse to load an extractor with another digest")
    p.add_argument("--key", required=True, help="Owner's binary secret")
    p.add_argument("--key-source", default="unspecified", help="Provenance label written to the report")
    p.add_argument("--min-reference-tpr", type=float, default=.9,
                   help="Validity check for the key/extractor/marked-reference combination")
    p.add_argument("--fpr", type=float, default=.001)
    p.add_argument("--detector", choices=["single", "double"], default="double",
                   help="FPR is the TOTAL probability across both tails for double")
    p.add_argument("--lpips", action="store_true", help="Owner-only independent quality metric; may download AlexNet")
    args = p.parse_args()
    EVENTS.clear()
    if args.batch_size < 0:
        raise ValueError("Batch size must be nonnegative")
    if (not args.key or set(args.key) - {"0", "1"} or not 0 < args.fpr < 1
            or not 0 <= args.min_reference_tpr <= 1):
        raise ValueError("Invalid key/FPR")
    extractor_digest = file_sha256(args.extractor)
    if args.expected_extractor_sha256 and extractor_digest != args.expected_extractor_sha256:
        raise ValueError(f"Extractor SHA256 mismatch: expected {args.expected_extractor_sha256}, got {extractor_digest}")
    threshold = detection_threshold(len(args.key), args.fpr, args.detector)
    root = Path(args.run)
    report = json.loads((root / "report.json").read_text())
    if report["test_used_for_selection"]:
        raise ValueError("Test leakage")
    verify_frozen_run(root, report, args.image_root)
    manifest = json.loads((root / "manifest.json").read_text())
    images = resolve_image_root(root, manifest, args.image_root)
    branches, reference_label = evaluation_branches(manifest)
    n = manifest["args"]["test_n"]
    if not isinstance(n, int) or n < 1:
        raise ValueError("Invalid test sample count")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    net = torch.jit.load(args.extractor, map_location=device).eval()
    key = torch.tensor([int(b) for b in args.key], device=device).bool()
    metric = None
    if args.lpips:
        import lpips
        metric = lpips.LPIPS(net="alex").to(device).eval()
    effective_batch = batch_size(args.batch_size, device)
    def tensor(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    rows = []
    with torch.inference_mode():
        for branch in branches:
            label = branch["label"]
            paths = sorted((images / label).glob("*.png"))
            if len(paths) != n:
                raise ValueError(f"Wrong image count in {label}")
            accuracies, detected, perceptual = [], [], []
            if any(path.name != f"{i:04d}.png" for i, path in enumerate(paths)):
                raise ValueError("Unexpected paired image name")
            def evaluate(chunk):
                x = torch.cat([tensor(path) for path in chunk]).to(device)
                mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
                std = x.new_tensor([.229, .224, .225])[None, :, None, None]
                logits = net((x - mean) / std)
                if logits.numel() != len(chunk) * len(key) or not torch.isfinite(logits).all():
                    raise ValueError("Extractor output invalid")
                matches = ((logits.reshape(len(chunk), len(key)) > 0) == key).sum(1)
                distances = torch.zeros(len(chunk), device=device)
                if metric is not None:
                    reference = torch.cat([tensor(images / reference_label / path.name) for path in chunk]).to(device)
                    distances = metric(x * 2 - 1, reference * 2 - 1).reshape(len(chunk), -1).mean(1)
                return torch.stack([matches, distances], 1).cpu().tolist()
            for _, results in batches(paths, evaluate, effective_batch, "owner_evaluation"):
                for matches, distance in results:
                    accuracies.append(matches / len(key))
                    detected.append(matches >= threshold or (args.detector == "double" and matches <= len(key) - threshold))
                    if metric is not None:
                        perceptual.append(distance)
            quality = report.get("branch_quality", {}).get(label)
            selection = report.get("selections", {}).get(label)
            texture = (quality or {}).get("per_image_texture", [])
            ratios = [r["highpass_energy_ratio"] for r in texture if r["highpass_energy_ratio"] is not None]
            tpr_low, tpr_high = wilson_interval(sum(detected), n)
            rows.append({"method": label, "comparison_group": branch.get("comparison_group"),
                         "role": branch.get("role"), "detector": args.detector,
                         "test_quality_valid": quality["feasible"] if quality else None,
                         "search_quality_valid": selection["search_feasible"] if selection else None,
                         "bit_accuracy": float(np.mean(accuracies)),
                         "tpr": float(np.mean(detected)), "n": n,
                         "evasion_rate": 1. - float(np.mean(detected)),
                         "evasion_ci95_low": 1. - tpr_high, "evasion_ci95_high": 1. - tpr_low,
                         "mean_abs_bit_accuracy_minus_half": float(np.mean(np.abs(np.asarray(accuracies) - .5))),
                         "detected_count": int(sum(detected)),
                         "tpr_ci95_low": tpr_low, "tpr_ci95_high": tpr_high,
                         "threshold_matches": threshold,
                         "lower_threshold_matches": len(key) - threshold if args.detector == "double" else None,
                         "lpips": float(np.mean(perceptual)) if perceptual else None,
                         "psnr": quality.get("psnr") if quality else None,
                         "ssim": quality.get("ssim") if quality else None,
                         "highpass_energy_ratio": float(np.mean(ratios)) if ratios else None,
                         "highpass_mse": float(np.mean([r["highpass_mse"] for r in texture])) if texture else None,
                         "per_image_bit_accuracy": accuracies, "per_image_detected": detected})
    reference_row = next(row for row in rows if row["method"] == reference_label)
    clean_detected = np.array(reference_row["per_image_detected"], dtype=bool)
    reference_valid = reference_row["tpr"] >= args.min_reference_tpr
    for row in rows:
        row["reference_valid"] = reference_valid
        flags = np.array(row["per_image_detected"], dtype=bool)
        row["survival_on_originally_detected"] = float(flags[clean_detected].mean()) if clean_detected.any() else None
        row["evasion_on_originally_detected"] = 1. - row["survival_on_originally_detected"] if clean_detected.any() else None
        row["survival_ci95_low"], row["survival_ci95_high"] = wilson_interval(
            (flags & clean_detected).sum(), clean_detected.sum())
        row["tpr_drop_percentage_points"] = 100 * (reference_row["tpr"] - row["tpr"])
        low, high = paired_drop_interval(clean_detected, flags)
        row["tpr_drop_ci95_low_pp"], row["tpr_drop_ci95_high_pp"] = 100 * low, 100 * high
        row["bit_accuracy_retained_percent"] = 100 * row["bit_accuracy"] / reference_row["bit_accuracy"] if reference_row["bit_accuracy"] else None
        row["tpr_retained_percent"] = 100 * row["tpr"] / reference_row["tpr"] if reference_row["tpr"] else None
    payload = {"selection_already_frozen": True, "fpr_target": args.fpr,
               "detector": args.detector,
               "theoretical_fpr_at_threshold": float((2 if args.detector == "double" else 1) * binom.sf(threshold - 1, len(args.key), .5)),
               "evaluation_runtime": {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "batch_size": effective_batch, "events": list(EVENTS),
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "tf32": False,
                    "bitwise_reproducibility_across_devices": False},
               "fpr_calibration": "theoretical fair-independent-bit null, not empirical FPR",
               "uncertainty": {"confidence_level": .95, "tpr_and_survival": "Wilson",
                   "tpr_drop": "paired image percentile bootstrap; 2000 resamples; seed 3407",
                   "interpretation": "Nominal pointwise intervals assume independent image units, conditional on this model/key and prompt sampling scheme; not simultaneous, across-model uncertainty, or evidence that hand-written prompts represent a deployment population. Bootstrap intervals may degenerate with identical paired outcomes."},
               "reference_tpr": reference_row["tpr"],
               "baseline_detected_count": int(clean_detected.sum()),
               "reference_validation": {"valid": reference_valid,
                   "minimum_tpr": args.min_reference_tpr,
                   "interpretation": "Failure suggests a key/extractor/fixture/preprocessing mismatch or a weak marked baseline; attack effects are not scientifically interpretable."},
               "key_provenance": {"source": args.key_source,
                   "sha256": hashlib.sha256(args.key.encode("ascii")).hexdigest(), "bits": len(args.key)},
               "extractor_provenance": {"path": str(Path(args.extractor).resolve()),
                   "sha256": extractor_digest, "expected_sha256": args.expected_extractor_sha256,
                   "verified": bool(args.expected_extractor_sha256)},
               "baseline_warning": None if reference_valid else
                   "INVALID REFERENCE: marked images do not meet the minimum reference TPR. Do not attribute low attacked TPR to quantization.",
               "rows": rows,
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

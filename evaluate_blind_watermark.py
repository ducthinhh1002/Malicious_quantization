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
        expected = {b["label"] for b in branches if b["label"] != reference}
        if (digest(root / "manifest.json") != frozen["manifest_sha256"]
                or digest(root / "search.json") != frozen["search_sha256"]
                or digest(root / "selection_frozen.json") != report["selection_frozen_sha256"]
                or set(frozen["branches"]) != expected or set(report["selections"]) != expected
                or set(report["branch_quality"]) != {b["label"] for b in branches}):
            raise ValueError("Protocol, search or branch set changed after selection freeze")
        for branch in branches:
            label = branch["label"]
            if label == reference:
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
        actual_images = {str(p.relative_to(root)): digest(p) for b in branches
                         for p in (root / b["label"]).glob("*.png")}
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
    p.add_argument("--extractor", required=True, help="Owner's Stable Signature TorchScript extractor")
    p.add_argument("--expected-extractor-sha256", help="Refuse to load an extractor with another digest")
    p.add_argument("--key", required=True, help="Owner's binary secret")
    p.add_argument("--key-source", default="unspecified", help="Provenance label written to the report")
    p.add_argument("--min-reference-tpr", type=float, default=.9,
                   help="Validity check for the key/extractor/marked-reference combination")
    p.add_argument("--fpr", type=float, default=.001)
    p.add_argument("--lpips", action="store_true", help="Owner-only independent quality metric; may download AlexNet")
    args = p.parse_args()
    if (not args.key or set(args.key) - {"0", "1"} or not 0 < args.fpr < 1
            or not 0 <= args.min_reference_tpr <= 1):
        raise ValueError("Invalid key/FPR")
    extractor_digest = file_sha256(args.extractor)
    if args.expected_extractor_sha256 and extractor_digest != args.expected_extractor_sha256:
        raise ValueError(f"Extractor SHA256 mismatch: expected {args.expected_extractor_sha256}, got {extractor_digest}")
    threshold = detection_threshold(len(args.key), args.fpr)
    root = Path(args.run)
    report = json.loads((root / "report.json").read_text())
    if report["test_used_for_selection"]:
        raise ValueError("Test leakage")
    verify_frozen_run(root, report)
    manifest = json.loads((root / "manifest.json").read_text())
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
    def tensor(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
    rows = []
    with torch.inference_mode():
        for branch in branches:
            label = branch["label"]
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
                    reference = tensor(root / reference_label / path.name)
                    perceptual.append(metric(x * 2 - 1, reference * 2 - 1).item())
            quality = report.get("branch_quality", {}).get(label)
            selection = report.get("selections", {}).get(label)
            tpr_low, tpr_high = wilson_interval(sum(detected), n)
            rows.append({"method": label, "comparison_group": branch.get("comparison_group"),
                         "test_quality_valid": quality["feasible"] if quality else None,
                         "search_quality_valid": selection["search_feasible"] if selection else None,
                         "bit_accuracy": float(np.mean(accuracies)),
                         "tpr": float(np.mean(detected)), "n": n,
                         "detected_count": int(sum(detected)),
                         "tpr_ci95_low": tpr_low, "tpr_ci95_high": tpr_high,
                         "threshold_matches": threshold,
                         "lpips": float(np.mean(perceptual)) if perceptual else None,
                         "per_image_bit_accuracy": accuracies, "per_image_detected": detected})
    reference_row = next(row for row in rows if row["method"] == reference_label)
    clean_detected = np.array(reference_row["per_image_detected"], dtype=bool)
    reference_valid = reference_row["tpr"] >= args.min_reference_tpr
    for row in rows:
        flags = np.array(row["per_image_detected"], dtype=bool)
        row["survival_on_originally_detected"] = float(flags[clean_detected].mean()) if clean_detected.any() else None
        row["survival_ci95_low"], row["survival_ci95_high"] = wilson_interval(
            (flags & clean_detected).sum(), clean_detected.sum())
        row["tpr_drop_percentage_points"] = 100 * (reference_row["tpr"] - row["tpr"])
        low, high = paired_drop_interval(clean_detected, flags)
        row["tpr_drop_ci95_low_pp"], row["tpr_drop_ci95_high_pp"] = 100 * low, 100 * high
        row["bit_accuracy_retained_percent"] = 100 * row["bit_accuracy"] / reference_row["bit_accuracy"] if reference_row["bit_accuracy"] else None
        row["tpr_retained_percent"] = 100 * row["tpr"] / reference_row["tpr"] if reference_row["tpr"] else None
    payload = {"selection_already_frozen": True, "fpr_target": args.fpr,
               "theoretical_fpr_at_threshold": float(binom.sf(threshold - 1, len(args.key), .5)),
               "evaluation_runtime": {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
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

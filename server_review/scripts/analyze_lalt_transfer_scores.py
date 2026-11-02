"""Post-hoc analysis of saved LALT scores; does not train or change a detector.

Bootstrap intervals condition on each existing trained model pair. They do not
estimate variability over training seeds or establish calibrated model attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def auc(positive, negative):
    ordered = np.sort(negative)
    return float(np.mean(
        (np.searchsorted(ordered, positive, side="left")
         + np.searchsorted(ordered, positive, side="right"))
        / (2 * len(ordered))))


def summarize_scores(root, rank):
    directory = root / f"r{rank}"
    raw_path = directory / "raw_scores.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    marked = np.asarray(raw["marked_query"]["scores"], dtype=float)
    clean = np.asarray(raw["clean_query"]["scores"], dtype=float)
    calibration = np.asarray(raw["clean_calibration"]["scores"], dtype=float)
    assert marked.shape == clean.shape == (len(prompts["query"]), 2)
    assert np.isfinite(np.concatenate([marked, clean, calibration])).all()
    assert not ({p["image_id"] for p in prompts["query"]}
                & {p["image_id"] for p in prompts["calibration"]})
    delta = marked[:, 0] - clean[:, 0]
    marked_contrast = marked[:, 0] - marked[:, 1]
    clean_contrast = clean[:, 0] - clean[:, 1]
    specificity_gain = marked_contrast - clean_contrast
    calibration_contrast = calibration[:, 0] - calibration[:, 1]
    index = int(np.ceil((len(calibration_contrast) + 1) * .99)) - 1
    if index >= len(calibration_contrast):
        raise ValueError("Insufficient clean calibration samples")
    threshold = float(np.sort(calibration_contrast)[index])
    rng = np.random.default_rng(20261020)
    resamples = rng.integers(0, len(delta), (10000, len(delta)))

    def ci(values):
        return np.quantile(values[resamples].mean(axis=1), [.025, .975]).tolist()

    original = next(r for r in summary["rows"]
                    if r["branch"] == "marked" and r["key"] == 221)
    return {
        "rank": rank,
        "n": len(delta),
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "original_auc": auc(marked[:, 0], clean[:, 0]),
        "original_single_key_tpr": original["single_key_1pct_detection_rate"],
        "paired_score_gain": float(delta.mean()),
        "paired_score_gain_ci95": ci(delta),
        "paired_specificity_gain": float(specificity_gain.mean()),
        "paired_specificity_gain_ci95": ci(specificity_gain),
        "contrast_auc": auc(marked_contrast, clean_contrast),
        "contrast_threshold": threshold,
        "contrast_nominal_single_statistic_fpr": .01,
        "calibration_n": len(calibration_contrast),
        "contrast_marked_detected": int((marked_contrast > threshold).sum()),
        "contrast_clean_detected": int((clean_contrast > threshold).sum()),
        "contrast_tpr": float((marked_contrast > threshold).mean()),
        "contrast_observed_clean_fpr": float((clean_contrast > threshold).mean()),
    }


def summarize_proxy(root):
    directory = root / "meta"
    result = []
    for step in [0, 16, 32, 48, 64]:
        path = directory / ("learning_proxy_baseline.json" if step == 0
                            else f"learning_proxy_step{step}.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        limiting = min(
            (float(row[metric]), row["key"], metric)
            for row in data["rows"]
            for metric in ["paired_effect", "key_specificity"])
        result.append({
            "step": step, "selection_score": limiting[0],
            "limiting_key": limiting[1], "limiting_metric": limiting[2],
            "paired_effects": {str(r["key"]): r["paired_effect"]
                               for r in data["rows"]},
            "horizons": {str(r["key"]): r["horizons"] for r in data["rows"]},
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "date": "2026-10-20",
        "status": "post_hoc_development_analysis",
        "score_definition": "s(key221) minus s(key34); complementary keys",
        "limitations": [
            "Statistic was proposed after inspecting development results.",
            "Calibration uses clean-trained outputs available in the experiment.",
            "CI resamples paired query images, not independent training runs.",
            "No claim of unseen-key or new-model attribution.",
            "Contrast TPR is not uniformly better than the original score.",
            "Bootstrap or repeated subsets cannot create independent model trials.",
        ],
        "ranks": [summarize_scores(args.root, r) for r in [4, 16, 32]],
        "proxy_selection": summarize_proxy(args.root),
    }
    output = args.root / "analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

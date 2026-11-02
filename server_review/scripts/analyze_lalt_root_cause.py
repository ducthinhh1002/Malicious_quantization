"""Reproducible descriptive audit of the failed LALT repair; no training/test tuning."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / "server_review/lalt_root_cause_20261021"
    load = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    raw = load(ROOT / "server_review/lalt_repair/lalt_repair_20261020_135310Z/latest/recovery/raw_scores.json")
    c, m = (np.asarray(raw[n]["scores"]) for n in ("clean_query", "marked_query"))
    calibration = np.asarray(raw["unadapted_calibration"]["scores"])
    contrast = calibration[:, 0] - calibration[:, 1]
    delta = m - c
    proxy = []
    for step in (0, 16, 32, 48, 64):
        path = folder / "adaptive" / ("learning_proxy_baseline.json" if step == 0 else f"learning_proxy_step{step}.json")
        d = load(path)["rows"][0]
        proxy.append(dict(step=step, paired_effect=d["paired_effect"], specificity=d["key_specificity"],
                          horizons=d["horizons"]))
    history = [json.loads(line) for line in (folder / "adaptive/learning_channel_training.jsonl").read_text().splitlines()]
    channels = load(folder / "adaptive/channel_audit.json")
    cfg = load(folder / "adaptive/config.json")
    result = dict(
        old_selected_proxy_step=32, validation_proxy=proxy,
        proxy_train_images_per_outer_step=cfg["meta_batch"],
        inner_updates=cfg["inner_steps"], long_student_images=2048, long_student_updates=5000,
        marked_minus_clean_target_score=float(delta[:, 0].mean()),
        marked_minus_clean_wrong_score=float(delta[:, 1].mean()),
        marked_minus_clean_contrast=float((delta[:, 0]-delta[:, 1]).mean()),
        marked_minus_clean_common_component=float(((delta[:, 0]+delta[:, 1])/2).mean()),
        note_common_component="Algebraic common score component, not an isolated measurement of presence-head contribution",
        clean_calibration_target_std=float(calibration[:, 0].std()),
        clean_calibration_contrast_std=float(contrast.std()),
        old_score_to_contrast_scale_ratio=float(calibration[:, 0].std()/contrast.std()),
        channel_rows=channels["rows"],
        training_ranges={k: dict(min=float(min(r[k] for r in history)),
                                 median=float(np.median([r[k] for r in history])),
                                 max=float(max(r[k] for r in history)))
                         for k in ("paired_effect", "key_specificity", "wm", "gradient_norm")},
        established=[
            "Single-key teacher TPR improved but 5000-step ordinary-student TPR did not.",
            "Selected proxy checkpoint still has negative key specificity.",
            "Training repeats one image per 8-update unroll; it does not expose 8 distinct images.",
            "Old loss rewards absolute target score as well as key contrast.",
            "Old specificity uses raw-score standard deviation rather than contrast standard deviation.",
            "Shared/amplitude ablations changed carrier geometry and failed frozen-reader preservation; no student was tested."],
        not_established=[
            "VAE is the main bottleneck: measured round-trip contrast survives.",
            "One-step x0 reconstruction predicts inheritance: its marked input can carry signal through an identity term.",
            "Any one code/design issue explains all failure: causal ablations are still required.",
            "AUC about .74 proves practical 1% FPR detection or independent model attribution."])
    (folder / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("training_ranges", "channel_rows", "validation_proxy")}, indent=2))


if __name__ == "__main__":
    main()

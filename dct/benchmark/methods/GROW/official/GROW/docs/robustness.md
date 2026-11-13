# Robustness evaluation

GROW ships with an off-the-shelf attack suite in
[`grow/attacks.py`](../grow/attacks.py), covering the attacks reported in
the paper:

<!--
  ASSET: assets/attacks_panel.png
  Suggested content: Figure 3 of the paper — 8 panels showing
  (a) clean, (b) rotation, (c) JPEG, (d) crop&scale, (e) blur,
  (f) noise, (g) brightness, (h) adversarial purification.
-->
<p align="center"><img src="../assets/attacks_panel.png" alt="Attack gallery" width="90%"/></p>

| Attack        | Default parameters                  | Extraction helper used                               |
|---------------|-------------------------------------|-------------------------------------------------------|
| Clean         | —                                   | `GrowWatermarker.extract`                             |
| Rotation 75°  | `r_degree=75`                       | `GrowWatermarker.extract_robust_rotation`             |
| JPEG q=25     | `jpeg_ratio=25`                     | `GrowWatermarker.extract`                             |
| Crop & Scale  | `crop_scale=0.75, crop_ratio=1`     | `GrowWatermarker.extract_robust_crop_scale`           |
| Gaussian Blur | `gaussian_blur_r=4`                 | `GrowWatermarker.extract`                             |
| Gaussian Noise| `gaussian_std=0.05`                 | `GrowWatermarker.extract`                             |
| Brightness ×6 | `brightness_factor=6`               | `GrowWatermarker.extract`                             |

## Running the suite manually

```python
from PIL import Image
from grow import GrowConfig, GrowWatermarker, AttackSuite

cfg = GrowConfig(message="OKOK", secret_key="watermark")
wm = GrowWatermarker(cfg)

image = Image.open("watermarked.png")

def extractor(img, name):
    if name == "Rot 75":
        return wm.extract_robust_rotation(img)
    if name == "C&S 75":
        return wm.extract_robust_crop_scale(img)
    return wm.extract(img)

suite = AttackSuite(
    extractor=extractor,
    expected_message=cfg.message,
    output_dir="outputs/attacks",
)
results, accuracy = suite.run(image, seed=cfg.seed)
AttackSuite.pretty_print(results, accuracy)
```

## Adding a new attack

Append a new entry to `DEFAULT_ATTACKS` in `grow/attacks.py`, or pass
your own `attacks={"my_attack": {...}}` dictionary to `AttackSuite`. The
parameter names follow the signature of `image_distortion`.

## Adversarial purification (optional)

The adversarial row in Table 1 uses Diff-Pure with `s=0.5`. Because the
attacker is an external library, it is not bundled in this repository;
follow the original [Diff-Pure instructions](https://github.com/NVlabs/DiffPure)
and feed the purified image through `GrowWatermarker.extract` as usual.

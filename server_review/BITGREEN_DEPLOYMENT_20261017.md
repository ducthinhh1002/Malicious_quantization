# BitGreen pilot deployment ? 17/10/2026

- Server: vm_4, H200; tmux session `bitgreen`.
- Isolated code: `/root/logs/bitgreen_release_8a46d09`.
- Archive SHA256: `8a46d09a2f9c973e178202f2840b50b4c314c94d3ff685ae07f786edbeeeebbb`.
- Run: `/root/logs/results/bitgreen_20261017_064252_359192`.
- Heavy files: `/root/logs/runtime/bitgreen_artifacts/bitgreen_20261017_064252_359192`.
- Log: `/root/logs/bitgreen.log`; exit code: `/root/logs/bitgreen.exit` once finished.
- No new SERUM experiment launched. Existing centered-teacher job left running.

## Validation

Five unit tests passed locally and on server: exact bit-pair counts/no cross-token boundary, useful gradient, detached reference/weak-scale penalty, identical ordinary student commands, and clean-adjusted reporting with frozen threshold.

Two owner CUDA updates passed with real pretrained tokenizer, all encoder/quantizer parameters verified against SHA-pinned checkpoint; official hard BSQ bits equal residual signs. Gradients finite/nonzero. Observed peak PyTorch allocated VRAM 13.37 GiB; this is a smoke measurement, not a whole-run upper bound. Configurable per-process PyTorch cap 40 GiB.

Fixed BF16 autocast caching across no-grad/grad rollout segments; disabled the cache in the owner gradient rollout to retain actual parameter gradients.

Merge prediction maximum absolute error 4.827976226806641e-06; no adapter modules or special inference inputs remain. Saved/reloaded UNet weights exactly equal.

Tiny integration completed with exit 0: native merged teacher generation, clean/fixed/wm ordinary students (2 updates each), validation-MSE selection, final image generation, frozen bit scoring. The two marked integration branches intentionally used the same two-step smoke owner to test plumbing; these are NOT experimental findings. Main pilot trains two distinct owner arms from the same clean initialization.

## Frozen pilot

Plain/robust owners: 250 updates each; same BSQ carrier and rank-16 owner capacity. Robust adds frozen reconstruction/noising channels. Three ordinary r4 student branches, 500 training images, 2500 updates, with checkpoint selection by independent validation MSE only. No detector, key, owner weights or weighting in student training.

Calibration: 1000 clean SD1.4 images, empirical FPR target 1%; final test 256. Teacher design gate uses separate 128-image split. If neither teacher passes, preserve diagnostics and skip expensive student training (not evidence that inheritance is impossible).

Main run had entered `train_plain` at last verification. No scientific claim of strong teacher, transfer, newness or secret-key attribution yet.

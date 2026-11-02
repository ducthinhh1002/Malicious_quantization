# Baseline diagnostic decision

C10 seed 0 completed 20,000 LoRA steps. Checkpoint 20,000 was selected using minimum unseen denoising MSE (0.1460962672) on 1,000 watermarked validation images. At the frozen threshold 0.04637297987937927, 162 of 5,000 student outputs were positive (3.24%). Teacher watermarked image detection was 99.98%.

The teacher calibration distribution cannot establish the false-positive rate on SD1.4 student outputs. Do not interpret C10's 3.24% as established radioactive transfer before measuring student controls.

Launched on gpu-node-2 in separate tmux sessions:

- `quant-sd14-control`, GPU 0: generate 5,000 outputs from unadapted SD1.4, then score with the frozen detector and threshold.
- `quant-c00`, GPU 1: train on 5,000 paired clean teacher images using the same rank 4, learning rate 1e-5, batch 8, seed 0 and 20,000-step budget as C10. Select checkpoint using minimum MSE on the 1,000 clean validation images, generate 5,000 student outputs, and score at the same threshold.

Both evaluations use the student_eval prompt split, sampling seeds 50000 + index, 50 PNDM steps and guidance 7.5, matching C10. No threshold recalibration or checkpoint selection by detector scores is permitted. The summary script reports control detection rates, confidence intervals and paired discordance tests against C10.

The next decision depends on these controls. A weak C10 baseline must be resolved before launching the laundering pilot or interpreting reduction as attack success. The C00 clean control is itself a required core experiment and is reusable in the final comparison.

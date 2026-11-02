# Authorized continuation after Stage 2

Scope: complete the seed-0 core experiment (items 8–13), while preserving the finding that the C10 baseline is weak and has not reproduced the published result. This queue does not create a C10 pass gate.

- Reuse completed C00_seed0 and C10_seed0, including validation selection and all 5,000 evaluation outputs.
- Use pilot-selected strengths mild 0.25 and strong 1.00. Reuse the matching first 500 pilot training outputs at each strength, checking source identity and seeds.
- Create C01 and C11 datasets for both strengths: 5,000 training and 1,000 validation images each, preserving prompt pairing. Decode and audit every resulting image and its provenance before training.
- Run four new SD1.4 LoRA students, each with rank 4, LR 1e-5, batch 8, 20,000 steps and seed 0. GPU 0 handles clean/C01, GPU 1 handles watermarked/C11.
- Select each checkpoint using minimum denoising MSE on its corresponding 1,000 validation images; generate 5,000 disjoint student_eval outputs; audit using the existing frozen SERUM threshold.
- Generate a shared 5,000-image clean teacher reference on student_eval prompts, with separate sampling seed 60000. Compute prompt CLIPScore with ViT-B/32 and both KID/FID for all six students. These distribution metrics measure similarity to the clean teacher distribution, not COCO FID.
- Save per-image CLIP scores, 95% bootstrap intervals, prompt-paired contrasts and clean-adjusted watermark effects, a combined results table, and a watermark–utility figure.
- Write a follow-up decision for items 14–15. Three-seed replication requires a credible baseline and a useful first-seed attack result. Mechanism/defense runs require stable success across seeds. Neither is automatically claimed satisfied by this exploratory run; the current queue ends with the seed-0 report.

Tracking: `/root/workspace/serum/metrics/core_queue_status.json`, tmux `quant-core-seed0`, main log `/root/workspace/serum/logs/core_queue.log`, per-condition logs `/root/workspace/serum/logs/core_C01_step*_seed0.log` and `/root/workspace/serum/logs/core_C11_step*_seed0.log`.

Final outputs: `/root/workspace/serum/metrics/core_seed0_summary.json`, `core_seed0_frontier.png`, `core_seed0_frontier.svg`, and `core_followup_decision.json`.

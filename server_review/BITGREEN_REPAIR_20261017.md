# Supervision repair diagnostics

Immutable code snapshot: /root/logs/bitgreen_supervision_1017
SSH alias: vm4. Both experiments use source run /root/logs/results/bitgreen_transport_1017 only for assets, prompts and frozen threshold; no final-test feedback in training.

- tmux bitgreen_repair_overfit; /root/logs/results/bitgreen_repair_overfit_1017; 4 training and 4 held-out owner-train probes; 300 FP32 updates; native rollout every 50 updates; merged FP16 audit at end.
- tmux bitgreen_repair_shared; /root/logs/results/bitgreen_repair_shared_1017; 16 training and 8 held-out owner-train probes; 150 shared-delta updates. Carrier-only experiment; does not train native teacher or student.

Launchers: /root/logs/start_bitgreen_repair_overfit.sh and /root/logs/start_bitgreen_repair_shared.sh.
Logs: /root/logs/bitgreen_repair_overfit.log and /root/logs/bitgreen_repair_shared.log. Matching .exit files hold process exit status.

22 tests passed. First real overfit optimizer updates completed: step2 loss 0.99837, gradient norm 0.47425, peak allocated 9.894 GiB. All initial fixed probes have loss=1, latent-target relative MSE=1 and PSNR=120: clean initial model exactly reproduces fresh FP32 references. This is a mechanics check, not evidence of a successful watermark.

Heavy artifacts: BITMARK_RUNTIME_ROOT/bitgreen_artifacts/<run>/owner. JSON diagnostics in result folders.
Research and interpretation: survey/BitGreen_Learnability_Repair_VI.md.

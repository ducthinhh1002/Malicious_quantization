# Latent transport pilot deployment

Server vm4; tmux `bitgreen_transport`.
Immutable code: `/root/logs/bitgreen_transport_final_1017`.
Run: `/root/logs/results/bitgreen_transport_1017`.
Launcher: `/root/logs/start_bitgreen_transport.sh`.
Log: `/root/logs/bitgreen_transport.log`; exit file: `/root/logs/bitgreen_transport.exit`.

64 owner-train prompts per arm, maximum 32 latent steps each (early stop on simultaneous quality/detection success), 2000 owner optimizer updates with 4 transition samples/update. LR 5e-5. SD1.4. Allocator cap 24 GiB. Ordinary student unchanged; only runs if teacher pilot gate passes.

Validation: 20 unit tests pass, including exact DDIM transition equivalence across all 20 timesteps, CFG supervision, frozen references and equality of student commands. Robust CUDA smoke passes 2 updates / 8 transitions with 2 successful targets; peak allocated 9.904 GiB after VAE checkpointing. Initial no-checkpoint smoke exceeded 24 GiB and was replaced; original failed log retained.

Smoke targets are not native-teacher results. Native generation and ordinary-LoRA inheritance must be measured by the full run. Existing K4 runs completed with no meaningful teacher signal; summaries saved in server_review/bitgreen_stablegrad_final.

Research and formulas: survey/BitGreen_Latent_Transport_VI.md.

# Structure32 teacher → full UNet student

Server `.21`: `/var/lib/structure32_student_fullft`, GPU1, process `VLLM::EngineCore`.
Reuses the frozen5000 Structure32 refine600 teacher images, prompts, carrier/key
and pinned SD1.4 base. Fresh full UNet fine-tuning; no LoRA. VAE and text encoder
are frozen. Ordinary diffusion denoising loss, without watermark supervision.

12500steps, batch4, deterministic seed0 epoch sampling. LR1e-5 through7500,
cosine to2e-6 at12500. BF16 for full-model training; original LoRA used FP16
and LR1e-4. This is a full-FT recipe comparison, not an experiment where only
parameter scope changes. No EMA or data augmentation. Inference uses the
same base sampler/steps/guidance and prompt/seed panel through the existing harness.

Technical4-step full-model smoke runs first. Save full Accelerate checkpoints
every1250steps (UNet, optimizer, scheduler, RNG), retaining the latest2;
also retain inference snapshot12500 and final model. Resume from latest saved
checkpoint. VAE/text freeze and absence of LoRA are asserted in the trainer.

Evaluation: fixed12500,1000test images; bit/key metrics and key acceptance
threshold>=24of32, groups1/32/256 with100random partitions for aggregate.
Theoretical FPR0.350018% assumes independent fair random bit matches; it is not
an empirical FPR estimate. Also records LPIPS/FID/KID/CLIP and strict32/32 recovery.

Status/logs under `deployment`, models and evaluation under `run`. No source
teacher weights or earlier experimental results are overwritten.

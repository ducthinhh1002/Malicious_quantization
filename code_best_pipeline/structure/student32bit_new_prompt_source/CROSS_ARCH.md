# Additional queued students on disjoint COCO prompts

Models pinned to exact revisions:

* Sana0.6B512: `Efficient-Large-Model/Sana_600M_512px_diffusers`, revision `83d7a190bfd1fd070570a793d2dab5c7a3231b9d`.
* PixArt-Alpha512: `PixArt-alpha/PixArt-XL-2-512x512`, revision `50f702106901db6d0f8b67eb88e814c56ded2692`.

Both jobs wait for completion of the existing SD1.4 newprompt run, then for the shared new COCO image generation. Downloads and prompt preparation may occur while queued. `coco_disjoint.py` selects 5000 train captions from COCO train2014 and 1000 validation plus 1000 test captions from val2014. All original teacher split image IDs, normalized texts and effective 77-token sequences for both original tokenizers are excluded; the new splits are mutually disjoint by these rules. The unchanged teacher refine600 generates one common image set for both students, with clean heldout references. No manipulation. These two models use neither Gustavosta nor the teacher's original training prompts. This is an in-dataset generalization experiment, not a claim of out-of-distribution prompts. Carrier/key and the owner's SD2.1-based reader stay frozen. `coco_prompt_audit.json` records counts, hashes, exclusions and seeds.

Shared budget/configuration: fresh student weights; LoRA rank32/alpha32; batch4;seed0;20000attempted steps; AdamW1e-4 through7500, cosine to2e-5 at12500, constant afterward; maxgradientnorm1; BF16 transformer/text/latents with FP32 adapters+AdamW+VAE. Only transformer adapters train. Text/latents are cached without training: deterministic embeddings; Sana deterministic DC-AE latent; PixArt posterior mean+std cache resampled each training batch. No watermark detector in training. Effective token limit77 and no prompt rewriting/cleaning, to keep prompt content processing explicit.

Architectural differences are necessary and reported, so this is not a bitwise same-trainer reproduction:

* SD1.4uses UNet attention/conv adapters. These DiTs use LoRA on all transformer Linear and ungrouped Conv2d modules. Exact target names and parameter counts are saved. Frozen VAEs and text encoders differ by architecture.
* Sana uses native flow-matching target `noise-latent`, uniform timestep density and FlowMatchEulerDiscreteScheduler loaded per the Diffusersv0.35.1 Sana training reference (its default shift1); inference retains its checkpoint DPM flow scheduler.
* PixArt uses its native DDPM epsilon MSE with uniform1000timesteps; predicted variance channels are discarded following the official PixArt LoRA trainer. Native inference scheduler is preserved.

Inference comparison uses512square,50steps,CFG7.5,seed500000+i,1000sameheldout COCO prompts; resolution binning disabled. This matches SD1.4's requested numerical inference settings while keeping model-specific solvers. These are controlled settings, not a claim that each model is at its optimal published inference configuration. Generate both unadapted and trained student images for pairedLPIPS and original-carrier BA/exactFK/acceptance>=24of32. Group1/4/8/32/64/256;100randompartitions, not independent groups. Save CLIP/FID/KID against the exact same teacher-clean reference images.

Each model must pass a4step real-gradient/parameter-update smoke before full training. Full checkpoints every1250save adapter parameters, AdamW, scaler, data order/offset, independent data sampler RNG and Python/NumPy/Torch CPU/CUDA RNG. Scheduler is an absolute-step function whose code hash is part of the checkpoint identity. Adapter-only final reload is checked tensor-for-tensor before generation. Loss and applied/skipped AMP update counts are logged separately. No scientific/quality early-stopping gate. Technical failures stop and remain visible in status.json.

Primary sources inspected:

* https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/examples/dreambooth/train_dreambooth_lora_sana.py
* https://github.com/PixArt-alpha/PixArt-alpha/blob/master/train_scripts/train_pixart_lora_hf.py
* https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers
* https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512

These jobs are image-based student learning experiments. Neither transformer is an autoregressive image model; changing student architecture does not itself turn the existing image-supervision experiment into autoregressive distillation.

Technical fixes after failed initial smokes: Sana FP16 produced a nonfinite activation at the first attention output projection despite finite inputs and weights; the same forward in BF16 was finite. Both DiTs now use BF16 with FP32 adapter optimization. PixArtTransformer2DModel0.35.1 lacks add_adapter/load_adapter, so use PEFT inject_adapter_in_model and set_peft_model_state_dict, with exact adapter reload checks. PixArt pipeline0.35.1 also does not cast final latents to VAE dtype: request native denoised latents, then explicitly decode with the FP32 VAE and native image processor. The failed attempts and identities are retained; no full training checkpoint existed. Frozen feature cache bytes, prompts, images, teacher, carrier/key remain unchanged. The experiment/cache identity links were explicitly migrated before training rather than silently reusing an old code hash.

Latest user correction: source is `/var/lib/structure32_newprompts_rank32/coco_disjoint_source`; the teacher checkpoint is shared read-only from `/var/lib/structure32_student12500/runs/structure32`. The active SD1.4 Gustavosta job is unchanged.

Additional matched SD1.4 control: `sd14_coco_queue.py` queues a fresh SD1.4 rank32 attention/conv student in `sd14_coco_disjoint/student`. It links the identical COCO source, generates no replacement training images, and uses batch4/seed0/20000 steps with full state every1250. It runs on GPU0 after Sana releases the device, including after a technical failure, so a failed smoke does not permanently block the control. The existing Gustavosta trial is retained separately. The supervisor first scores the shared teacher and unadapted SD1.4 outputs, then initializes and runs the existing SD1.4 campaign with fixed final checkpoint selection.

"""Generate held-out native outputs from the verified final refined adapter."""
import bootstrap
from bootstrap import ROOT, read, write, sha, now
from pathlib import Path
import torch
from semantic_checkpoint import verified_state
from semantic_teacher import generate_set, numerical_policy
from student import load_adapter

run = ROOT / 'run'
checkpoint = run / 'teacher_fp/refined/checkpoint-1200'
state = verified_state(checkpoint)
assert state['step'] == 1200
del state
numerical_policy()
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
assets = read(run / 'assets.json')
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(assets['teacher_clean'], torch_dtype=torch.float32, safety_checker=None, local_files_only=True).to('cuda')
pipe.set_progress_bar_config(disable=True)
load_adapter(pipe.unet, checkpoint)
pipe.unet.eval()
generator = torch.Generator('cuda').manual_seed(12345)
z = torch.randn((1,4,64,64), generator=generator, device='cuda')
tokens = pipe.tokenizer(['a photograph of a tree'], padding='max_length', truncation=True, max_length=77, return_tensors='pt').input_ids.cuda()
with torch.no_grad():
    hidden = pipe.text_encoder(tokens)[0]
    before = pipe.unet(z, torch.tensor(123, device='cuda'), hidden).sample
pipe.fuse_lora(safe_fusing=True)
pipe.unload_lora_weights()
from peft.tuners.tuners_utils import BaseTunerLayer
assert not any(isinstance(m, BaseTunerLayer) for m in pipe.unet.modules())
pipe.unet.eval()
with torch.no_grad():
    after = pipe.unet(z, torch.tensor(123, device='cuda'), hidden).sample
    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)
write(run / 'generation_merge_check.json', dict(passed=True, max_absolute_error=float((after-before).abs().max()), adapter_sha256=sha(checkpoint/'adapter_model.safetensors')))
# Keep FP32 to match the differentiable native training forward.
write(run / 'generation_status.json', dict(state='running', updated=now(),
    checkpoint=str(checkpoint), adapter_sha256=sha(checkpoint/'adapter_model.safetensors')))
generate_set(run, assets['teacher_clean'], 'test', 'teacher_jpeg600_test', pipe=pipe)
write(run / 'generation_status.json', dict(state='completed', n=1000, updated=now(),
    checkpoint=str(checkpoint), adapter_sha256=sha(checkpoint/'adapter_model.safetensors')))

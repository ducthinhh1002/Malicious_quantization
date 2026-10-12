"""Optional real CPU integration check (needs dependencies, but no model downloads).

Builds a tiny random SD locally, trains both *official* trainers for two steps,
checks learned weights, reloads LoRA and full checkpoints, then runs inference.
It does NOT test Infinity CUDA kernels or claim scientific radioactivity results.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import HERE, verify_vendor


def main():
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    verify_vendor()
    sys.path.insert(0, str(HERE / 'vendor/bitmark'))
    from extended_watermark_processor import WatermarkDetector, WatermarkLookupProcessor
    detector = WatermarkDetector(vocab=[0, 1], gamma=.5, delta=2., device='cpu',
                                 ignore_repeated_ngrams=False, green_list='01,10')
    with contextlib.redirect_stdout(io.StringIO()):
        alternating = detector.detect(tokenized_text=torch.tensor([0, 1]*64))
        constant = detector.detect(tokenized_text=torch.zeros(128, dtype=torch.long))
    assert alternating['green_fraction'] == 1 and constant['green_fraction'] == 0
    assert alternating['z_score'] > 0 and constant['z_score'] < 0
    processor = WatermarkLookupProcessor(vocab=[0, 1], device='cpu', delta=2., green_list={'01','10'})
    logits = torch.zeros(1, 2, 2)
    biased = processor(torch.tensor([[0, 1]]), logits)
    assert biased[0, 0, 1] > biased[0, 0, 0] and biased[0, 1, 0] > biased[0, 1, 1]
    print('Official detector and logit bias CPU checks passed', flush=True)
    with tempfile.TemporaryDirectory(prefix='bitmark-smoke-') as tmp:
        root = Path(tmp)
        vocab = root/'vocab.json'
        vocab.write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2, 'test</w>': 3}))
        merges = root/'merges.txt'
        merges.write_text('#version: 0.2\n')
        tokenizer = CLIPTokenizer(str(vocab), str(merges), model_max_length=77)
        text = CLIPTextModel(CLIPTextConfig(vocab_size=4, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
                             bos_token_id=0, eos_token_id=1, pad_token_id=1))
        vae = AutoencoderKL(block_out_channels=(16, 32), in_channels=3, out_channels=3,
             down_block_types=('DownEncoderBlock2D',)*2, up_block_types=('UpDecoderBlock2D',)*2,
             latent_channels=4, norm_num_groups=8, sample_size=32)
        unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
             down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
             up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
             block_out_channels=(16, 32), layers_per_block=1, norm_num_groups=8,
             cross_attention_dim=32, attention_head_dim=4)
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
             scheduler=DDIMScheduler(num_train_timesteps=100, steps_offset=1, clip_sample=False), safety_checker=None,
             feature_extractor=None, requires_safety_checker=False)
        base = root/'base'
        pipe.save_pretrained(base)
        data = root/'data'
        data.mkdir()
        (data/'generation.json').write_text('{"test": true}')
        (data/'image_hashes.json').write_text('[]')
        rng = np.random.default_rng(0)
        with (data/'metadata.jsonl').open('w') as f:
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(data/f'{i}.png')
                f.write(json.dumps({'file_name': f'{i}.png', 'text': 'a test'})+'\n')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
        for kind in ('lora', 'full'):
            name = 'train_text_to_image_lora.py' if kind == 'lora' else 'train_text_to_image.py'
            out = root/kind
            cmd = [sys.executable, str(HERE/'vendor/diffusers'/name),
                   '--pretrained_model_name_or_path', str(base), '--train_data_dir', str(data),
                   '--output_dir', str(out), '--resolution', '32', '--center_crop',
                   '--train_batch_size', '1', '--max_train_steps', '2', '--gradient_checkpointing',
                   '--learning_rate', '0.0001', '--mixed_precision', 'no', '--report_to', 'none',
                   '--checkpointing_steps', '1000000', '--seed', '0']
            if kind == 'lora':
                cmd += ['--rank', '4']
            subprocess.run(cmd, check=True, env=env)
            model = StableDiffusionPipeline.from_pretrained(base if kind == 'lora' else out)
            if kind == 'lora':
                tensors = load_file(str(out/'pytorch_lora_weights.safetensors'))
                assert any(t.abs().max().item() > 0 for k, t in tensors.items() if 'lora.up' in k)
                model.load_lora_weights(out)
            else:
                assert any(not torch.equal(p, q) for p, q in zip(pipe.unet.parameters(), model.unet.parameters()))
            with torch.inference_mode():
                image = model('a test', height=32, width=32, num_inference_steps=2).images[0]
            assert image.size == (32, 32)
            print(f'{kind}: two real optimizer steps, saved/reloaded and inferred successfully', flush=True)


if __name__ == '__main__':
    main()

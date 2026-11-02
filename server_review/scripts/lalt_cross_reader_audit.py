"""Read-only 2x2 carrier/reader compatibility audit, using cached validation latents."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--step2', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.code))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from new_watermark.learning_channel.carrier import key_bits
from new_watermark.learning_channel.objective import rgb8, separation_metrics
from new_watermark.learning_channel.reader import key_scores
from new_watermark.learning_channel.runtime import read_json, storage, gpu_setup, write_json, csv_rows
from new_watermark.learning_channel.train import load_state
from new_watermark.learning_channel.sampling import decode

args.output.mkdir(parents=True, exist_ok=False)
args.artifacts.mkdir(parents=True, exist_ok=False)
config = read_json(args.run / 'config.json')
raw = (storage(args.run)['artifacts'] / 'teacher_last.pt').read_bytes()
state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
if state['step'] != 3000 or state['config'] != config:
    raise RuntimeError('Step 3000 is no longer the last checkpoint; do not substitute another step')
path3 = args.artifacts / 'step3000.pt'
with path3.open('xb') as stream:
    stream.write(raw)
del raw, state
write_json(args.output / 'status.json', dict(status='running'))
gpu_setup(dict(config, vram_limit_gb=16))
models = {2000: load_state(args.step2, None, config), 3000: load_state(path3, None, config)}
for carrier, reader in models.values():
    carrier.eval().requires_grad_(False); reader.eval().requires_grad_(False)
vae = AutoencoderKL.from_pretrained(read_json(args.run/'assets.json')['base'], subfolder='vae',
    torch_dtype=torch.float32, local_files_only=True).cuda().eval().requires_grad_(False)
keys = config['validation_keys']; codebook = key_bits(keys, device='cuda')
rows = read_json(args.run/'prompts.json')['teacher_validation']
cache = storage(args.run)['artifacts'] / 'prefix_cache'
metrics = list(csv.DictReader((args.run/'teacher_validation.csv').open()))
cuts = {(int(r['step']), int(r['key'])):float(r['threshold']) for r in metrics if int(r['step']) in models}
write_json(args.output / 'protocol.json', dict(carrier_steps=[2000,3000],reader_steps=[2000,3000],
    n_prompts=len(rows),keys=keys,thresholds='original model-specific clean calibration thresholds',
    no_training=True,final_test_used=False,source_checkpoint3000_sha256=hashlib.sha256(path3.read_bytes()).hexdigest(),
    caveat='Crossed pairs measure compatibility; they are not independently retrained causal ablations'))
details = []
with torch.no_grad():
    for j,key in enumerate(keys):
        for start in range(0,len(rows),2):
            batch=rows[start:start+2]
            data=[load_file(str(cache/(str(r['image_id'])+'.safetensors')),device='cpu') for r in batch]
            latent=torch.cat([r['latent'] for r in data]).cuda()
            clean=rgb8(torch.cat([r['clean'] for r in data]).cuda())
            codes=codebook[j:j+1].expand(len(batch),-1)
            for cs,(carrier,_) in models.items():
                marked=rgb8(decode(vae,carrier(latent,codes)))
                for rs,(_,reader) in models.items():
                    out=reader(marked);base=reader(clean)
                    scores=key_scores(out,codebook)[:,j]
                    clean_scores=key_scores(base,codebook)[:,j]
                    signed_logits=(codes*2-1)*out[1]
                    for i,row in enumerate(batch):
                        details.append(dict(carrier_step=cs,reader_step=rs,key=key,image_id=row['image_id'],
                            score=float(scores[i]),clean_score=float(clean_scores[i]),
                            bit_accuracy=float(((out[1][i]>0)==codes[i].bool()).float().mean()),
                            presence_probability=float(out[0][i].sigmoid()),
                            bit_nll=float(F.softplus(-signed_logits[i]).mean()),
                            signed_bit_margin=float(signed_logits[i].mean()),
                            confidently_wrong_bit_fraction=float((signed_logits[i]<-2).float().mean())))
            if (start+len(batch))%32==0:
                print(f'Cross audit key={key}: {start+len(batch)}/{len(rows)}',flush=True)
summary=[]
for cs in models:
    for rs in models:
        for key in keys:
            group=[r for r in details if (r['carrier_step'],r['reader_step'],r['key'])==(cs,rs,key)]
            a=np.array([r['score'] for r in group]);b=np.array([r['clean_score'] for r in group]);cut=cuts[rs,key]
            summary.append(dict(carrier_step=cs,reader_step=rs,key=key,n=len(group),threshold=cut,
                tpr=float((a>cut).mean()),clean_fpr=float((b>cut).mean()),score_median=float(np.median(a)),
                **{name:float(np.mean([r[name] for r in group])) for name in ('bit_accuracy','presence_probability',
                  'bit_nll','signed_bit_margin','confidently_wrong_bit_fraction')},**separation_metrics(a,b)))
csv_rows(args.output/'per_image.csv',details)
csv_rows(args.output/'summary.csv',summary)
write_json(args.output/'summary.json',dict(rows=summary,not_a_transfer_evaluation=True))
write_json(args.output/'status.json',dict(status='complete'))
print(json.dumps(summary,indent=2),flush=True)

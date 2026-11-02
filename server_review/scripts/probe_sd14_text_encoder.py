"""Compare identical SD1.4 text inputs across the two existing environments."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import CLIPTokenizer, CLIPTextModel

p=argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a=p.parse_args()
torch.set_num_threads(4)
model='CompVis/stable-diffusion-v1-4'
revision='133a221b8aa7292a167afc5127cb63fb5005638b'
prompts=Path('/root/workspace/serum/prompts/adapt_train.txt').read_text().splitlines()[:8]+['']
tokenizer=CLIPTokenizer.from_pretrained(model,subfolder='tokenizer',revision=revision)
encoder=CLIPTextModel.from_pretrained(model,subfolder='text_encoder',revision=revision).eval()
tokens=tokenizer(prompts,padding='max_length',max_length=77,truncation=True,return_tensors='pt')
with torch.no_grad():
    embeddings=encoder(tokens.input_ids)[0].numpy()
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez(a.out,input_ids=tokens.input_ids.numpy(),embeddings=embeddings)
Path(str(a.out)+'.json').write_text(json.dumps({'transformers':transformers.__version__, 'tokenizer':type(tokenizer).__name__, 'revision':revision},indent=2))
print(a.out,flush=True)

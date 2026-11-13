"""Fixed held-out prompts monitor parameter learning during refinement."""
import torch
from diffusers import StableDiffusionPipeline,PNDMScheduler
from bootstrap import read,write
from jpeg_channel import jpeg_ste

@torch.no_grad()
def monitor(model,step):
    rows=read(model.run/'prompts.json')['validation'][:16]
    was_training=model.unet.training
    model.unet.eval()
    pipe=StableDiffusionPipeline(vae=model.ext.vae,text_encoder=model.ext.text_encoder,
        tokenizer=model.ext.tokenizer,unet=model.unet,
        scheduler=PNDMScheduler.from_pretrained(model.assets['teacher_clean'],subfolder='scheduler',local_files_only=True),
        safety_checker=None,feature_extractor=None,requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    scores={'original':[],'jpeg25':[]}
    for start in range(0,len(rows),2):
        batch=rows[start:start+2]
        generators=[torch.Generator('cuda').manual_seed(model.c['seed']*1000000+r['image_id']) for r in batch]
        latent=pipe([r['text'] for r in batch],generator=generators,num_inference_steps=model.c['inference_steps'],guidance_scale=7.5,output_type='latent').images
        rgb=pipe.vae.decode(latent/pipe.vae.config.scaling_factor).sample.clamp(-1,1)
        rgb=((rgb+1)*127.5).round()/127.5-1
        for label in scores:
            view=rgb if label=='original' else jpeg_ste(rgb,25)
            q=model.carrier.coordinates(model.ext.encode_image(view,[r['text'] for r in batch]))
            scores[label].extend(q.cpu().tolist())
    result={}
    for label,values in scores.items():
        q=torch.tensor(values);matched=((q>0)==(model.carrier.signs.cpu()>0)).sum(1)
        result[label]=dict(bit_accuracy=float(matched.float().mean()/32),acceptance_ge24=float((matched>=24).float().mean()),coordinates=values)
    write(model.run/'validation'/f'step_{step:04d}.json',dict(step=step,n=16,selection='monitor only; final checkpoint fixed',results=result))
    model.unet.train(was_training)
    print('VALIDATION',step,{k:v['bit_accuracy'] for k,v in result.items()},flush=True)

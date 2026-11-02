"""Fixed held-out denoising draws; preserve training RNG; validation-only stopping."""
from pathlib import Path
import json,time,torch,numpy as np
from bootstrap import ROOT,read_json,write_json
RUN=ROOT/'run'
POLICY={'interval':500,'patience':4,'min_delta':1e-4,'min_steps':2500,'validation_n':1000,'draws':2,'seed':314159}
def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
class ValidationController:
    def __init__(self):self.cache=None
    @torch.no_grad()
    def __call__(self,unet,vae,text_encoder,tokenizer,scheduler,accelerator,step,output):
        device=accelerator.device;was_training=unet.training
        try:
            with torch.random.fork_rng(devices=[device.index or 0]):
                unet.eval()
                if self.cache is None:
                    path=RUN/'validation_cache_fixed.pt'
                    if path.exists():self.cache=torch.load(path,map_location='cpu',weights_only=True)
                    else:
                        from PIL import Image
                        rows=read_json(RUN/'prompts.json')['validation'];assert len(rows)==1000
                        z=[];hidden=[];g=torch.Generator(device=device).manual_seed(POLICY['seed'])
                        for start in range(0,len(rows),4):
                            images=[]
                            for i in range(start,min(start+4,len(rows))):
                                with Image.open(RUN/'images/teacher_wm_validation'/f'sample_{i:06d}.png') as im:
                                    assert im.size==(512,512)
                                    images.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1)
                            x=torch.stack(images).to(device=device,dtype=vae.dtype)
                            z.append((vae.encode(x).latent_dist.sample(generator=g)*vae.config.scaling_factor).float().cpu())
                            ids=tokenizer([r['text'] for r in rows[start:start+len(x)]],padding='max_length',truncation=True,max_length=tokenizer.model_max_length,return_tensors='pt').input_ids.to(device)
                            hidden.append(text_encoder(ids)[0].cpu())
                        self.cache={'latents':torch.cat(z),'hidden':torch.cat(hidden)}
                        tmp=path.with_suffix('.partial');torch.save(self.cache,tmp);tmp.replace(path)
                cache=self.cache;gen=torch.Generator('cpu').manual_seed(POLICY['seed']+1);total=count=0
                for draw in range(POLICY['draws']):
                    for start in range(0,1000,4):
                        z=cache['latents'][start:start+4].to(device)
                        noise=torch.randn(z.shape,generator=gen).to(device)
                        t=torch.randint(0,scheduler.config.num_train_timesteps,(len(z),),generator=gen).to(device)
                        x=scheduler.add_noise(z,noise,t)
                        with accelerator.autocast():pred=unet(x,t,encoder_hidden_states=cache['hidden'][start:start+4].to(device)).sample
                        target=noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(z,noise,t)
                        total+=float((pred.float()-target.float()).square().mean((1,2,3)).sum());count+=len(z)
                loss=total/count
        finally:unet.train(was_training)
        statepath=RUN/'early_stopping.json'
        state=read_json(statepath) if statepath.exists() else {'best_loss':None,'best_step':None,'bad_checks':0}
        best=state['best_loss'];improved=best is None or loss<best-POLICY['min_delta']
        checkpoint=Path(output)/f'validation-state-{step}'
        accelerator.save_state(str(checkpoint))
        write_json(checkpoint/'VALIDATION_COMPLETE.json',{'step':step,'validation_loss':loss})
        if improved:
            state.update(best_loss=loss,best_step=step,best_checkpoint=str(checkpoint),bad_checks=0)
        elif step>=POLICY['min_steps']:state['bad_checks']+=1
        stopped=step>=POLICY['min_steps'] and state['bad_checks']>=POLICY['patience']
        row={'step':step,'validation_loss':loss,'improved':improved,'bad_checks':state['bad_checks'],'best_step':state['best_step'],'best_loss':state['best_loss'],'early_stop':stopped,'time':time.time()}
        append(RUN/'metrics/validation_loss.jsonl',row)
        state.update(last_step=step,last_loss=loss,stopped_early=stopped,policy=POLICY)
        write_json(statepath,state);print('VALIDATION_CONTROL',json.dumps(row),flush=True)
        return stopped

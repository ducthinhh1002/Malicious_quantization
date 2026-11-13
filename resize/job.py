"""Latest Structure teacher -> Janus-Pro-1B LoRA32 autoregressive distillation."""
import os,sys,json,time,gc,hashlib,collections,collections.abc
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TEACHER_CODE=Path(os.environ.get('JPEG_CODE_DIR',str(ROOT.parent/'jpeg'))).expanduser().resolve()
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'Janus'),str(TEACHER_CODE),str(TEACHER_CODE/'core'),str(TEACHER_CODE/'harness')]
for name in ['Mapping','MutableMapping','Sequence']:
    if not hasattr(collections,name):setattr(collections,name,getattr(collections.abc,name))
import setproctitle
setproctitle.setproctitle('VLLM::Enginecore')
import torch,numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from semantic_teacher import extractor,numerical_policy
from semantic_carrier import SemanticCarrier
SOURCE=Path(os.environ.get('JPEG_TEACHER_RUN',str(TEACHER_CODE/'run'))).expanduser().resolve()
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
ROWS=json.loads(Path(os.environ.get('JANUS_PROMPTS',str(ROOT/'inputs/prompts.json'))).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def status(stage,**kwargs):write(ROOT/'status.json',dict(stage=stage,time=time.time(),**kwargs))

def image_tensor(im,size=384):
    im=im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/127.5-1

def save_rgb(value,path):
    array=((value.float().detach().cpu().clamp(-1,1)+1)*127.5).numpy().transpose(1,2,0).astype(np.uint8)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.png');Image.fromarray(array).save(tmp);tmp.replace(path)

def load_janus():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    processor=VLChatProcessor.from_pretrained(str(ROOT/'model'))
    model=AutoModelForCausalLM.from_pretrained(str(ROOT/'model'),trust_remote_code=True,torch_dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    return processor,model

def prompt_ids(processor,text):
    conv=[dict(role='<|User|>',content=text),dict(role='<|Assistant|>',content='')]
    text=processor.apply_sft_template_for_multi_turn_prompts(conversations=conv,sft_format=processor.sft_format,system_prompt='')+processor.image_start_tag
    return processor.tokenizer.encode(text)

def prefixes(processor,texts,drop=False,cfg=False):
    rows=[prompt_ids(processor,t) for t in texts];length=max(map(len,rows))
    ids=torch.full((len(rows)*(2 if cfg else 1),length),processor.pad_id,device='cuda',dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        j=2*i if cfg else i
        ids[j,-len(row):]=torch.tensor(row,device='cuda');mask[j,-len(row):]=1
        if drop and torch.rand(()).item()<.1:ids[j,-len(row)+1:-1]=processor.pad_id
        if cfg:
            ids[j+1]=ids[j];ids[j+1,-len(row)+1:-1]=processor.pad_id;mask[j+1]=mask[j]
    return ids,mask

def score(directory,rows,label):
    result=ROOT/'scores'/f'{label}.json'
    if result.exists():return
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    ext=extractor(SOURCE);carrier=SemanticCarrier.load(SOURCE/'carrier','cuda');values=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4];ims=[]
        for i in range(start,start+len(batch)):
            with Image.open(Path(directory)/f'sample_{i:05d}.png') as im:ims.append(image_tensor(im,512))
        with torch.inference_mode():q=carrier.coordinates(ext.encode_image(torch.stack(ims).cuda(),[r['text'] for r in batch]))
        values.extend(q.cpu().tolist())
    q=np.asarray(values);signs=carrier.signs.cpu().numpy();metrics=[]
    for size in [1,32,256]:
        if size>len(q):continue
        if size==1:groups=q
        else:
            rng=np.random.default_rng(0)
            groups=np.concatenate([q[rng.permutation(len(q))[:len(q)//size*size]].reshape(-1,size,32).mean(1) for _ in range(100)])
        matched=((groups>0)==(signs>0)).sum(1)
        metrics.append(dict(aggregate=size,n_groups=len(groups),BA=float(matched.mean()/32),accept_ge24=float((matched>=24).mean()),exact32=float((matched==32).mean())))
    write(result,dict(n=len(q),metrics=metrics,coordinates=values,threshold='fixed24/32; not empirical TPR@1%FPR',reader='frozen codec-stable Structure carrier',resize='LANCZOS to512 before frozen reader'))
    print('SCORED',label,metrics,flush=True)
    del ext,carrier;gc.collect();torch.cuda.empty_cache()

def preflight():
    if (ROOT/'preflight_done.json').exists():return
    status('tokenizer_probe')
    processor,model=load_janus()
    rows=json.loads((SOURCE/'prompts.json').read_text())['test'][:128]
    for i,row in enumerate(rows):
        path=SOURCE/'images/teacher_jpeg600_test'/f'sample_{i:06d}.png'
        if not path.exists():path=SOURCE/'images/teacher_jpeg600_test'/f'sample_{i:05d}.png'
        with Image.open(path) as im:rgb=image_tensor(im)
        save_rgb(rgb,ROOT/'probe_resize'/f'sample_{i:05d}.png')
        with torch.inference_mode():
            quant,_,info=model.gen_vision_model.encode(rgb[None].cuda().to(torch.bfloat16))
            codes=info[2].reshape(1,-1)
            if codes.shape!=(1,576):raise RuntimeError(f'Incorrect image token shape {codes.shape}')
            dec=model.gen_vision_model.decode_code(codes,shape=[1,8,24,24])
        save_rgb(dec[0],ROOT/'probe_vq'/f'sample_{i:05d}.png')
    del processor,model;gc.collect();torch.cuda.empty_cache()
    score(ROOT/'probe_resize',rows,'teacher_resize384')
    score(ROOT/'probe_vq',rows,'teacher_vq384')
    write(ROOT/'preflight_done.json',dict(n=128))

def teacher_images():
    if (ROOT/'teacher_done.json').exists():return
    from diffusers import StableDiffusionPipeline
    from student import load_adapter
    numerical_policy();torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    assets=json.loads((SOURCE/'assets.json').read_text())
    pipe=StableDiffusionPipeline.from_pretrained(assets['teacher_clean'],torch_dtype=torch.float32,safety_checker=None,local_files_only=True).to('cuda')
    load_adapter(pipe.unet,SOURCE/'teacher_fp/refined/checkpoint-1200')
    pipe.fuse_lora(safe_fusing=True);pipe.unload_lora_weights();pipe.set_progress_bar_config(disable=True)
    original=json.loads((SOURCE/'prompts.json').read_text())['train']
    forbidden={r['text'] for r in original};assert not forbidden.intersection(r['text'] for split in ROWS.values() if isinstance(split,list) for r in split)
    for split,n in [('train',5000),('validation',128),('test',1000)]:
        directory=DATA/split;directory.mkdir(exist_ok=True)
        for start in range(0,n,8):
            batch=ROWS[split][start:min(start+8,n)]
            paths=[directory/f'sample_{i:05d}.png' for i in range(start,start+len(batch))]
            if not all(p.exists() for p in paths):
                gens=[torch.Generator('cuda').manual_seed(r['image_id']) for r in batch]
                with torch.inference_mode():images=pipe([r['text'] for r in batch],generator=gens,num_inference_steps=50,guidance_scale=7.5).images
                for im,p in zip(images,paths):
                    tmp=p.with_suffix('.partial.png');im.save(tmp);tmp.replace(p)
            status('teacher_generation',split=split,n=start+len(batch),total=n)
    del pipe;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'teacher_done.json',dict(checkpoint='Structure codec checkpoint1200',train=5000,validation=128,test=1000))

def tokenize(processor,model):
    data={}
    for split,n in [('train',5000),('validation',128)]:
        path=DATA/f'{split}_tokens.pt'
        if path.exists():data[split]=torch.load(path,weights_only=True);continue
        values=[]
        for start in range(0,n,8):
            ims=[]
            for i in range(start,min(n,start+8)):
                with Image.open(DATA/split/f'sample_{i:05d}.png') as im:ims.append(image_tensor(im))
            with torch.inference_mode():_,_,info=model.gen_vision_model.encode(torch.stack(ims).cuda().to(torch.bfloat16))
            codes=info[2].reshape(len(ims),576).cpu().long();values.append(codes)
            status('tokenizing',split=split,n=start+len(ims),total=n)
        data[split]=torch.cat(values)
        temporary=path.with_suffix('.partial.pt');torch.save(data[split],temporary);temporary.replace(path)
    return data

def loss_for(processor,model,codes,rows,drop=False):
    codes=codes.clone()
    ids,mask=prefixes(processor,[r['text'] for r in rows],drop=drop)
    text=model.language_model.get_input_embeddings()(ids)
    images=model.prepare_gen_img_embeds(codes[:,:-1])
    embeds=torch.cat([text,images],dim=1)
    attention=torch.cat([mask,torch.ones_like(codes[:,:-1])],dim=1)
    out=model.language_model.model(inputs_embeds=embeds,attention_mask=attention,use_cache=False).last_hidden_state
    logits=model.gen_head(out[:,ids.shape[1]-1:])
    assert logits.shape[:2]==codes.shape
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),codes.reshape(-1))

def train(processor,model,data):
    base=model.language_model.model
    model.language_model.model=get_peft_model(base,LoraConfig(r=32,lora_alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_dropout=0.,bias='none'))
    adapter=model.language_model.model
    params=[p for p in adapter.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4)
    start=0
    checkpoints=sorted((ROOT/'checkpoints').glob('step_*'),key=lambda p:int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else -1) if (ROOT/'checkpoints').exists() else []
    checkpoints=[p for p in checkpoints if p.name.split('_')[-1].isdigit() and (p/'training.pt').exists()]
    if checkpoints:
        last=checkpoints[-1];state=torch.load(last/'training.pt',weights_only=False,map_location='cpu')
        set_peft_model_state_dict(adapter,load_peft_weights(str(last),device='cpu'))
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step']
    else:torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    log=ROOT/'train.jsonl'
    if log.exists():
        valid=[line for line in log.read_text().splitlines() if json.loads(line)['step']<=start]
        log.write_text(''.join(line+'\n' for line in valid))
    adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});adapter.enable_input_require_grads();adapter.train()
    for step in range(start,12500):
        epoch,offset=divmod(step*4,5000)
        order=torch.randperm(5000,generator=torch.Generator().manual_seed(epoch)).tolist();indices=order[offset:offset+4]
        codes=data['train'][indices].cuda();rows=[ROWS['train'][i] for i in indices]
        optimizer.zero_grad(set_to_none=True);loss=loss_for(processor,model,codes,rows,drop=True)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite AR loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=1e-4*min(1.,(step+1)/100)
        optimizer.step()
        record=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm))
        with (ROOT/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        status('student_training',**record,total=12500)
        if (step+1)%1250==0:
            dest=ROOT/'checkpoints'/f'step_{step+1}';temporary=dest.with_name(dest.name+'.partial');temporary.mkdir(parents=True,exist_ok=True)
            adapter.save_pretrained(temporary)
            torch.save(dict(step=step+1,optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temporary/'training.pt')
            temporary.rename(dest)
            adapter.eval()
            with torch.inference_mode():vl=[float(loss_for(processor,model,data['validation'][i:i+4].cuda(),ROWS['validation'][i:i+4])) for i in range(0,128,4)]
            write(dest/'validation.json',dict(CE=sum(vl)/len(vl)));adapter.train()
    adapter.eval();adapter.gradient_checkpointing_disable()

@torch.inference_mode()
def generate(processor,model,label):
    directory=ROOT/label;directory.mkdir(exist_ok=True)
    for start in range(0,1000,8):
        batch=ROWS['test'][start:start+8];paths=[directory/f'sample_{i:05d}.png' for i in range(start,start+len(batch))]
        if all(p.exists() for p in paths):continue
        torch.manual_seed(100000+start);torch.cuda.manual_seed_all(100000+start)
        ids,mask=prefixes(processor,[r['text'] for r in batch],cfg=True)
        embeds=model.language_model.get_input_embeddings()(ids);past=None;codes=[]
        for _ in range(576):
            out=model.language_model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=True,past_key_values=past)
            past=out.past_key_values;logits=model.gen_head(out.last_hidden_state[:,-1]);cond,uncond=logits[::2],logits[1::2]
            token=torch.multinomial(torch.softmax((uncond+5*(cond-uncond)).float(),-1),1)
            codes.append(token);embeds=model.prepare_gen_img_embeds(token.repeat_interleave(2,dim=0).reshape(-1)).unsqueeze(1)
            mask=torch.cat([mask,torch.ones((len(ids),1),device='cuda',dtype=mask.dtype)],1)
        dec=model.gen_vision_model.decode_code(torch.cat(codes,1).int(),shape=[len(batch),8,24,24])
        for image,path in zip(dec,paths):save_rgb(image,path)
        status(label,n=start+len(batch),total=1000)

def main():
    write(ROOT/'prompts.json',ROWS)
    preflight();teacher_images()
    processor,model=load_janus();data=tokenize(processor,model)
    generate(processor,model,'base_test')
    train(processor,model,data);generate(processor,model,'student_test')
    del processor,model;gc.collect();torch.cuda.empty_cache()
    score(DATA/'test',ROWS['test'],'teacher_test')
    score(ROOT/'base_test',ROWS['test'],'base_test')
    score(ROOT/'student_test',ROWS['test'],'student_test')
    status('completed')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));raise

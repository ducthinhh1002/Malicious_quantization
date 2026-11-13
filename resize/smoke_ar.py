import job
from job import torch,ROOT,Image,image_tensor,write,loss_for,load_janus,get_peft_model,LoraConfig
processor,model=load_janus()
row=job.json.loads((job.SOURCE/'prompts.json').read_text())['test'][0]
directory=job.SOURCE/'images/teacher_jpeg600_test'
paths=sorted(directory.glob('sample_*.png'))
with Image.open(paths[0]) as im:rgb=image_tensor(im)
with torch.inference_mode():_,_,info=model.gen_vision_model.encode(rgb[None].cuda().to(torch.bfloat16))
codes=info[2].reshape(1,576)
model.language_model.model=get_peft_model(model.language_model.model,LoraConfig(r=32,lora_alpha=32,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none'))
adapter=model.language_model.model
adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});adapter.enable_input_require_grads();adapter.train()
params=[p for p in adapter.parameters() if p.requires_grad]
before=[p.detach().clone() for p in params]
optimizer=torch.optim.AdamW(params,lr=1e-4)
loss=loss_for(processor,model,codes,[row]);loss.backward()
norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
optimizer.step()
changed=any(not torch.equal(a,b) for a,b in zip(before,params))
assert changed and torch.isfinite(loss)
adapter.eval();adapter.gradient_checkpointing_disable()
with torch.inference_mode():
    ids,mask=job.prefixes(processor,[row['text']],cfg=True)
    embeds=model.language_model.get_input_embeddings()(ids);past=None
    for _ in range(2):
        out=adapter(inputs_embeds=embeds,attention_mask=mask,use_cache=True,past_key_values=past)
        past=out.past_key_values
        logits=model.gen_head(out.last_hidden_state[:,-1])
        assert torch.isfinite(logits).all()
        token=logits[::2].argmax(-1)
        embeds=model.prepare_gen_img_embeds(token.repeat_interleave(2)).unsqueeze(1)
        mask=torch.cat([mask,torch.ones((2,1),device='cuda',dtype=mask.dtype)],1)
write(ROOT/'smoke_ar.json',dict(generation_cache_steps=2,passed=True,CE=float(loss.detach()),grad_norm=float(norm),trainable_parameters=sum(p.numel() for p in params),weights_changed=changed,image_token_shape=list(codes.shape),note='one real teacher-forced update; diagnostic model discarded'))
print('AR_SMOKE_PASSED',float(loss.detach()),flush=True)
del model,adapter,params,optimizer,before,processor
job.gc.collect();torch.cuda.empty_cache()
probe=ROOT/'probe_original';probe.mkdir(exist_ok=True)
for i,path in enumerate(paths[:128]):
 target=probe/f'sample_{i:05d}.png'
 if not target.exists():target.symlink_to(path)
job.score(probe,job.json.loads((job.SOURCE/'prompts.json').read_text())['test'][:128],'teacher_original128')

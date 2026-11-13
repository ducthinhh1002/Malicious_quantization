"""Use existing clean architecture inference helpers; never copy a marked VAE."""
from pathlib import Path

def generate(arch,mode,model_path,job,rows,output):
    import torch
    from diffusers import StableDiffusionPipeline,SanaPipeline,PixArtAlphaPipeline
    from student import load_adapter,prompt_kwargs
    from common import save_images,manifest
    cls={'sd14':StableDiffusionPipeline,'sana600m':SanaPipeline,'pixart512':PixArtAlphaPipeline}[arch]
    dtype=torch.float16 if arch=='sd14' else torch.bfloat16
    kw={'safety_checker':None,'requires_safety_checker':False} if arch=='sd14' else {}
    pipe=cls.from_pretrained(model_path,torch_dtype=dtype,local_files_only=True,**kw).to('cuda');pipe.set_progress_bar_config(disable=True)
    attr='unet' if arch=='sd14' else 'transformer';model=getattr(pipe,attr)
    branch=Path(job)/'students'/('sd14_full' if mode=='full' else f'sd14_{mode}')
    if mode=='full':
        folder=branch/'unet' if arch=='sd14' else Path(job)/'transformer'
        setattr(pipe,attr,type(model).from_pretrained(folder,torch_dtype=dtype,local_files_only=True).cuda())
    elif arch=='sd14':
        # Reused Diffusers trainer can emit legacy pytorch_lora_weights rather
        # than PEFT adapter_model. Use the historical loader that supports both.
        from benchmark.shared.student_training.integration import function,REF
        namespace={'Path':Path}
        exec(function(REF/'harness/student.py','load_adapter'),namespace)
        namespace['load_adapter'](model,branch,pipe)
    else:load_adapter(model,Path(job)/'adapter')
    if arch!='sd14':pipe.vae.to(torch.float32)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4):
        ids=list(range(start,min(start+4,len(rows))))
        with torch.inference_mode():
            args=dict(prompt=[rows[i]['text'] for i in ids],height=512,width=512,num_inference_steps=50,guidance_scale=7.5,
                generator=[torch.Generator('cuda').manual_seed(500000+i) for i in ids],**prompt_kwargs(arch))
            if arch!='sd14':args['use_resolution_binning']=False
            if arch=='pixart512':
                z=pipe(output_type='latent',**args).images
                images=pipe.image_processor.postprocess(pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,return_dict=False)[0],output_type='pil')
            else:images=pipe(**args).images
        save_images(images,output,ids)
    manifest(output,rows);print('DONE student images',arch,mode)

"""Matched native-output evaluation of stored Drop96 checkpoints; no training."""
from pathlib import Path
import sys,os,json,time,subprocess,fcntl,gc
ROOT=Path(__file__).resolve().parent
OUT=Path('/var/lib/drop96_checkpoint_review')
sys.path.insert(0,'/var/lib/structure32_student12500/harness')
from common import read_json,write_json,freeze,sha256

def source(rank,step):
    if rank==4:
        run=Path('/var/lib/structure32_student_manipulations_rank4/branches/drop96/runs/structure32_student20000_seed0/student20000')
    elif step<=12500:
        run=Path('/var/lib/structure32_student_manipulations/branches/drop96/runs/structure32_student12500_seed0/student12500')
    else:run=Path('/var/lib/structure32_student_manipulations/continuation20k/drop96/run')
    return run,run/'students'/f'lora_r{rank}_conv_wm'

def evaluate(rank,step,check_only=False):
    import torch
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig,get_peft_model_state_dict
    from safetensors.torch import load_file
    from semantic_teacher import numerical_policy,extractor,pixels
    from semantic_carrier import SemanticCarrier
    from student import generate_images
    from common import image_name
    sys.path.insert(0,str(ROOT/'branches/drop96'))
    from acceptance_metrics import aggregate_metrics
    numerical_policy()
    src,student=source(rank,step)
    cp=student/f'checkpoint-{step}'
    proof=read_json(cp/'resume_manifest.json')
    assert proof['step']==step and proof['identity']['rank']==rank
    assert proof['identity']['training_manipulation']['name']=='drop96'
    assert sha256(cp/'trainable.safetensors')==proof['files']['trainable.safetensors']
    run=OUT/f'rank{rank}'/f'step{step}';run.mkdir(parents=True,exist_ok=True)
    for n in ['config.json','assets.json','prompts.json','carrier']:
        dst=run/n
        if not dst.exists():dst.symlink_to((src/n).resolve(),target_is_directory=(src/n).is_dir())
    protocol={'rank':rank,'step':step,'source_checkpoint':str(cp),'weight_sha256':proof['files']['trainable.safetensors'],
        'source_progress':proof['progress'],'prompts_sha256':sha256(run/'prompts.json'),
        'carrier_sha256':sha256(run/'carrier/complete.json'),'test_n':1000,
        'code_sha256':sha256(Path(__file__)),'selection':'exploratory checkpoint comparison on existing test; no training changes',
        'generation':'same SD1.4 base, CFG7.5, native scheduler,50steps,seed500000+i,FP16,batch16; output not attacked'}
    freeze(run/'protocol.json',protocol)
    if (run/'result.json').exists():return
    write_json(run/'status.json',{'stage':'loading_checkpoint','updated':time.time()})
    pipe=StableDiffusionPipeline.from_pretrained(read_json(run/'assets.json')['student'],torch_dtype=torch.float16,
        safety_checker=None,local_files_only=True).to('cuda')
    pipe.unet.requires_grad_(False)
    config=LoraConfig.from_pretrained(str(student));assert config.r==rank
    pipe.unet.add_adapter(config)
    weights=load_file(str(cp/'trainable.safetensors'))
    params={n:p for n,p in pipe.unet.named_parameters() if 'lora_' in n}
    assert params.keys()==weights.keys()
    with torch.no_grad():
        for n,p in params.items():
            assert p.shape==weights[n].shape and torch.isfinite(weights[n]).all()
            p.copy_(weights[n])
    if check_only:
        # Final checkpoint loader must equal native final adapter, tensor by tensor.
        native=load_file(str(student/'adapter_model.safetensors'))
        restored=get_peft_model_state_dict(pipe.unet)
        assert native.keys()==restored.keys()
        for n,p in restored.items():torch.testing.assert_close(p.cpu(),native[n].to(p.dtype),atol=0,rtol=0)
        print('CHECKPOINT_LOADER_EXACT_PARITY',rank,step,flush=True);return
    pipe.unet.eval();pipe.vae.eval();pipe.text_encoder.eval();pipe.set_progress_bar_config(disable=True)
    write_json(run/'status.json',{'stage':'generate1000','updated':time.time()})
    generate_images(run,f'lora_r{rank}_conv_wm',pipe,{'trainable.safetensors':proof['files']['trainable.safetensors']},label='student')
    del pipe,params,weights;gc.collect();torch.cuda.empty_cache()
    write_json(run/'status.json',{'stage':'score','updated':time.time()})
    ext=extractor(run);carrier=SemanticCarrier.load(run/'carrier','cuda')
    prompts=read_json(run/'prompts.json')['test'];coords=[]
    for start in range(0,1000,8):
        ids=range(start,min(start+8,1000))
        with torch.inference_mode():
            q=carrier.coordinates(ext.encode_image(pixels([run/'images/student'/image_name(i) for i in ids]),[prompts[i]['text'] for i in ids]))
        coords.extend(q.cpu().tolist())
    metrics=aggregate_metrics(coords,carrier.signs.cpu().tolist())
    freeze(run/'result.json',{'protocol':protocol,'coordinates':coords,'metrics':metrics})
    write_json(run/'status.json',{'stage':'completed','updated':time.time()})
    print('RESULT',rank,step,json.dumps([r for r in metrics['rows'] if r['images_per_group'] in [1,256]]),flush=True)

def manager():
    OUT.mkdir(exist_ok=True)
    with (OUT/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Highest priority: rank4 before its poor final checkpoint; rank32 trend.
        priority=[(4,12500),(4,15000),(4,17500),(32,17500),(4,10000),(32,15000),(32,18750),(4,18750)]
        jobs=priority+[(rank,s) for s in range(1250,20000,1250) for rank in [4,32]
                       if (rank,s) not in priority and not (rank==32 and s==12500)]
        jobs=[(r,s) for r,s in jobs if not (OUT/f'rank{r}'/f'step{s}'/'result.json').exists()]
        slots=[{'gpu':0},{'gpu':1},{'gpu':0},{'gpu':1}];finished=[]
        while jobs or any('p' in s for s in slots):
            for slot in slots:
                if 'p' in slot and slot['p'].poll() is not None:
                    finished.append({'job':slot['job'],'returncode':slot['p'].returncode})
                    slot['log'].close();del slot['p'];del slot['log'];del slot['job']
                if 'p' not in slot and jobs:
                    rank,step=jobs.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(slot['gpu']),PYTHONPATH=str(ROOT),
                        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                    log=(OUT/f'r{rank}_step{step}.log').open('ab')
                    proc=subprocess.Popen([sys.executable,str(Path(__file__)),str(rank),str(step)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    slot.update(p=proc,job=[rank,step],log=log)
            write_json(OUT/'status.json',{'running':[{'job':s['job'],'gpu':s['gpu'],'pid':s['p'].pid} for s in slots if 'p' in s],
                                        'queued':jobs,'finished':finished,'updated':time.time()})
            if any('p' in s for s in slots):time.sleep(15)

if __name__=='__main__':
    if len(sys.argv)==1:manager()
    else:
        rank,step=map(int,sys.argv[1:3])
        try:evaluate(rank,step,'--check-only' in sys.argv)
        except Exception as e:
            run=OUT/f'rank{rank}'/f'step{step}';run.mkdir(parents=True,exist_ok=True)
            write_json(run/'status.json',{'stage':'failed','error':repr(e),'updated':time.time()});raise

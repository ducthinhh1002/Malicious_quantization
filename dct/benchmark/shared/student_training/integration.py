"""Reuse historical trainers, making path/adaptation changes explicit and auditable."""
import ast,hashlib,json,shutil,sys
from pathlib import Path
REF=Path(__file__).parent/'reference'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(path,name):
    source=Path(path).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'

def sd14(job,model,data,rank):
    job=Path(job).resolve();job.mkdir(parents=True,exist_ok=True)
    helper=job/'helpers';helper.mkdir(exist_ok=True)
    for p in (REF/'pipeline/student').glob('*.py'):shutil.copyfile(p,helper/p.name)
    shutil.copyfile(REF/'harness/common.py',helper/'common.py')
    # The original full-state hook intentionally accepted LoRA names only.
    # Full FT uses the same hooks, but must accept all finite trainable names.
    if rank is None:
        p=helper/'resume_state.py';s=p.read_text()
        s=s.replace("'lora_' not in n or not torch.isfinite(t).all()","not torch.isfinite(t).all()")
        p.write_text(s)
    identity={'trainer_sha256':sha(REF/'harness/vendor/diffusers'/('train_text_to_image.py' if rank is None else 'train_text_to_image_lora.py')),
              'training_images_manifest_sha256':sha(Path(data)/'image_hashes.json'),'seed':0,'batch_size':4,
              'steps_per_epoch':1250,'rank':rank,'scope':'full' if rank is None else 'attention_conv'}
    (job/'resume_identity.json').write_text(json.dumps(identity,indent=2))
    scope={};exec(function(REF/'harness/student.py','training_command'),{'sys':sys,'HERE':REF/'harness'},scope)
    # The reused command expects run/images/teacher_wm_train. Link only on user execution.
    return {'helper':helper,'identity':job/'resume_identity.json','builder':scope['training_command'],
            'config':{'seed':0,'batch_size':4,'learning_rate':1e-4,'precision':'fp16','max_train_steps':20000,'checkpoint_steps':[20000]},
            'assets':{'student':str(Path(model).resolve())},
            'branch':{'name':'sd14_full' if rank is None else f'sd14_r{rank}','kind':'full' if rank is None else 'lora','condition':'wm','rank':rank,'scope':'attention_conv'}}

def dit_source(rank):
    """Original cache/train functions; replace only paths, owner provenance and scope."""
    source=REF/'cross_arch_jobs.py'
    names=['pipeline','prompt_kwargs','lr_factor','cache','train']
    code='\n\n'.join(function(source,n) for n in names)
    code=code.replace('from semantic_teacher import pixels','from benchmark.shared.student_training.worker import pixels')
    code=code.replace('from semantic_teacher import numerical_policy','from benchmark.shared.student_training.worker import numerical_policy')
    if rank is not None:
        code=code.replace('LoraConfig(r=32,lora_alpha=32',f'LoraConfig(r={rank},lora_alpha={rank}').replace("'rank':32",f"'rank':{rank}")
    else:
        start=code.index('    model.requires_grad_(False)',code.index('def train('));end=code.index('    scheduler=',start)
        code=code[:start]+'''    model.requires_grad_(True);model.to(torch.float32)
    model.train();model.enable_gradient_checkpointing()
    named=list(model.named_parameters());params=[v for n,v in named]
'''+code[end:]
        start=code.index('    if smoke:\n        assert applied>0')
        code=code[:start]+'''    dest=p/'transformer';model.save_pretrained(dest,safe_serialization=True)
    freeze(p/'trained.json',{'attempted_steps':total,'applied':applied,'skipped':skipped,
        'files':{f.name:sha256(f) for f in dest.iterdir() if f.is_file()}})
'''
    return code

"""Compact complete state, deterministic epoch order and audited continuation."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import save_file, load_file


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_checkpoint(path,identity):
    path=Path(path)
    proof=json.loads((path/'resume_manifest.json').read_text())
    if proof['identity']!=identity:
        raise RuntimeError('Resume protocol/base/data identity changed')
    if any(digest(path/name)!=value for name,value in proof['files'].items()):
        raise RuntimeError('Checkpoint content hash mismatch')
    if proof['step'] % proof['steps_per_epoch']:
        raise RuntimeError('Only complete epoch checkpoints are resumable')
    required=['trainable.safetensors','optimizer.bin','scheduler.bin','random_states_0.pkl','custom_checkpoint_0.pkl']
    if any(name not in proof['files'] for name in required):
        raise RuntimeError('Missing optimizer/scheduler/RNG/progress checkpoint component')
    return proof


class Progress:
    def __init__(self):
        self.step=0;self.applied=0;self.skipped=0
    def state_dict(self):
        return dict(step=self.step,applied=self.applied,skipped=self.skipped)
    def load_state_dict(self,state):
        self.step=state['step'];self.applied=state['applied'];self.skipped=state['skipped']


def accelerator_class(identity, seed, steps_per_epoch, audit_dir):
    audit_dir=Path(audit_dir)

    class ResumableAccelerator(Accelerator):
        def __init__(self,*args,**kwargs):
            kwargs['dataloader_config']=DataLoaderConfiguration(use_seedable_sampler=True,data_seed=seed)
            super().__init__(*args,**kwargs)
            if self.num_processes!=1 or self.gradient_accumulation_steps!=1:
                raise RuntimeError('This protocol requires one process and accumulation1')
            self.progress=Progress()
            self.register_for_checkpointing(self.progress)
            self.register_save_state_pre_hook(self.save_adapters)
            self.register_load_state_pre_hook(self.load_adapters)
            self.last_time=time.monotonic()

        def save_adapters(self,models,weights,directory):
            if len(models)!=1:raise RuntimeError('Expected one UNet')
            state={n:p.detach().cpu().contiguous() for n,p in self.unwrap_model(models[0]).named_parameters() if p.requires_grad}
            if not state or any('lora_' not in n or not torch.isfinite(t).all() for n,t in state.items()):
                raise RuntimeError('Unexpected/nonfinite trainable parameters')
            save_file(state,str(Path(directory)/'trainable.safetensors'))
            weights.clear()

        def load_adapters(self,models,directory):
            if len(models)!=1:raise RuntimeError('Expected one UNet')
            state=load_file(str(Path(directory)/'trainable.safetensors'))
            params={n:p for n,p in self.unwrap_model(models[0]).named_parameters() if p.requires_grad}
            if params.keys()!=state.keys():raise RuntimeError('Adapter state keys differ')
            with torch.no_grad():
                for name,p in params.items():
                    if p.shape!=state[name].shape or not torch.isfinite(state[name]).all():
                        raise RuntimeError('Invalid adapter state')
                    p.copy_(state[name])
            models.clear()

        def log(self,values,step=None,**kwargs):
            if step is not None and 'train_loss' in values:
                if step!=self.progress.step+1:raise RuntimeError('Noncontiguous optimizer step history')
                self.progress.step=step
                skipped=int(self.optimizer_step_was_skipped)
                self.progress.skipped+=skipped;self.progress.applied+=1-skipped
                now=time.monotonic()
                row={**self.progress.state_dict(),'loss':float(values['train_loss']),
                    'lr':self._optimizers[0].param_groups[0]['lr'],
                    'seconds':now-self.last_time,
                    'allocated_gib':torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0.}
                self.last_time=now
                audit_dir.mkdir(parents=True,exist_ok=True)
                with (audit_dir/'updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                if step%100==0:print('UPDATE',json.dumps(row),flush=True)
            return super().log(values,step=step,**kwargs)

        def save_state(self,output_dir=None,**kwargs):
            step=self.progress.step
            if not output_dir or step<=0 or step%steps_per_epoch:
                raise RuntimeError('Save only complete epoch boundaries')
            target=Path(output_dir)
            if target.exists():raise RuntimeError('Refusing to replace an existing state checkpoint')
            temp=target.with_name('.pending-'+target.name+'-'+str(os.getpid()))
            if temp.exists():raise RuntimeError('Pending save directory already exists')
            result=super().save_state(str(temp),**kwargs)
            files={p.name:digest(p) for p in temp.iterdir() if p.is_file()}
            proof={'format':1,'identity':identity,'step':step,'steps_per_epoch':steps_per_epoch,
                'next_epoch':step//steps_per_epoch,'progress':self.progress.state_dict(),
                'data_order':'SeedableRandomSampler(seed+epoch); explicit set_epoch; epoch-boundary resume',
                'files':files}
            (temp/'resume_manifest.json').write_text(json.dumps(proof,indent=2))
            verify_checkpoint(temp,identity)
            if self.scaler is not None and 'scaler.pt' not in files:
                raise RuntimeError('FP16 scaler missing from saved state')
            temp.rename(target)
            print('FULL_STATE_SAVED',str(target),flush=True)
            return result

        def load_state(self,input_dir=None,**kwargs):
            proof=verify_checkpoint(input_dir,identity)
            result=super().load_state(str(input_dir),**kwargs)
            if self.progress.state_dict()!=proof['progress']:
                raise RuntimeError('Progress state not restored')
            # Preserve interrupted suffix as evidence, then continue the log
            # from the last complete state. No checkpoint is deleted.
            log=audit_dir/'updates.jsonl'
            if log.exists():
                rows=[json.loads(s) for s in log.read_text().splitlines()]
                if any(r['step']>proof['step'] for r in rows):
                    backup=log.with_name(f'updates.before_resume.{time.time_ns()}.jsonl')
                    log.rename(backup)
                    log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=proof['step']))
            print('FULL_STATE_RESTORED',proof['step'],flush=True)
            return result

    return ResumableAccelerator


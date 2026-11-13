"""Choose JPEG-stable semantic PCs using clean TRAIN pairs only."""
import bootstrap
from bootstrap import ROOT,read,write,sha,freeze
from pathlib import Path
import shutil,torch,numpy as np
import os
from semantic_teacher import extractor,pixels
from semantic_carrier import SemanticCarrier
from common import image_name
from jpeg_channel import jpeg_ste
source=Path(os.environ.get('JPEG_SOURCE_RUN',str(ROOT/'source'))).expanduser().resolve()
run=ROOT/'run';run.mkdir(exist_ok=True)
for name in ['assets.json','prompts.json','config.json']:
 value=read(source/name)
 if name=='config.json':value['generation_batch']=1
 freeze(run/name,value)
(run/'images').mkdir(exist_ok=True)
for name in ['teacher_clean_train','teacher_clean_validation']:
 target=run/'images'/name
 if not target.exists():target.symlink_to(source/'images'/name,target_is_directory=True)
carrier=run/'carrier';carrier.mkdir(exist_ok=True)
if not (run/'carrier_validation.json').exists():
 ext=extractor(run);rows=read(run/'prompts.json')
 pca=torch.load(source/'carrier/pca.pt',weights_only=True,map_location='cpu')
 old=read(source/'carrier/key.json')
 basis=pca['basis']; mean=pca['mean']
 def features(split,n,compressed):
  cache=run/f'features_{split}_{n}_{compressed}.pt'
  if cache.exists():return torch.load(cache,weights_only=True)
  values=[]
  for start in range(0,n,4):
   im=pixels([run/'images'/('teacher_clean_'+split)/image_name(i) for i in range(start,min(n,start+4))])
   with torch.inference_mode():
    if compressed:im=jpeg_ste(im,25)
    f=ext.encode_image(im,[r['text'] for r in rows[split][start:start+4]])
   values.append(f.cpu())
   if start%100==0:print(split,compressed,start,flush=True)
  result=torch.cat(values);torch.save(result,cache);return result
 a=features('train',1024,False);b=features('train',1024,True)
 qa=(a-mean)@basis;qb=(b-mean)@basis
 variance=qa.var(0);nmse=(qa-qb).square().mean(0)/variance.clamp_min(1e-8)
 retention=((qa>0)==(qb>0)).float().mean(0)
 # Avoid near-null PCs whose standardized shifts could be numerically cheap.
 config=read(run/'config.json')
 eligible=[i for i in range(config['carrier_pc_min'],config['carrier_pc_max']+1) if variance[i]>1e-6]
 # Learn a stable subspace, including correlated JPEG errors and mean bias.
 pool=basis[:,eligible].double()
 clean=(a.double()-a.double().mean(0))@pool
 delta=(a.double()-b.double())@pool
 signal=clean.T@clean/(len(clean)-1)
 noise=delta.T@delta/len(delta)
 eigen,rotation=torch.linalg.eigh(signal)
 ridge=signal.diag().mean()*1e-4
 whitening=rotation@torch.diag((eigen.clamp_min(ridge)).rsqrt())@rotation.T
 error=whitening@noise@whitening
 _,vectors=torch.linalg.eigh((error+error.T)/2)
 stable=pool@whitening@vectors[:,:32]
 orthogonal,_=torch.linalg.qr(stable,mode='reduced')
 projected=(a.double()-a.double().mean(0))@orthogonal
 _,rotation=torch.linalg.eigh(projected.T@projected/(len(projected)-1))
 new_basis=(orthogonal@rotation.flip(1)).float()
 old_nmse=nmse.clone();old_retention=retention.clone()
 basis=new_basis
 qa=(a-a.mean(0))@basis;qb=(b-a.mean(0))@basis
 variance=qa.var(0)
 nmse=(qa-qb).square().mean(0)/variance.clamp_min(1e-8)
 retention=((qa>0)==(qb>0)).float().mean(0)
 selected=list(range(32))
 sigmas=variance[selected].sqrt()
 # Recenter using the actual fitting subset; the orthogonal PCA basis is unchanged.
 train_mean=a.mean(0)
 key=dict(old,selected_pc_indices=selected,sigmas=sigmas.tolist(),basis_sha256='pending',carrier_version=None)
 pca=dict(pca,mean=train_mean,basis=basis);torch.save(pca,carrier/'pca.pt')
 key['basis_sha256']=sha(carrier/'pca.pt')
 write(carrier/'key.json',key)
 write(carrier/'provenance.json',dict(method='generalized eigenvectors of paired JPEG error/signal covariance in original PC8..64 subspace; orthogonalized and signal-decorrelated;1024 clean pairs',signs_source='original32-bit key',training_only=True))
 write(carrier/'stability.json',dict(new=[dict(pc=i,nmse=float(nmse[i]),retention=float(retention[i])) for i in selected],old=[dict(pc=i,nmse=float(old_nmse[i]),retention=float(old_retention[i])) for i in old['selected_pc_indices']]))
 write(carrier/'complete.json',{n:sha(carrier/n) for n in ['pca.pt','key.json','provenance.json','stability.json']})
 new=SemanticCarrier.load(carrier)
 # Freeze scalar target on independent clean validation images after PC selection.
 f=features('validation',1000,False);q=new.coordinates(f);scores=(q*new.signs).mean(1)
 threshold=float(torch.sort(scores).values[-10]);sigma=float(scores.std(unbiased=False))
 write(run/'teacher_fp/calibration_target.json',dict(threshold=threshold,sigma_clean=sigma,n=1000,target_fpr=.01,data_source='clean validation, excluded from carrier fitting and model training',empirical_fpr=float((scores>=threshold).float().mean())))
 # Independent clean paired validation measures carrier stability, not generator performance.
 fv=features('validation',1000,True);qv=new.coordinates(fv)
 previous=SemanticCarrier.load(source/'carrier')
 oldq=previous.coordinates(f);oldqv=previous.coordinates(fv)
 write(run/'carrier_validation.json',dict(old_validation_sign_retention=float(((oldq>0)==(oldqv>0)).float().mean()),old_validation_normalized_mse=float((oldq-oldqv).square().mean()),new_sign_retention=float(((q>0)==(qv>0)).float().mean()),new_normalized_mse=float((q-qv).square().mean()),old_train_sign_retention=float(old_retention[old['selected_pc_indices']].mean()),new_train_sign_retention=float(retention[selected].mean())))
 del ext
write(run/'teacher_choice.json',dict(method='JPEG-stable semantic Structure',source_checkpoint=str(Path(os.environ.get('JPEG_WARM_ADAPTER',str(ROOT/'inputs/refine600'))).expanduser().resolve()),source_adapter_sha256='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'))
write(run/'refine_spec.json',dict(teacher_steps=1200,checkpoint_every=50,teacher_lr=1e-5,transfer=False,native_every=2,backward_calls=20,quality_fraction=1.,score_fraction=1.,max_gradient_gain=100.,pixel_weight=.1,structure_weight=.25,project_step=True,mu=1.,native_batch=1,curriculum_updates=100,selection='fixed1200; no test selection'))
paths=list(ROOT.glob('*.py'))+list((ROOT/'core').glob('*.py'))+list((ROOT/'harness').rglob('*.py'))
write(run/'code_manifest.json',{str(p):sha(p) for p in paths})
print('PREPARED',read(run/'carrier_validation.json'),flush=True)

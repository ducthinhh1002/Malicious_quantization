from pathlib import Path
source=(Path(__file__).resolve().parent/'latent32_student_job.py').read_text()
exec(source.split('\nwm=reader()\n')[0])
from grow.utils import image_to_tensor,extract_center_patch
wm=reader(); key=np.array(message_to_bits(cfg.message)); out=folder/'student/evaluation/12500'
rng=np.random.default_rng(20261111)
groups=np.stack([rng.choice(1000,32,replace=False) for _ in range(5000)])
np.save(out/'agg32_groups.npy',groups)
results={}
for role,directory in [('teacher',folder/'images/teacher_wm_test'),('student',out/'test_images')]:
 cache=out/(role+'_dct_coordinates.npz')
 if cache.exists():
  z=np.load(cache); scores={k:z[k] for k in z.files}
 else:
  scores={'original':[],'jpeg25':[]}
  for idx,path in enumerate(sorted(directory.glob('sample_*.png'))):
   with Image.open(path) as image:
    im=image.convert('RGB'); b=io.BytesIO(); im.save(b,format='JPEG',quality=25,subsampling=2); b.seek(0)
    with Image.open(b) as ji: variants=[im,ji.convert('RGB')]
    with torch.no_grad():
     x=torch.stack([image_to_tensor(v) for v in variants]).to(wm.device,wm.dtype)*2-1
     latent=wm.vae.encode(x).latent_dist.mean*wm.vae.config.scaling_factor
     center=extract_center_patch(latent,cfg.center_ratio); values=dct(center)
     h,w=center.shape[-2:]; coords=[(a,b) for a in range(int(h*cfg.dct_min),int(h*cfg.dct_max)) for b in range(int(w*cfg.dct_min),int(w*cfg.dct_max))]
     np.random.default_rng(sum(map(ord,cfg.secret_key))).shuffle(coords)
     # 8 bits/channel, 50 copies/bit, exactly the deployed reader order.
     v=torch.stack([values[:,ch,[c[0] for c in coords],[c[1] for c in coords]].reshape(2,50,8) for ch in cfg.channels],dim=1).cpu().numpy()
    for i,label in enumerate(scores):scores[label].append(v[i])
   if (idx+1)%100==0:print(role,idx+1,flush=True)
  scores={k:np.stack(v) for k,v in scores.items()}; np.savez_compressed(cache,**scores)
 results[role]={}
 for label,v in scores.items():
  matches=[]
  for g in groups:
   avg=v[g].mean(axis=0)
   positive=avg>0; count=positive.sum(axis=1)
   # Match Counter tie breaking: first copy determines tied votes.
   bits=np.where(count==25,positive[:,0,:],count>25).reshape(32)
   matches.append(int((bits==key).sum()))
  a=np.array(matches)
  results[role][label]={'BA':float(a.mean()/32),'accept_ge24':float((a>=24).mean()),'exact32':float((a==32).mean())}
 reused.write(out/'aggregate32_results.json',dict(group_size=32,n_groups=5000,unique_images=1000,seed=20261111,replacement_within_group=False,aggregation='mean DCT coefficients across images, then per-bit repetition majority vote; ties use first copy',results=results))
 print(role,results[role],flush=True)

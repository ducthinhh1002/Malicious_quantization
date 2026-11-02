"""Exploratory severity curves; all severities retained, no training changes."""
import bootstrap
from pathlib import Path
import json,os,time,fcntl
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode as I
from common import read_json,write_json,freeze,sha256,image_name
from semantic_teacher import extractor,pixels
from semantic_carrier import SemanticCarrier
from attacks import quantize
from key_metrics import aggregate_metrics

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run'
OUT=RUN/'result/severity_sweep'
# Order gives a representative mild result for each family first, then covers
# the entire predeclared range. Never stop on reaching a desired score.
SETTINGS=[('rotation',5),('crop',.95),('drop_keep',.95),
          ('rotation',1),('rotation',3),('rotation',10),('rotation',15),
          ('rotation',30),('rotation',45),('rotation',90),
          ('crop',.98),('crop',.90),('crop',.85),('crop',.80),('crop',.75),
          ('drop_keep',.99),('drop_keep',.98),('drop_keep',.90),
          ('drop_keep',.85),('drop_keep',.80),('drop_keep',.64)]

def transform(x,family,severity,index):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((20261029 if family=='rotation' else 20261030)+index)
        if family=='rotation':
            angle=T.RandomRotation.get_params([-severity,severity])
            return quantize(F.rotate(x,angle,interpolation=I.NEAREST,fill=.5))
        if family=='crop':
            h,w=x.shape[-2:];ch,cw=round(h*severity**.5),round(w*severity**.5)
            top=int(torch.randint(h-ch+1,()).item());left=int(torch.randint(w-cw+1,()).item())
            return F.resized_crop(x,top,left,ch,cw,[h,w],I.BILINEAR,antialias=True)
        if family=='drop_keep':
            keep=torch.rand((1,*x.shape[-2:]))<severity
            return torch.where(keep,x,.5)
        raise ValueError(family)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={'settings':SETTINGS,'split':'test','n':1000,
          'purpose':'exploratory severity sweep; test data reused; not confirmatory setting selection',
          'rotation':'uniform[-angle,+angle], nearest, gray fill0.5, RGB8 truncation',
          'crop':'retained AREA fraction, random square crop, bilinear antialias resize, tensor',
          'drop_keep':'fraction pixels KEPT, rest gray0.5, shared RGB mask, tensor',
          'seed':'20261029+i rotation;20261030+i crop/drop; identical draws before/after',
          'code_sha256':sha256(Path(__file__)),
          'key_metrics_sha256':sha256(ROOT/'key_metrics.py'),
          'carrier_sha256':sha256(RUN/'carrier/complete.json')}
        # JSON round trip normalizes tuples for freeze comparisons on resume.
        freeze(OUT/'protocol.json',json.loads(json.dumps(identity)))
        rows=read_json(RUN/'prompts.json')['test'];assert len(rows)==1000
        ext=extractor(RUN);carrier=SemanticCarrier.load(RUN/'carrier','cuda')
        labels=['teacher_wm_test','before_test']
        # Uint8 cache avoids rereading PNGs for each severity, ~0.75GiB/teacher.
        from PIL import Image
        import numpy as np
        cache={}
        for label in labels:
            folder=RUN/'images'/label
            entries=read_json(folder/'image_hashes.json')
            assert len(entries)==len(rows)
            data=[]
            for i,r in enumerate(entries):
                path=folder/r['file_name'];assert sha256(path)==r['sha256']
                assert r['file_name']==image_name(i)
                with Image.open(path) as im:
                    data.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1))
            cache[label]=torch.stack(data)
        summary={'protocol':identity,'results':{}}
        anchors={'rotation':90,'crop':.75,'drop_keep':.64}
        aliases={'rotation':'rotation','crop':'crop_scale','drop_keep':'random_drop'}
        for family,severity in SETTINGS:
            setting=f'{family}_{severity:g}';summary['results'][setting]={}
            for label in labels:
                status={'stage':'running','family':family,'severity':severity,'label':label,
                        'updated':time.time(),'pid':os.getpid()}
                write_json(OUT/'status.json',status)
                dest=OUT/f'{label}_{setting}.json'
                if dest.exists():record=read_json(dest)
                else:
                    anchor=RUN/'result/test'/f'{label}_{aliases[family]}.json'
                    if severity==anchors[family] and anchor.exists():
                        q=read_json(anchor)['coordinates'];source='existing exact-setting evaluation'
                    else:
                        q=[];source='new evaluation'
                        for start in range(0,len(rows),8):
                            ids=range(start,min(start+8,len(rows)))
                            batch=torch.stack([transform(cache[label][i].float()/255,
                                  family,severity,i)*2-1 for i in ids]).cuda()
                            with torch.inference_mode():
                                coords=carrier.coordinates(ext.encode_image(batch,[rows[i]['text'] for i in ids]))
                            q.extend(coords.cpu().tolist())
                    record={'family':family,'severity':severity,'label':label,'source':source,
                            'coordinates':q,'metrics':aggregate_metrics(q,carrier.signs.cpu().tolist())}
                    freeze(dest,record)
                summary['results'][setting][label]=record['metrics']
                print(setting,label,json.dumps(record['metrics']['rows']),flush=True)
                write_json(OUT/'summary_progress.json',summary)
        freeze(OUT/'summary.json',summary)
        write_json(OUT/'status.json',{'stage':'completed','updated':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'status.json',{'stage':'failed','error':repr(e),'updated':time.time()})
        raise

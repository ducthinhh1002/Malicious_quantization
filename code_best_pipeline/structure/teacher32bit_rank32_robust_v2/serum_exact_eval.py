"""Exact official SERUM Augment, paired teachers, one label per GPU worker."""
import bootstrap
from pathlib import Path
import argparse,json,random,types,time,fcntl
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from common import read_json,write_json,freeze,sha256,image_name
from semantic_teacher import extractor
from semantic_carrier import SemanticCarrier
from key_metrics import aggregate_metrics

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run';OUT=RUN/'result/serum_exact'
VENDOR=Path('/var/lib/structure_teacher_robustness/vendor/serum_augmentation.py')
ATTACKS=[('rotation',0),('crop75',2),('drop64',3),('gaussian01',6),('brightness6',7),('blur15',4)]

def main(label):
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/(label+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        vendor_hash=sha256(VENDOR)
        assert vendor_hash=='c4d93a164c22c893cb0493d80bb41e64e2c371e3ab6e449967dd438fd425c917'
        module=types.ModuleType('_serum_official')
        exec(compile(VENDOR.read_text().replace('from ..utils import utils','').replace('from kornia.filters import median_blur',''),str(VENDOR),'exec'),module.__dict__)
        augment=module.Augment()
        prompts=read_json(RUN/'prompts.json')['test'];assert len(prompts)==1000
        folder=RUN/'images'/label;hashes=read_json(folder/'image_hashes.json')
        freeze(OUT/(label+'_protocol.json'),{'label':label,'vendor_sha256':vendor_hash,
            'code_sha256':sha256(Path(__file__)),'image_manifest_sha256':sha256(folder/'image_hashes.json'),
            'carrier_sha256':sha256(RUN/'carrier/complete.json'),
            'prompt_sha256':sha256(RUN/'prompts.json'),'n':1000,
            'attacks':ATTACKS,'excluded':['JPEG','salt_pepper'],'supplemental':['horizontal_flip'],
            'seed':'20261029+i; reset python/numpy/torch CPU per image before Augment',
            'pipeline':'official Augment(x,idx) including its own RGB8 postprocess; no extra re-quantization',
            'selection':'fixed two teachers; all requested transforms reported'})
        cached=[]
        for i,row in enumerate(hashes):
            f=folder/row['file_name'];assert sha256(f)==row['sha256'] and row['file_name']==image_name(i)
            with Image.open(f) as im:cached.append(TF.to_tensor(im.convert('RGB')))
        # Source image cache is uint8, so the model sees official ToTensor input.
        cached=[(x*255).round().to(torch.uint8) for x in cached]
        ext=extractor(RUN);carrier=SemanticCarrier.load(RUN/'carrier','cuda')
        results={}
        for name in ['clean','horizontal_flip']:
            r=read_json(RUN/'result/test'/f'{label}_{name}.json')
            results[name]=r['metrics']
        for name,idx in ATTACKS:
            dest=OUT/f'{label}_{name}.json'
            if dest.exists():record=read_json(dest)
            else:
                coords=[]
                for start in range(0,1000,16):
                    batch=[];ids=list(range(start,min(start+16,1000)))
                    for i in ids:
                        seed=20261029+i
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        x=cached[i].float().div(255).mul(2).sub(1).unsqueeze(0)
                        batch.append(augment(x,idx=idx)[0])
                    with torch.inference_mode():
                        q=carrier.coordinates(ext.encode_image(torch.stack(batch).cuda(),[prompts[i]['text'] for i in ids]))
                    coords.extend(q.cpu().tolist())
                    write_json(OUT/(label+'_status.json'),{'stage':'running','attack':name,'images':len(coords),'updated':time.time()})
                record={'coordinates':coords,'metrics':aggregate_metrics(coords,carrier.signs.cpu().tolist())}
                freeze(dest,record)
            results[name]=record['metrics']
            write_json(OUT/(label+'_summary_progress.json'),results)
            print(name,json.dumps(record['metrics']['rows']),flush=True)
        freeze(OUT/(label+'_summary.json'),results)
        write_json(OUT/(label+'_status.json'),{'stage':'completed','updated':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=['before_test','teacher_wm_test'],required=True)
    a=p.parse_args()
    try:main(a.label)
    except Exception as e:
        write_json(OUT/(a.label+'_status.json'),{'stage':'failed','error':repr(e),'updated':time.time()});raise

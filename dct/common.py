"""Standalone public-baseline utilities; no private teacher refinement imports."""
from pathlib import Path
import os,json,hashlib,time,contextlib,logging,sys
REPO=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('BASELINE_WORK',REPO/'work')).resolve()
CACHE=Path(os.environ.get('BASELINE_CACHE',REPO/'.cache')).resolve()
METHODS={'stable_signature':'Stable Signature','gaussmarker':'GaussMarker','trustmark':'TrustMark','grow':'GROW','serum':'SERUM'}
STUDENTS={'sd14':'SD1.4','sana600m':'SANA-0.6B','pixart512':'PixArt-α 512'}
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def freeze(p,value):
    p=Path(p)
    if p.exists():
        if read(p)!=value:raise RuntimeError('Frozen experiment identity changed: '+str(p))
    else:write(p,value)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def config():return read(ROOT/'config.json') if (ROOT/'config.json').exists() else read(REPO/'config/default.json')
def image_name(i):return f'sample_{i:05d}.png'
def status(p,stage,**kw):write(Path(p)/'status.json',{'stage':stage,'time':time.time(),'pid':os.getpid(),**kw})
def policy():
    import torch
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def pixels(paths):
    import torch,numpy as np
    from PIL import Image
    return torch.stack([torch.from_numpy(np.array(Image.open(p).convert('RGB'),copy=True)).permute(2,0,1).float()/255 for p in paths])
def save_images(images,folder,indices):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for image,i in zip(images,indices):
        p=folder/image_name(i);tmp=p.with_suffix('.partial.png');image.save(tmp);tmp.replace(p)
def manifest(folder,rows):
    folder=Path(folder);items=[{'file_name':image_name(i),'text':r['text'],'sha256':sha(folder/image_name(i))} for i,r in enumerate(rows)]
    freeze(folder/'image_hashes.json',items)
    with (folder/'metadata.jsonl').open('w') as f:
        for item in items:f.write(json.dumps({k:item[k] for k in ['file_name','text']})+'\n')
    return items
def upstream(name):
    p=CACHE/'upstream'/name
    if not p.exists():raise RuntimeError('Run download.py before loading '+name)
    return p
@contextlib.contextmanager
def quiet(log):
    """Native verbose output goes to file; completion lines stay on console."""
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with open(log,'a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):yield
def code_identity():
    return {p.name:sha(p) for p in REPO.glob('*.py')}

"""Fixed per-image manipulated training targets in [0,1], before Normalize.

Tensor channels remain floating point. Only rotation/brightness/blur cross
the declared RGB8 channel. RNG is forked so denoising noise is not changed.
"""
from pathlib import Path
import hashlib,json
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F,InterpolationMode as I

SETTINGS={
 'flip':{'operation':'horizontal_flip','channel':'tensor'},
 'rotation15':{'operation':'rotation','degrees':[-15,15],'fill':.5,'interpolation':'nearest','channel':'rgb8'},
 'crop75':{'operation':'crop','retained_area':.75,'ratio':1,'interpolation':'bilinear','antialias':True,'channel':'tensor'},
 'drop96':{'operation':'drop','keep_probability':.96,'fill':.5,'shared_rgb_mask':True,'channel':'tensor'},
 'gaussian005':{'operation':'gaussian','sigma':.05,'bounds':[0,1],'channel':'tensor'},
 'brightness07':{'operation':'brightness','factor':.7,'channel':'rgb8'},
 'blur15':{'operation':'blur','kernel':15,'sigma':[.1,2.],'channel':'rgb8'},
}

def pixel_digest(x):
    return hashlib.sha256((x*255).round().to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()

def transform(x,name,index):
    assert x.device.type=='cpu'
    spec=SETTINGS[name]
    with torch.random.fork_rng(devices=[]):
        seed=(20261029 if name in ['rotation15','blur15'] else 20261030)+index
        torch.random.default_generator.manual_seed(seed)
        if name=='flip':y=F.hflip(x)
        elif name=='rotation15':y=F.rotate(x,T.RandomRotation.get_params([-15,15]),interpolation=I.NEAREST,fill=.5)
        elif name=='crop75':
            h,w=x.shape[-2:];ch,cw=round(h*.75**.5),round(w*.75**.5)
            top=int(torch.randint(h-ch+1,()).item());left=int(torch.randint(w-cw+1,()).item())
            y=F.resized_crop(x,top,left,ch,cw,[h,w],I.BILINEAR,antialias=True)
        elif name=='drop96':
            keep=torch.rand((1,*x.shape[-2:]))<.96;y=torch.where(keep,x,.5)
        elif name=='gaussian005':y=(x+torch.randn_like(x)*.05).clamp(0,1)
        elif name=='brightness07':y=F.adjust_brightness(x,.7)
        elif name=='blur15':y=F.gaussian_blur(x,[15,15],T.GaussianBlur.get_params(.1,2.))
        else:raise ValueError(name)
        if spec['channel']=='rgb8':y=(y.clamp(0,1)*255).to(torch.uint8).float()/255
    return y

class ManipulatedTrainingImage:
    def __init__(self,root):
        self.root=Path(root)
        self.spec=json.loads((self.root/'manipulation.json').read_text())
        self.name=self.spec['name'];assert self.spec['setting']==SETTINGS[self.name]
        self.lookup=json.loads((self.root/'pixel_index.json').read_text())
        self.calls=0
        print('TRAINING_MANIPULATION',json.dumps(self.spec),flush=True)

    def __call__(self,x):
        digest=pixel_digest(x)
        if digest not in self.lookup:raise RuntimeError('Unexpected training image; manipulation mapping missing')
        i=self.lookup[digest]
        y=transform(x,self.name,i)
        if not torch.isfinite(y).all():raise RuntimeError('Invalid manipulated training target')
        self.calls+=1
        if self.calls==1 or self.calls%5000==0:
            print('MANIPULATION_APPLIED',json.dumps({'name':self.name,'calls':self.calls,'index':i,
                'mean_absolute_change':float((x-y).abs().mean()),'tensor_target':True}),flush=True)
        return y

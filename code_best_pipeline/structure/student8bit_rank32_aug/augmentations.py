"""Explicit exploratory attack settings; CPU torch RNG is saved by the trainer."""
import torch
from torchvision.transforms import functional as F, InterpolationMode

ATTACKS=('rotation','crop_scale','random_drop','salt_pepper','gaussian_noise','brightness','gaussian_blur')
CONFIG={'version':1,'clean_probability':.5,'one_attack_per_augmented_image':True,
        'rotation_degrees':[-15,15],'crop_area':.75,'drop_keep_probability':.64,
        'salt_pepper_fraction':.05,'noise_std_0_1':.1,
        'brightness_factor':[.7,1.3],'blur_sigma':[.5,1.5],
        'evaluation_rotation_degrees':15,'evaluation_brightness_factor':.7,
        'evaluation_blur_sigma':1.5,'jpeg':False}

def uniform(a,b):return a+(b-a)*torch.rand(()).item()

def apply_attack(x,name,training=False):
    if name=='clean':return x
    if name=='rotation':
        x=F.rotate(x,uniform(-15,15) if training else 15,
                   interpolation=InterpolationMode.BILINEAR,fill=.5)
    elif name=='crop_scale':
        h,w=x.shape[-2:];ch,cw=round(h*.75**.5),round(w*.75**.5)
        top=int(torch.randint(h-ch+1,()).item());left=int(torch.randint(w-cw+1,()).item())
        x=F.resized_crop(x,top,left,ch,cw,[h,w],InterpolationMode.BILINEAR,antialias=True)
    elif name=='random_drop':
        keep=torch.rand((1,*x.shape[-2:]))<.64
        x=torch.where(keep,x,torch.full_like(x,.5))
    elif name=='salt_pepper':
        mask=torch.rand((1,*x.shape[-2:]));x=torch.where(mask<.025,0.,torch.where(mask>.975,1.,x))
    elif name=='gaussian_noise':x=x+torch.randn_like(x)*.1
    elif name=='brightness':x=F.adjust_brightness(x,uniform(.7,1.3) if training else .7)
    elif name=='gaussian_blur':x=F.gaussian_blur(x,[9,9],uniform(.5,1.5) if training else 1.5)
    else:raise ValueError(name)
    return x.clamp(0,1)

class WatermarkAugment:
    def __call__(self,x):
        if torch.rand(()).item()<.5:return x
        return apply_attack(x,ATTACKS[int(torch.randint(len(ATTACKS),()).item())],training=True)

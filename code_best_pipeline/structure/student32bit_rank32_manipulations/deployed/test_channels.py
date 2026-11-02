from pathlib import Path
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F,InterpolationMode as I
from channels import transform,SETTINGS

def main():
    torch.set_num_threads(2);x=torch.rand(3,512,512)
    for name in SETTINGS:
        state=torch.get_rng_state().clone();y=transform(x,name,7)
        assert torch.equal(state,torch.get_rng_state()) and torch.equal(y,transform(x,name,7))
        assert y.shape==x.shape and y.dtype==torch.float32 and torch.isfinite(y).all()
        assert y.min()>=0 and y.max()<=1
    zero=torch.zeros_like(x)
    dropped=transform(zero,'drop96',7)
    assert .038<float((dropped[0]==.5).float().mean())<.042
    assert torch.equal(dropped[0],dropped[1])
    n=transform(torch.full_like(x,.5),'gaussian005',7)-.5
    assert abs(float(n.std())-.05)<.001
    # Historical best-table training channels: exact transform parity.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261029+7)
        y=F.rotate(x,T.RandomRotation.get_params([-15,15]),interpolation=I.NEAREST,fill=.5)
        y=(y*255).to(torch.uint8).float()/255
    assert torch.equal(y,transform(x,'rotation15',7))
    assert torch.equal(transform(x,'flip',7),F.hflip(x))
    # Tensor Gaussian values must retain more than RGB8 precision.
    y=transform(x,'gaussian005',7)
    assert float(((y*255)-(y*255).round()).abs().mean())>.1
    print('ALL_CHANNEL_TESTS_PASSED',flush=True)

if __name__=='__main__':main()

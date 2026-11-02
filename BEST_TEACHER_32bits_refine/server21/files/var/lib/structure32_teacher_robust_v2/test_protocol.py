"""Numerical checks for the scientific channel and gradient/metric semantics."""
import json
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from attacks import ATTACKS, CONFIG, apply_attack, eval_attack, quantize, training_pair
from robust_objectives import bit_loss, pcgrad_pair
from key_metrics import aggregate_metrics

def main():
    torch.set_num_threads(2)
    x=torch.rand(3,64,64)
    # Exact PIL truncation parity, including endpoints.
    x[:,0,0]=1.;x[:,0,1]=0.
    torch.testing.assert_close(quantize(x),F.to_tensor(F.to_pil_image(x)),rtol=0,atol=0)
    for name in ATTACKS:
        before=torch.get_rng_state().clone()
        a=eval_attack(x,name,3);b=eval_attack(x,name,3)
        assert torch.equal(a,b) and torch.equal(before,torch.get_rng_state())
        assert a.shape==x.shape and torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
        leaf=x.clone().requires_grad_(True)
        apply_attack(leaf,name,training=True).square().mean().backward()
        assert torch.isfinite(leaf.grad).all() and leaf.grad.abs().sum()>0,name
    # Matched SERUM rows retain original torch transform + PIL channel.
    for name,transform in [('rotation',T.RandomRotation(90,fill=.5)),
                           ('gaussian_blur',T.GaussianBlur(15))]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONFIG[name]['seed_base']+3)
            expected=F.to_tensor(F.to_pil_image(transform(x)))
        torch.testing.assert_close(eval_attack(x,name,3),expected,atol=0,rtol=0)
    # Drop means keep64, Gaussian means clamp rather than modulo.
    y=eval_attack(torch.zeros(3,512,512),'random_drop',8)
    assert .35<float((y[0]==.5).float().mean())<.37
    noise=eval_attack(torch.zeros(3,512,512),'gaussian_noise',8)
    assert .048<float(noise.std())<.07 and noise.max()<1
    # Hard-bit loss acts on a weak bit even if the mean score looks positive.
    signed=torch.ones(1,32,requires_grad=True)
    with torch.no_grad():signed[0,0]=-.2
    bit_loss(signed).sum().backward()
    assert signed.grad[0,0]<0 and torch.equal(signed.grad[0,1:],torch.zeros(31))
    a=torch.tensor([[[[1.,0.]]]]);b=torch.tensor([[[[-1.,1.]]]])
    merged,stats=pcgrad_pair(a,b)
    assert stats['conflict_fraction']==1 and (merged*a).sum()>=0 and (merged*b).sum()>=0
    # Detach/merge/chain-rule implementation matches direct mean gradient when
    # the two views agree, so teacher gradients cannot silently disappear.
    p=torch.tensor(.3,requires_grad=True);image=p*torch.ones(1,3,4,4)
    expected=torch.autograd.grad((image.square().sum()+2*image.square().sum())/2,p,retain_graph=True)[0]
    leaf=image.detach().requires_grad_(True)
    g1=torch.autograd.grad(leaf.square().sum(),leaf,retain_graph=True)[0]
    g2=torch.autograd.grad(2*leaf.square().sum(),leaf)[0]
    combined,_=pcgrad_pair(g1,g2)
    actual=torch.autograd.grad(image,p,grad_outputs=combined)[0]
    torch.testing.assert_close(actual,expected)
    q=np.ones((256,32));q[:,:8]=-1
    m=aggregate_metrics(q,[1]*32)
    assert all(r['key_acceptance_ge24_rate']==1 and r['exact_full_key_rate']==0 for r in m['rows'])
    assert abs(m['theoretical_fpr_ge24']-.0035001833457499743)<1e-14
    assert set(training_pair(i,0)[0] for i in range(20))=={'rotation','crop_scale','horizontal_flip'}
    print(json.dumps({'passed':True,'checks':['PIL parity','historical rotation/blur parity',
         'deterministic paired RNG','all attack gradients','drop/noise semantics',
         'weak-bit gradient','PCGrad conflict','teacher chain rule','exact vs statistical FK']}))

if __name__=='__main__':main()

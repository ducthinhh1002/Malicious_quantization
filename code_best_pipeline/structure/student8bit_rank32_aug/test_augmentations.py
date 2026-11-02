"""CPU checks for corruption strength, range, RNG replay and trainer integration."""
import torch
from augmentations import ATTACKS,apply_attack,WatermarkAugment
from driver import patched_source
from bootstrap import HARNESS
def main():
    x=torch.full((3,512,512),.5)
    for name in ATTACKS:
        torch.manual_seed(17);state=torch.get_rng_state();a=apply_attack(x,name,True)
        torch.set_rng_state(state);b=apply_attack(x,name,True)
        assert torch.equal(a,b),name
        assert a.shape==x.shape and a.isfinite().all() and a.min()>=0 and a.max()<=1,name
    x=torch.full_like(x,.25)
    torch.manual_seed(17);a=apply_attack(x,'random_drop');assert abs(float((a[0]==.5).float().mean())-.36)<.005
    a=apply_attack(x,'salt_pepper');assert abs(float(((a[0]==0)|(a[0]==1)).float().mean())-.05)<.003
    x=torch.full_like(x,.5);a=apply_attack(x,'gaussian_noise');assert abs(float((a-x).std())-.1)<.002
    aug=WatermarkAugment();torch.manual_seed(31);state=torch.get_rng_state();a=[aug(x) for _ in range(12)]
    torch.set_rng_state(state);b=[aug(x) for _ in range(12)];assert all(torch.equal(i,j) for i,j in zip(a,b))
    source=patched_source((HARNESS/'vendor/diffusers/train_text_to_image_lora.py').read_text())
    assert source.count('            WatermarkAugment(),')==1
    assert source.index('            WatermarkAugment(),')<source.index('            transforms.Normalize([0.5], [0.5]),')
    compile(source,'pinned_trainer_augmented','exec')
    print('PASS: attack strength, tensor bounds, exact RNG replay, pinned trainer insertion')
if __name__=='__main__':main()

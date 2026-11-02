"""Select the latest fully readable locally-produced Accelerate checkpoint."""
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

root=Path(sys.argv[1])
for c in sorted(root.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True):
    try:
        weights=load_file(str(c/'pytorch_lora_weights.safetensors'),device='cpu')
        assert weights and all(torch.isfinite(t).all() for t in weights.values())
        for name in ['optimizer.bin','scheduler.bin','scaler.pt','random_states_0.pkl']:
            torch.load(c/name,map_location='cpu',weights_only=False)
        print(c.name)
        break
    except Exception as exc:
        print(f'Cannot resume {c}: {type(exc).__name__}',file=sys.stderr)
else:
    print('')

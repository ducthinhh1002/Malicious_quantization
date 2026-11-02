import os,json,math
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import torch,torch_fidelity
import numpy as np
torch.serialization.add_safe_globals([
    np.ndarray, np.dtype, type(np.dtype('float64')), type(np.dtype('float32')),
    np._core.multiarray._reconstruct, np._core.multiarray.scalar,
])
torch.set_num_threads(4)
r=Path('/root/workspace/serum')
result=torch_fidelity.calculate_metrics(input1=str(r/'data/utility_preflight/clean'),input2=str(r/'data/utility_preflight/wm'),
    cuda=True,fid=True,kid=False,isc=False,batch_size=8,
    input1_cache_name='preflight_clean',input2_cache_name='preflight_wm',cache_root=str(r/'data/utility_feature_cache'))
assert all(math.isfinite(float(v)) for v in result.values())
(r/'metrics/fid_offline_preflight.json').write_text(json.dumps({'ok':True,'scope':'FID execution smoke check only, 16-image sets; not a reported research result'},indent=2))
print('Offline FID execution passed',flush=True)

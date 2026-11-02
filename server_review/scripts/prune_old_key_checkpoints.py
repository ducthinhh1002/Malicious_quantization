"""Remove obsolete old-key checkpoints; retain historical selected adapters."""
import json
import shutil
import time
from pathlib import Path

R=Path('/root/workspace/serum').resolve()
targets=[];retained=[]
for name in ['C10_seed0','C11_step0.25_seed0','C11_step1.00_seed0']:
    folder=R/'students'/name
    best=Path((folder/'BEST_LORA_PATH.txt').read_text().strip()).resolve()
    assert best==folder or best.parent==folder,(name,best)
    assert (best/'pytorch_lora_weights.safetensors').is_file()
    retained.append(str(best))
    targets.extend(p for p in folder.glob('checkpoint-*') if p.resolve()!=best)
# The abandoned corrected old-key C10 has no evaluation result to preserve.
targets.extend((R/'students/C10_seed0_clipfix').glob('checkpoint-*'))
legacy=R/'third_party/SERUM/results/SERUM_sd21/checkpoints'
targets.extend(p for p in legacy.glob('checkpoint_epoch_*.pt') if p.name!='checkpoint_epoch_50.pt')
retained.extend([str(legacy/'checkpoint_epoch_50.pt'),str(legacy/'latest_checkpoint.pt')])
allowed_roots=[R/'students'/name for name in ['C10_seed0','C11_step0.25_seed0','C11_step1.00_seed0','C10_seed0_clipfix']]+[legacy]
for p in targets:
    assert not p.is_symlink() and any(p.resolve().is_relative_to(base) for base in allowed_roots),str(p)
    assert '_author' not in str(p) and 'C00' not in str(p) and 'C01' not in str(p)
resume=json.loads((R/'metrics/serum_author/C00_resume_provenance.json').read_text())
assert (Path(resume['resume_checkpoint'])/'optimizer.bin').is_file()
assert (R/'assets/serum_author/latest_checkpoint.pt').is_file()
before=shutil.disk_usage(R).free
record={'removed_paths':[str(p) for p in targets],'retained_historical_selected':retained,
    'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':'validated_pending_delete',
    'scope':'obsolete old-key C10/C11 intermediate checkpoints and old SERUM epochs below50; keep historical best adapters and legacy final detector checkpoint'}
manifest=R/'manifests/pruned_old_key_checkpoints.json';manifest.write_text(json.dumps(record,indent=2))
for p in targets:
    if p.is_dir():shutil.rmtree(p)
    else:p.unlink()
assert all(not p.exists() for p in targets)
assert all(Path(p).exists() for p in retained)
assert (Path(resume['resume_checkpoint'])/'optimizer.bin').is_file()
record.update(status='deleted',removed_count=len(targets),disk_free_increase_bytes=shutil.disk_usage(R).free-before)
manifest.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='removed_paths'},indent=2))

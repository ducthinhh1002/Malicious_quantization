"""Delete the 11k retired old-key teacher images and their hard-link aliases."""
import json
import os
import shutil
import time
from pathlib import Path

R=Path('/root/workspace/serum').resolve()
targets={R/'data/raw/wm_train':5000,R/'data/raw/wm_val':1000,R/'data/raw/calib_wm_unused':5000}
identities=set();unique_bytes=0;counts={}
for folder,n in targets.items():
    assert not folder.is_symlink() and folder.resolve().is_relative_to(R/'data/raw')
    files=sorted(folder.glob('*.png'));assert len(files)==n,(folder,len(files))
    counts[str(folder)]=n
    for p in files:
        assert not p.is_symlink()
        st=p.stat();identity=(st.st_dev,st.st_ino)
        if identity not in identities:unique_bytes+=st.st_size
        identities.add(identity)
assert len(identities)==11000
aliases=[]
for base,dirs,files in os.walk(R/'data',followlinks=False):
    for name in files:
        if not name.endswith('.png'):continue
        p=Path(base)/name
        if p.is_symlink():continue
        st=p.stat()
        if (st.st_dev,st.st_ino) in identities and p.parent not in targets:
            assert p.resolve().is_relative_to(R/'data')
            assert not p.resolve().is_relative_to(R/'data/serum_author'),str(p)
            # Identity proves this is another directory entry for the same retired image,
            # including legacy pilot subsets, imagefolder training and generation shards.
            aliases.append(p)
before=shutil.disk_usage(R).free
record={'requested_images':11000,'raw_directories':counts,'hard_link_aliases':[str(p) for p in aliases],
    'unique_image_bytes':unique_bytes,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'authorization':'user requested deletion of the retired 11000 old-key watermarked teacher images',
    'status':'validated_pending_delete'}
manifest=R/'manifests/deleted_old_watermarked_teacher.json';manifest.write_text(json.dumps(record,indent=2))
for p in aliases:p.unlink()
for folder in targets:shutil.rmtree(folder)
assert all(not p.exists() for p in targets)
assert all(not p.exists() for p in aliases)
record.update(status='deleted',hard_link_alias_count=len(aliases),disk_free_increase_bytes=shutil.disk_usage(R).free-before)
manifest.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='hard_link_aliases'},indent=2),flush=True)

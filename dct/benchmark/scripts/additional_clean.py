from pathlib import Path
import json,hashlib
from benchmark.scripts.ood_clean_20261111 import read,sha

def verify_images(folder,rows):
 if not (folder/'image_hashes.json').exists():return False
 entries=read(folder/'image_hashes.json')
 if len(entries)!=len(rows):raise ValueError('Incomplete image manifest: '+str(folder))
 for i,(entry,row) in enumerate(zip(entries,rows)):
  if entry['file_name']!=f'sample_{i:05d}.png' or entry['text']!=row['text'] or sha(folder/entry['file_name'])!=entry['sha256']:
   raise ValueError('Image/caption hash mismatch: '+str(folder))
 return True

def verify_checkpoint(path,identity):
 # Extract the existing CPU integrity functions without importing helper/common.py.
 from benchmark.shared.student_training.integration import function,REF
 ns={'Path':Path,'hashlib':hashlib,'json':json}
 exec(function(REF/'pipeline/student/resume_state.py','digest'),ns)
 exec(function(REF/'pipeline/student/resume_state.py','verify_checkpoint'),ns)
 return ns['verify_checkpoint'](path,identity)

"""Stop only the identified project queue group, preserving readable manifests."""
import os,signal,time,csv,io,json
from pathlib import Path
r=Path('/root/workspace/serum')
matches=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        if b'scripts/queue_core_seed0.py' in args and b'--worker' not in args and (proc/'cwd').resolve()==r:
            matches.append(int(proc.name))
    except (OSError,ValueError): pass
assert len(matches)==1,f'Expected one current queue, found {matches}'
pgid=os.getpgid(matches[0]); assert pgid!=os.getpgrp()
backups=[]
for manifest in (r/'data/laundered').glob('*/laundering_manifest.csv'):
    for attempt in range(20):
        content=manifest.read_text()
        try:
            rows=list(csv.DictReader(io.StringIO(content)))
            assert rows and all(int(row['index'])==i and row['source'] and row['output'] and row['seed'] for i,row in enumerate(rows))
            assert content.endswith('\n')
            backup=manifest.with_suffix('.csv.pre_hardening.bak'); backup.write_text(content)
            backups.append((manifest,backup))
            break
        except (ValueError,KeyError,AssertionError): time.sleep(.1)
    else: raise RuntimeError(f'Could not snapshot {manifest}')
os.killpg(pgid,signal.SIGINT)
time.sleep(5)
try: os.killpg(pgid,signal.SIGTERM)
except ProcessLookupError: pass
time.sleep(2)
for manifest,backup in backups:
    temporary=manifest.with_suffix('.restore.tmp'); temporary.write_bytes(backup.read_bytes()); os.replace(temporary,manifest)
(r/'metrics/core_handoff.json').write_text(json.dumps({'old_queue_pid':matches[0],'old_pgid':pgid,'manifest_backups':len(backups)},indent=2))
print('Stopped the owned queue group and restored verified manifest snapshots',flush=True)

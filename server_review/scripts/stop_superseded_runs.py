"""Stop only explicitly identified quant experiment process groups."""
import json,os,signal,subprocess,time
from pathlib import Path
R=Path('/root/workspace/serum')
groups={30445:'run_c10_clipfix.py',44231:'queue_clipfix_strong.py',44236:'queue_clipfix_strong.py',44237:'queue_clipfix_strong.py',56538:'queue_author_serum.py'}
rows=subprocess.check_output(['ps','-eo','pid=,pgid=,args='],text=True).splitlines()
for pgid,marker in groups.items():
    members=[line for line in rows if len(line.split(None,2))==3 and int(line.split(None,2)[1])==pgid]
    if members:
        assert any(marker in line for line in members),(pgid,members)
        assert all('/root/workspace/serum/' in line or marker in line or 'train_text_to_image_lora.py' in line for line in members),(pgid,members)
        os.killpg(pgid,signal.SIGSTOP)
for pgid in groups:
    try:os.killpg(pgid,signal.SIGTERM);os.killpg(pgid,signal.SIGCONT)
    except ProcessLookupError:pass
time.sleep(3)
for pgid in groups:
    try:os.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError:pass
for session in ['quant-c10-clipfix','quant-clipfix-strong','quant-author-queue']:
    subprocess.run(['tmux','kill-session','-t',session],capture_output=True)
for name in ['C10_seed0_clipfix_status.json','C00_seed0_clipfix_status.json','clipfix_strong_queue_status.json','clipfix_strong_worker0.json','clipfix_strong_worker1.json']:
    p=R/'metrics'/name
    if p.exists():
        obj=json.loads(p.read_text());obj.update(state='cancelled',reason='User switched to author SD2.1 artifacts',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));p.write_text(json.dumps(obj,indent=2))
print('Superseded quant queues stopped. Assets/results retained.',flush=True)

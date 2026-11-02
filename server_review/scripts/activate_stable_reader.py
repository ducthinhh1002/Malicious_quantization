"""Activate a verified continuation only after the old teacher has exited."""
import json
from pathlib import Path
import shlex
import paramiko

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'server_review/lalt_stable_reader'
record=json.loads((folder/'deployment.json').read_text())
client=paramiko.SSHClient()
client.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
try:
    script=f'''
import json, os
from pathlib import Path
from datetime import datetime, timezone
old=Path('/root/logs/lalt_20261018_053126Z')
active=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try:
        if (path/'cwd').resolve()==old: active.append(int(path.name))
    except (OSError,RuntimeError): pass
if active: raise RuntimeError('Old teacher still active: '+str(active))
output=Path({record['output']!r})
check=json.loads((output/'stability_gpu_check.json').read_text())
if not check['passed'] or not check['reader_unchanged']: raise RuntimeError('GPU verification failed')
if json.loads((output/'status.json').read_text())['status']!='prepared': raise RuntimeError('Already launched')
parent=Path({record['parent']!r})
(parent/'superseded.json').write_text(json.dumps(dict(replacement=str(output),
    reason='User requested reader stabilization; validated reader 2000 frozen',
    last_saved_checkpoint_preserved=True, probe_unchanged=True,
    at_utc=datetime.now(timezone.utc).isoformat()),indent=2))
print(json.dumps(dict(old_teacher_exited=True,gpu_check=check)))
'''
    stdin,out,err=client.exec_command('/root/logs/gate_env/bin/python -',timeout=30)
    stdin.write(script);stdin.channel.shutdown_write()
    result,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status(): raise RuntimeError(result+error)
    print(result)
    _,out,err=client.exec_command('tmux new-session -d -s '+shlex.quote(record['session'])+' bash '+shlex.quote(record['script']),timeout=20)
    result,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status(): raise RuntimeError(result+error)
    record['phase']='training_launched'
    record['old_teacher_stopped']=True
    (folder/'deployment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    with client.open_sftp() as sftp:
        for name in ['stability_gpu_check.json','initialization.json','config.json']:
            sftp.get(record['output']+'/'+name,str(folder/name))
    print('Launched: '+record['session'])
finally:
    client.close()

"""Fetch current semantic pilot state without touching running processes."""
from datetime import datetime,timezone
import json
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=sorted((ROOT/'server_review/semantic_relation').glob('semantic_relation_*'))[-1]
    deployment=json.loads((folder/'deployment.json').read_text())
    client=paramiko.SSHClient()
    client.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with client.open_sftp() as sftp:
            for phase,remote in [('smoke',deployment['smoke_reports']),('pilot',deployment['reports'])]:
                target=folder/phase; target.mkdir(exist_ok=True)
                status={}
                try:
                    files=sftp.listdir(remote)
                except FileNotFoundError:
                    result[phase]='not_started'; continue
                for name in files:
                    if name.endswith('.json') and not name.startswith('job_'):
                        with sftp.open(remote+'/'+name,'rb') as stream:
                            data=stream.read()
                        (target/name).write_bytes(data)
                        if name=='status.json': status=json.loads(data)
                result[phase]=status
                stage=status.get('stage')
                if stage:
                    try:
                        with sftp.open(remote+'/'+stage+'.log','rb') as stream:
                            stream.seek(max(0,stream.stat().st_size-2400))
                            result[phase]['log_tail']=stream.read().decode(errors='replace')
                    except FileNotFoundError:
                        pass
            with sftp.open(deployment['log'],'rb') as stream:
                stream.seek(max(0,stream.stat().st_size-2400))
                result['runner_tail']=stream.read().decode(errors='replace')
        (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:
        client.close()


if __name__=='__main__':
    main()

"""Download stage reports and adapter audits; never modify the remote run."""
from datetime import datetime,timezone
import json
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=sorted((ROOT/'server_review/signature_transfer').glob('signature_transfer_*'))[-1]
    d=json.loads((folder/'deployment.json').read_text())
    c=paramiko.SSHClient(); c.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        # The original SD2.1-student experiment was explicitly superseded on
        # the server. Follow its replacement instead of reporting stale progress.
        with c.open_sftp() as s:
            try:
                with s.open(d['reports']+'/status.json','rb') as f: current=json.loads(f.read())
                replacement=s.stat('/root/logs/results/signature_author_sd21_to_sd14_launch.json')
            except FileNotFoundError:
                current={}; replacement=None
        if current.get('status')=='superseded' and replacement is not None:
            c.close()
            from check_signature_author_transfer import main as check_replacement
            return check_replacement()
        with c.open_sftp() as s:
            for phase,remote in [('smoke',d['smoke_reports']),('pilot',d['reports'])]:
                target=folder/phase; target.mkdir(exist_ok=True)
                try: names=s.listdir(remote)
                except FileNotFoundError:
                    result[phase]='not_started'; continue
                state={}
                for name in names:
                    if name.endswith('.json') and not name.startswith('job_'):
                        with s.open(remote+'/'+name,'rb') as f: data=f.read()
                        (target/name).write_bytes(data)
                        if name=='status.json': state=json.loads(data)
                        if name=='teacher_preflight.json': result[phase+'_teacher_preflight']=json.loads(data)
                        if name=='results.json': result[phase+'_results']=json.loads(data)
                    elif name.startswith(('r4_','r16_')) and not name.endswith('.log'):
                        branch=target/name; branch.mkdir(exist_ok=True)
                        for file in ('student_marked.json','adapter_audit.json','inference_audit.json','student_protocol_marked.json'):
                            try:
                                with s.open(remote+'/'+name+'/'+file,'rb') as f: data=f.read()
                            except FileNotFoundError: continue
                            (branch/file).write_bytes(data)
                result[phase]=state
                if state.get('stage'):
                    try:
                        with s.open(remote+'/'+state['stage']+'.log','rb') as f:
                            f.seek(max(0,f.stat().st_size-2200)); state['log_tail']=f.read().decode(errors='replace')
                    except FileNotFoundError: pass
            with s.open(d['log'],'rb') as f:
                f.seek(max(0,f.stat().st_size-2800)); result['runner_tail']=f.read().decode(errors='replace')
        (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        display=dict(result)
        for key in ('smoke_results','pilot_results'):
            if key in display:
                display[key]=[dict(rank=r['rank'],scope=r['scope'],
                    marked_ba=r['empirical_nominal_1pct']['marked']['bit_accuracy'],
                    marked_tpr=r['empirical_nominal_1pct']['marked']['detection_rate'],
                    clean_fpr=r['empirical_nominal_1pct']['clean']['detection_rate'],auroc=r['auroc'])
                    for r in display[key]['rows']]
        print(json.dumps(display,indent=2))
    finally: c.close()


if __name__=='__main__': main()

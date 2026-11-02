"""Read the active author SD2.1 -> clean SD1.4 transfer run and save its reports."""
from datetime import datetime,timezone
import json
import re
from pathlib import Path,PurePosixPath
import paramiko

ROOT=Path(__file__).resolve().parents[2]
LAUNCH='/root/logs/results/signature_author_sd21_to_sd14_launch.json'


def main():
    c=paramiko.SSHClient();c.load_host_keys(str(ROOT/'output_artifacts/vm4_known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('171.232.252.207',username='root',key_filename=str(Path.home()/'.ssh/id_ed25519_vm4'),timeout=20)
    try:
        with c.open_sftp() as s:
            with s.open(LAUNCH,'rb') as f: launch=json.loads(f.read())
            remote=launch['run']
            if PurePosixPath(remote).parent!=PurePosixPath('/root/logs/results'):
                raise ValueError('Unexpected result directory')
            folder=ROOT/'server_review/signature_author_transfer'/PurePosixPath(remote).name
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
            report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),run=remote)
            names=s.listdir(remote)
            records={}
            for name in names:
                if name.endswith('.json') and not name.startswith('job_'):
                    with s.open(remote+'/'+name,'rb') as f: data=f.read()
                    (folder/name).write_bytes(data)
                    records[name]=json.loads(data)
            with s.open(remote+'/status.json','rb') as f: status=json.loads(f.read())
            (folder/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            report['status']=status
            report['protocol']=records.get('protocol.json',{})
            report['results']=records.get('results.json')
            report['student_branches']={}
            for name in names:
                if name.startswith(('r4_','r16_')) and '.' not in name:
                    files=s.listdir(remote+'/'+name)
                    report['student_branches'][name]=dict(training_complete='student_marked.json' in files,
                        query_complete='query.json' in files)
            stage=status.get('stage')
            stages=status.get('active_stages') or ([dict(stage=stage)] if stage else [])
            report['active_progress']={}
            for entry in stages:
                label=entry['stage']
                try:
                    with s.open(remote+'/'+label+'.log','rb') as f:
                        f.seek(max(0,f.stat().st_size-2200));tail=f.read().decode(errors='replace')
                    progress=dict(entry,log_tail=tail)
                    matches=re.findall(r'(\d+)/(\d+)\s*\[',tail)
                    if matches:
                        progress.update(step=int(matches[-1][0]),total_steps=int(matches[-1][1]))
                    report['active_progress'][label]=progress
                    if label==stage: report['log_tail']=tail
                except FileNotFoundError: pass
            key=records.get('protocol.json',{}).get('key','')
            report['datasets']={}
            for split in ('train','calibration','query'):
                item={}
                for condition in ('clean','marked'):
                    name=f'teacher_{condition}_{split}'
                    record=records.get(name+'.json')
                    if record:
                        scores=record.get('logits',[])
                        item[condition]=dict(pngs=len(record['entries']),manifest_complete=True)
                        if scores and key:
                            counts=[sum((v>0)==(b=='1') for v,b in zip(row,key)) for row in scores]
                            item[condition].update(bit_accuracy=sum(counts)/(len(counts)*len(key)),
                                nominal_1pct_detections=sum(n>=33 for n in counts),n=len(counts))
                    else:
                        try:
                            files=[f for f in s.listdir_attr(records['storage.json']['images']+'/'+name) if f.filename.endswith('.png')]
                        except FileNotFoundError: files=[]
                        item[condition]=dict(pngs=len(files),manifest_complete=False)
                        if len(files)>2:
                            elapsed=max(f.st_mtime for f in files)-min(f.st_mtime for f in files)
                            item[condition]['seconds_per_image_so_far']=elapsed/(len(files)-1)
                report['datasets'][split]=item
        pids={int(launch['pid'])}|{int(row['pid']) for row in status.get('active_stages',[]) if 'pid' in row}
        _,out,err=c.exec_command('ps -p '+','.join(str(pid) for pid in sorted(pids))+' -o pid,etime,stat,comm')
        report['process']=out.read().decode(); report['process_check_stderr']=err.read().decode()
        (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        display={k:v for k,v in report.items() if k!='protocol'}
        print(json.dumps(display,indent=2))
    finally:c.close()


if __name__=='__main__':main()

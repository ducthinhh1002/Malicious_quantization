"""Fetch immutable research-screen reports without touching training jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[2]


def main():
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / 'output_artifacts/vm4_known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('171.232.252.207', username='root',
        key_filename=str(Path.home() / '.ssh/id_ed25519_vm4'), timeout=20)
    reports = dict(checked_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with client.open_sftp() as sftp:
            for category in ('lalt_readout_audit', 'lalt_spectral_screen', 'lalt_spectral_strength'):
                folder = sorted((ROOT / 'server_review' / category).glob(category + '_*'))[-1]
                deployment = json.loads((folder / 'deployment.json').read_text())
                report = {}
                for name in ('status.json', 'protocol.json', 'summary.json', 'results.json', 'calibration_reuse.json'):
                    try:
                        with sftp.open(deployment['reports'] + '/' + name, 'rb') as stream:
                            data = stream.read()
                    except FileNotFoundError:
                        continue
                    (folder / name).write_bytes(data)
                    if name == 'status.json':
                        report['status'] = json.loads(data)
                    if name == 'results.json':
                        report['rows'] = [dict(angle=r['angle'], auc=r['signal']['auroc'],
                            tpr=r['signal']['marked']['detection_rate'], quality=r['quality'], passes=r['passes'])
                            for r in json.loads(data)['rows']]
                reports[category] = report
                (folder / 'verification.json').write_text(json.dumps(dict(
                    checked_at_utc=reports['checked_at_utc'], **report), indent=2), encoding='utf-8')
        print(json.dumps(reports, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()

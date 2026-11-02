"""Continue the authorized current job, omitting the cancelled margin ablation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker-pid', type=int, required=True)
    p.add_argument('--worker-start', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root))
    import structure_margin_campaign as campaign
    q = campaign.QUEUE
    with (q / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(**fields):
            campaign.write_json(q / 'status.json', dict(updated=now(), pid=os.getpid(),
                cancelled_jobs=['margin_native'], **fields))
        try:
            while True:
                proc = Path(f'/proc/{a.worker_pid}/stat')
                if not proc.exists():
                    break
                stat = proc.read_text().split(') ', 1)[1].split()
                if stat[0] == 'Z' or stat[19] != a.worker_start:
                    break
                status(state='running', job='structure_margin_native', stage='02_data',
                       inherited_worker_pid=a.worker_pid)
                time.sleep(20)
            run = q / 'structure_margin_native'
            if not (run / 'data_complete.json').exists():
                raise RuntimeError('Inherited data worker exited without data_complete.json; inspect its log')
            campaign.verify()
            campaign.job('structure_margin_native')
            status(state='completed', job='structure_margin_native', stage='16_structure_audit',
                   reason='Finished preserved job; margin_native cancelled by user')
        except Exception as exc:
            status(state='failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()

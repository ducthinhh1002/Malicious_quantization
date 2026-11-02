import argparse
from pathlib import Path
import bootstrap
from common import read_json,freeze
from acceptance_metrics import aggregate_metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--branch',required=True)
    a=p.parse_args()
    rows=read_json(a.run/'scores'/f'{a.branch}.json')
    key=read_json(a.run/'carrier/key.json')['signs'];assert len(key)==32
    freeze(a.run/'result/student_acceptance32.json',{
        'metrics':aggregate_metrics([r['coordinates'] for r in rows],key),
        'student_step':12500,'teacher':'original Structure32 refine600',
        'training_manipulation':read_json(Path(__file__).parent/'manipulation.json'),
        'evaluation':'unmanipulated native student outputs; frozen original key/carrier'})

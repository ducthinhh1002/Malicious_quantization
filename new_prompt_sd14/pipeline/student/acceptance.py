import argparse
from pathlib import Path
import bootstrap
from common import read_json,freeze
from acceptance_metrics import aggregate_metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--branch',required=True)
    a=p.parse_args();key=read_json(a.run/'carrier/key.json')['signs'];assert len(key)==32
    result={'student_step':20000,'training':'new5000 prompt-source teacher images, no manipulation',
        'dataset_plan':read_json(Path(__file__).resolve().parent.parent/'plan.json'),'models':{}}
    for label,folder in [('student',a.run),('teacher',Path(__file__).resolve().parent.parent/'source')]:
        name=a.branch if label=='student' else 'teacher_wm_test'
        rows=read_json(folder/'scores'/f'{name}.json')
        result['models'][label]=aggregate_metrics([r['coordinates'] for r in rows],key)
    output=a.run/'students'/a.branch/'updates.jsonl'
    import json
    counts=json.loads(output.read_text().splitlines()[-1])
    result['optimizer_counts']={k:counts[k] for k in ['step','applied','skipped']}
    freeze(a.run/'result/student_acceptance32.json',result)

"""Keep original validation-MSE selection, report key recovery instead of TPR."""
import argparse
from pathlib import Path
import bootstrap
from common import read_json,freeze,sha256,write_json
from validation import validation_label,choose_checkpoint,SELECTION_RULE
from key_metrics import aggregate_metrics

def select(run,name):
    c=read_json(run/'config.json')
    rows=[read_json(run/'validation'/f'{validation_label(name,s)}.json') for s in c['checkpoint_steps']]
    winner=choose_checkpoint(rows)
    selection={'rule':SELECTION_RULE,'selected_step':winner['step'],
        'denoising_mse':winner['denoising_mse'],'checkpoint_hashes':winner['checkpoint_hashes'],
        'validation_cache_sha256':winner['validation_cache_sha256'],
        'candidate_loss_sha256':{str(r['step']):sha256(run/'validation'/f"{validation_label(name,r['step'])}.json") for r in rows},
        'test_used_for_selection':False,'detector_used_for_selection':False,'utility_used_for_selection':False}
    freeze(run/'result/selection'/f'{name}.json',selection)
    key=read_json(run/'carrier/key.json')['signs'];table=[]
    for r in rows:
        scores=read_json(run/'scores'/f"{validation_label(name,r['step'])}.json")
        table.append({'step':r['step'],'denoising_mse':r['denoising_mse'],
            'key_recovery':aggregate_metrics([x['coordinates'] for x in scores],key)})
    write_json(run/'result/validation'/f'{name}.json',{'selection':selection,'milestones':table})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage');p.add_argument('--branch',required=True)
    a=p.parse_args();select(a.run,a.branch)

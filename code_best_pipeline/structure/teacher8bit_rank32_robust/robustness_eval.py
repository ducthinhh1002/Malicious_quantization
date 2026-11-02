"""Held-out fixed manipulation panel; reports bit metrics, never selects a checkpoint."""
from pathlib import Path
import argparse,json
import bootstrap
from bootstrap import read_json,freeze
from attacks import ATTACKS,CONFIG,apply_attack
from key_metrics import aggregate_metrics

def main(run):
    import torch
    from semantic_teacher import extractor,pixels
    from semantic_carrier import SemanticCarrier
    from common import image_name
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    prompts=read_json(run/'prompts.json')['test'];n=len(prompts)
    labels=['teacher_wm_test','teacher_refine600']
    ext=extractor(run)
    result={'augmentation':CONFIG,'test_images':n,'groups':[1,8,32,64,256],
            'evaluation_seed':20261030,'test_used_for_selection':False,'labels':{}}
    for label in labels:
        result['labels'][label]={}
        for attack in ('clean',*ATTACKS):
            coords=[]
            for start in range(0,n,4):
                batch=[]
                for i in range(start,min(n,start+4)):
                    x=(pixels([run/'images'/label/image_name(i)],device='cpu')[0]+1)/2
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(20261030+i)
                        x=apply_attack(x,attack)
                    batch.append(x*2-1)
                with torch.inference_mode():
                    q=carrier.coordinates(ext.encode_image(torch.stack(batch).cuda(),
                         [p['text'] for p in prompts[start:start+len(batch)]]))
                coords.extend(q.cpu().tolist())
            record={'coordinates':coords,'metrics':aggregate_metrics(coords,carrier.signs.cpu().tolist())}
            freeze(run/'result/manipulations'/f'{label}_{attack}.json',record)
            result['labels'][label][attack]=record['metrics']
            print(label,attack,json.dumps(record['metrics']['rows']),flush=True)
    freeze(run/'result/manipulation_summary.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)

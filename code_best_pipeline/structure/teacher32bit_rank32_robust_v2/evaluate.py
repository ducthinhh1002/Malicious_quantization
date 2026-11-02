"""Paired evaluation, exact historical best settings; resume by completed attack."""
import bootstrap
import argparse
from pathlib import Path
import json
import torch
from common import read_json, freeze, image_name, sha256
from semantic_teacher import extractor, pixels
from semantic_carrier import SemanticCarrier
from attacks import ATTACKS, CONFIG, eval_attack
from key_metrics import aggregate_metrics

def evaluate(run, split='test'):
    rows=read_json(run/'prompts.json')[split]
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    ext=extractor(run)
    result={'split':split,'settings':CONFIG,'test_used_for_selection':False,'labels':{}}
    for label in [f'before_{split}',f'teacher_wm_{split}']:
        result['labels'][label]={}
        image_manifest=run/'images'/label/'image_hashes.json'
        hashes=read_json(image_manifest)
        if len(hashes)!=len(rows) or any(sha256(run/'images'/label/r['file_name'])!=r['sha256'] for r in hashes):
            raise RuntimeError('Evaluation image integrity failure: '+label)
        identity={'image_manifest_sha256':sha256(image_manifest),
                  'carrier_sha256':sha256(run/'carrier/complete.json'),
                  'prompts_sha256':sha256(run/'prompts.json'),
                  'evaluation_code':{n:sha256(Path(__file__).parent/n) for n in ['evaluate.py','attacks.py','key_metrics.py']}}
        for attack in ATTACKS:
            dest=run/'result'/split/f'{label}_{attack}.json'
            if dest.exists():
                record=read_json(dest)
                if record['identity']!=identity:raise RuntimeError('Evaluation identity changed')
            else:
                coordinates=[]
                for start in range(0,len(rows),8):
                    ids=list(range(start,min(start+8,len(rows))))
                    rgb=pixels([run/'images'/label/image_name(i) for i in ids],device='cpu')
                    batch=torch.stack([eval_attack((x+1)/2,attack,i)*2-1 for x,i in zip(rgb,ids)])
                    with torch.inference_mode():
                        q=carrier.coordinates(ext.encode_image(batch.cuda(),[rows[i]['text'] for i in ids]))
                    coordinates.extend(q.cpu().tolist())
                record={'identity':identity,'attack':attack,'coordinates':coordinates,
                        'metrics':aggregate_metrics(coordinates,carrier.signs.cpu().tolist())}
                freeze(dest,record)
            result['labels'][label][attack]=record['metrics']
            print(label,attack,json.dumps(record['metrics']['rows']),flush=True)
            # A progress summary can be read while a later attack is still running.
            progress=run/'result'/f'manipulation_{split}_progress.json'
            progress.write_text(json.dumps(result,indent=2))
    freeze(run/'result'/f'manipulation_{split}.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test')
    a=p.parse_args();evaluate(a.run,a.split)


"""Read-only training diagnostics; never changes thresholds or selects a checkpoint."""
import json
import math
from pathlib import Path
from statistics import mean


def audit_training(path, config, window_size=500):
    records=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if window_size<1:
        raise ValueError('Positive window size required')
    windows=[]
    for start in range(0,len(records),window_size):
        rows=records[start:start+window_size]
        names=sorted({key for r in rows for key in r.get('terms',{})})
        averages={}
        for name in names:
            values=[r['terms'][name] for r in rows if name in r.get('terms',{})
                    and math.isfinite(r['terms'][name])]
            if values:
                averages[name]={'mean':mean(values),'observations':len(values)}
        windows.append({'first_step':rows[0]['step'],'last_step':rows[-1]['step'],'terms':averages})
    norms=[r['gradient_norm'] for r in records
           if r.get('gradient_norm') is not None and math.isfinite(r['gradient_norm'])]
    gains=[r['meta']['proxy_gain'] for r in records if 'proxy_gain' in r.get('meta',{})
           and math.isfinite(r['meta']['proxy_gain'])]
    warnings=[]
    if norms and mean(n>1 for n in norms)>.9:
        warnings.append('Over 90% of finite updates were norm-clipped; inspect loss conflicts and drift. Clipping alone is not proof of a bug.')
    if windows:
        last=windows[-1]['terms']
        for metric,budget in [('mse','teacher_pixel_budget'),('lpips','teacher_lpips_budget')]:
            if metric in last and last[metric]['mean']>config[budget]:
                warnings.append(f'Last-window mean {metric} exceeds the configured training budget.')
    if gains and mean(gains)<=.01:
        warnings.append('Mean short-unroll proxy score gain is <=0.01; this is diagnostic, not proof of long-horizon transfer.')
    if config['teacher_tail_steps']:
        warnings.append('Sampling uses all timesteps but gradients are truncated. Compare a full-gradient ablation before attributing failure to LoRA capacity.')
    return {'schema_version':1,'quality_policy':'report and continue','test_used':False,
            'attempted':records[-1]['step'] if records else 0,
            'applied':records[-1]['applied'] if records else 0,
            'skipped':records[-1]['skipped'] if records else 0,
            'finite_gradient_observations':len(norms),
            'clipped_fraction':mean(n>1 for n in norms) if norms else None,
            'meta_proxy':{'observations':len(gains),'mean_gain':mean(gains) if gains else None,
                          'nonpositive_count':sum(g<=0 for g in gains)},
            'warnings':warnings,'windows':windows,
            'limits':'Loss means are not gradient magnitudes. Microbatch-averaged MSE does not recover per-image violation rates. Proxy gains are not TPR gains.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    a=p.parse_args()
    result=audit_training(a.run/'teacher/training.jsonl',json.loads((a.run/'config.json').read_text()))
    # Explicit CLI only prints; a completed run's frozen artifacts remain untouched.
    print(json.dumps(result,indent=2,allow_nan=False))

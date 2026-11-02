"""Conservative post-training candidates; original threshold objective is unchanged.

No test-guided selection. Five predeclared candidates, a fixed kappa=2 baseline,
and paired downstream students. Failed artifacts remain available for diagnosis.
"""
import math
from pathlib import Path
from common import freeze, read_json, sha256, image_name

SCALE = .85


def canonical_adapter_config(config):
    """PEFT serializes a set of target modules in arbitrary process order."""
    config = dict(config)
    if isinstance(config.get('target_modules'), list):
        config['target_modules'] = sorted(config['target_modules'])
    return config


def detection_eligible(row):
    checks=row.get('paired_gate',{}).get('checks',{})
    return checks.get('baseline_tpr_at_least_99pct') is True and checks.get('no_lost_positive_images') is True


def combine_states(states, weights):
    """Exact weighted LoRA deltas, including Conv2d; never average A and B."""
    import torch
    if not states or len(states)!=len(weights) or any(w<=0 or not math.isfinite(w) for w in weights):
        raise ValueError('Invalid positive mixture')
    if not math.isclose(sum(weights),1.):raise ValueError('Mixture weights must sum to one')
    keys=set(states[0])
    if any(set(s)!=keys for s in states):raise ValueError('Adapter target mismatch')
    out={}
    for key in sorted(keys):
        tensors=[s[key].float() for s in states]
        if not all(torch.isfinite(t).all() for t in tensors):raise ValueError('Nonfinite adapter')
        if '.lora_A.' in key:
            partner=key.replace('.lora_A.','.lora_B.')
            if partner not in keys:raise ValueError('Missing B factor')
            out[key]=torch.cat(tensors,dim=0).contiguous()
        elif '.lora_B.' in key:
            if key.replace('.lora_B.','.lora_A.') not in keys:raise ValueError('Missing A factor')
            out[key]=torch.cat([t*(SCALE*w) for t,w in zip(tensors,weights)],dim=1).contiguous()
        else:raise ValueError(f'Unexpected non-LoRA weight: {key}')
    if not out:raise ValueError('Empty adapter')
    return out


def paired_gate(baseline,candidate,base_scores,candidate_scores,threshold,comparisons=5):
    """Family-wise one-sided bootstrap bound; no regression on paired detections."""
    import numpy as np
    b=np.asarray(baseline['lpips_per_image'],dtype=np.float64)
    v=np.asarray(candidate['lpips_per_image'],dtype=np.float64)
    bs=np.asarray(base_scores,dtype=np.float64);cs=np.asarray(candidate_scores,dtype=np.float64)
    if b.shape!=v.shape or b.shape!=bs.shape or bs.shape!=cs.shape or len(b)<2:
        raise ValueError('Unpaired gate inputs')
    if not all(np.isfinite(x).all() for x in (b,v,bs,cs)) or not math.isfinite(threshold):
        raise ValueError('Nonfinite gate inputs')
    if not all(math.isfinite(q[k]) for q in (baseline,candidate) for k in ('clip_drop','paired_lpips')):
        raise ValueError('Nonfinite quality inputs')
    gain=b-v
    rng=np.random.default_rng(20261022)
    means=np.concatenate([gain[rng.integers(0,len(b),(500,len(b)))].mean(1) for _ in range(20)])
    lower=float(np.quantile(means,.05/comparisons))
    lost=int(((bs>=threshold)&(cs<threshold)).sum())
    checks={'baseline_tpr_at_least_99pct':float((bs>=threshold).mean())>=.99,
            'no_lost_positive_images':lost==0,
            'lpips_improves_at_least_2pct':float(v.mean())<=.98*float(b.mean()),
            'paired_gain_lower_bound_positive':lower>0,
            'clip_drop_at_most_001':candidate['clip_drop']<=.01,
            'clip_no_worse_than_baseline':candidate['clip_drop']<=baseline['clip_drop']}
    return {'passes':all(checks.values()),'checks':checks,'lost_positives':lost,
            'baseline_tpr':float((bs>=threshold).mean()),'candidate_tpr':float((cs>=threshold).mean()),
            'lpips_gain':float(gain.mean()),'gain_lower_bound':lower,
            'bootstrap_samples':10000,'one_sided_alpha':.05/comparisons,
            'note':'Development evidence, not a population or transfer guarantee'}


def build_adapter(run,folder,sources,weights):
    from safetensors.torch import load_file,save_file
    configs=[read_json(p/'adapter_config.json') for p in sources]
    for cfg in configs:
        if (cfg.get('use_dora') or cfg.get('use_rslora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern')
                or cfg['lora_alpha']!=cfg['r'] or cfg.get('bias','none')!='none'):
            raise ValueError('Only ordinary alpha=rank LoRA is supported')
    if any(canonical_adapter_config(cfg)!=canonical_adapter_config(configs[0]) for cfg in configs):
        raise ValueError('Mixture adapter configs differ')
    proof={'scale':SCALE,'weights':weights,'sources':[str(p.relative_to(run)) for p in sources],
           'source_sha256':[sha256(p/'adapter_model.safetensors') for p in sources],
           'source_config_sha256':[sha256(p/'adapter_config.json') for p in sources]}
    manifest=folder/'composition.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['inputs']!=proof or old['sha256']!=sha256(folder/'adapter_model.safetensors') or old['config_sha256']!=sha256(folder/'adapter_config.json'):
            raise RuntimeError('Composed adapter changed')
        return
    state=combine_states([load_file(str(p/'adapter_model.safetensors')) for p in sources],weights)
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'adapter_model.partial';save_file(state,str(tmp));tmp.replace(folder/'adapter_model.safetensors')
    cfg=dict(configs[0]);cfg['r']=sum(c['r'] for c in configs);cfg['lora_alpha']=cfg['r']
    freeze(folder/'adapter_config.json',cfg)
    freeze(manifest,{'inputs':proof,'sha256':sha256(folder/'adapter_model.safetensors'),
                     'config_sha256':sha256(folder/'adapter_config.json')})


def expand_and_select(run,original):
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set,score,numerical_policy
    from semantic_report import detection_metrics,teacher_quality,score_rows
    from semantic_assets import snapshot_identity
    numerical_policy()
    c=read_json(run/'config.json');n=c['validation_n']
    baseline=next(dict(r) for r in original if r['kappa']==2.)
    baseline.update(variant='baseline',adapter_scale=1.)
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    if baseline['metrics']['detection_rate']<max(.99,c['teacher_min_tpr']):
        freeze(run/'result/quality_gate_failed.json',{'reason':'Recreated baseline below 99% TPR','baseline':baseline})
        raise ValueError('Baseline detection not reproduced; refusing downstream training')
    base_label=baseline['metrics']['branch']
    bs=[r['score'] for r in score_rows(run,base_label,n)]
    by_k={r['kappa']:r for r in original}
    specs=[(f'scale085_k{k:g}',[k],[1.]) for k in (2.,1.5,1.)]
    specs += [('blend085_k2_k15',[2.,1.5],[.5,.5]),('blend085_k2_k1',[2.,1.],[.5,.5])]
    invariant_paths=[run/'carrier/complete.json',run/'result/threshold_teacher.json',run/'prompts.json']
    invariant_paths += [run/'carrier'/name for name in read_json(run/'carrier/complete.json')]
    invariants={str(p.relative_to(run)):sha256(p) for p in invariant_paths}
    rows=[{**r,'variant':'baseline' if r['kappa']==2. else 'unscaled_reference','adapter_scale':1.} for r in original]
    for tag,ks,weights in specs:
        folder=run/'teacher_fp/candidates'/tag
        sources=[(run/by_k[k]['model']).parent for k in ks]
        build_adapter(run,folder,sources,weights)
        model=merge_teacher(run,folder)
        label=f'candidate_{tag}_validation'
        generate_set(run,model,'validation',label);score(run,[label])
        quality=teacher_quality(run,label)
        scores=[r['score'] for r in score_rows(run,label,n)]
        gate=paired_gate(baseline['quality'],quality,bs,scores,threshold)
        row={'kappa':sum(k*w for k,w in zip(ks,weights)),'source_kappas':ks,'mixture_weights':weights,
             'variant':tag,'adapter_scale':SCALE,'model':str(model.relative_to(run)),
             'metrics':detection_metrics(run,label,threshold,n),
             'quality':{**quality,'passes':gate['passes'],'lpips_is_hard_gate':False,'quality_policy':'report_only_user_20261023'},
             'quality_gates_enforced':False,'paired_gate':gate,
             'model_hashes':snapshot_identity(model),'config_sha256':sha256(run/'config.json')}
        freeze(folder/'validation.json',row);rows.append(row)
        print(f"QUALITY {tag}: LPIPS={quality['paired_lpips']:.6f}, TPR={row['metrics']['detection_rate']:.4f}, accepted={gate['passes']}",flush=True)
    if any(sha256(run/p)!=digest for p,digest in invariants.items()):raise RuntimeError('Frozen protocol changed')
    freeze(run/'result/quality_comparison.json',{'baseline':baseline,'candidates':rows,'invariants':invariants,'test_used':False})
    eligible=[r for r in rows if detection_eligible(r)]
    if not eligible:
        freeze(run/'result/detection_gate_failed.json',{'reason':'No scale-0.85 candidate retains baseline detections','candidates':rows})
        raise ValueError('No candidate retains detection; student training stopped')
    return rows,min(eligible,key=lambda r:(r['quality']['paired_lpips'],-r['metrics']['mean_score'],r['variant']))


def final_acceptance(run):
    """One fixed test audit, after selection. Never select another teacher on test."""
    import torch
    import lpips
    from semantic_teacher import pixels
    from semantic_report import score_rows
    from student import branches
    c=read_json(run/'config.json');n=c['test_n']
    utility=read_json(run/'result/utility.json')['branches']
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    results=[]
    pairs=[('teacher_baseline_test','teacher_wm_test','teacher_clean_test','teacher')]
    pairs += [(b['name'][:-2]+'baseline',b['name'],'unadapted','student') for b in branches(c) if b['condition']=='wm']
    for base,cand,reference,family in pairs:
        threshold=read_json(run/'result'/f'threshold_{family}.json')['threshold']
        quality=[]
        for label in (base,cand):
            reference_generation=read_json(run/'images'/reference/'generation.json')
            generation=read_json(run/'images'/label/'generation.json')
            ignored={'checkpoint','model_hashes'}
            if {k:v for k,v in generation.items() if k not in ignored}!={k:v for k,v in reference_generation.items() if k not in ignored}:
                raise RuntimeError('Paired LPIPS requires identical prompt/seed/sampler/settings')
            values=[]
            for start in range(0,n,c['score_batch']):
                files=[image_name(i) for i in range(start,min(start+c['score_batch'],n))]
                with torch.inference_mode():
                    values+=perceptual(pixels([run/'images'/reference/f for f in files]),
                                      pixels([run/'images'/label/f for f in files])).flatten().cpu().tolist()
            quality.append({'lpips_per_image':values,'paired_lpips':sum(values)/n,
                            'clip_drop':utility[reference]['clip_score']-utility[label]['clip_score']})
        bs=[r['score'] for r in score_rows(run,base,n)]
        cs=[r['score'] for r in score_rows(run,cand,n)]
        gate=paired_gate(quality[0],quality[1],bs,cs,threshold,comparisons=len(pairs))
        # A teacher-only result is insufficient: keep the same conservative student gate.
        results.append({'candidate':cand,'baseline':base,'family':family,'reference':reference,'baseline_quality':quality[0],
                        'candidate_quality':quality[1],'gate':gate,
                        'baseline_utility':utility[base],'candidate_utility':utility[cand]})
    accepted=bool(results) and all(r['gate']['passes'] for r in results)
    freeze(run/'result/acceptance.json',{'meets_original_joint_targets':accepted,'status':'evaluated',
           'quality_gates_enforced':False,
           'results':results,'teacher_selection_sha256':sha256(run/'teacher_fp/selected.json'),
           'test_used_for_selection':False,'note':'No automatic retry or reselection from final test; failed artifacts kept for audit.'})
    print(f'FINAL EVALUATION COMPLETE; quality is report-only; meets original targets: {accepted}',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    final_acceptance(p.parse_args().run.resolve())

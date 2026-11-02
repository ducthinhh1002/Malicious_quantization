"""Frozen per-image verification; scientific failures are reported, not gated."""
import bootstrap
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from bootstrap import read_json, write_json, sha256

BRANCH='lora_r32_conv_wm'


def wilson(k,n):
    if n<=0 or not 0<=k<=n:raise ValueError('Invalid binomial counts')
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-radius),min(1.,center+radius)


def validated_rows(run,label,n,key):
    rows=read_json(run/'scores'/f'{label}.json')
    if len(rows)!=n or [r['file'] for r in rows]!=[f'sample_{i:06d}.png' for i in range(n)]:
        raise ValueError('Missing/unaligned score rows: '+label)
    signs=np.asarray(key['signs'],dtype=np.float64)
    coords=np.asarray([r['coordinates'] for r in rows],dtype=np.float64)
    scores=np.asarray([r['score'] for r in rows],dtype=np.float64)
    if coords.shape!=(n,len(signs)) or not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid/nonfinite semantic scores')
    if not np.allclose((coords*signs).mean(1),scores,atol=1e-6,rtol=1e-5):
        raise ValueError('Coordinate/score/key inconsistency')
    for row in rows:
        if sha256(run/'images'/label/row['file'])!=row['sha256']:
            raise ValueError('Image hash mismatch: '+row['file'])
    return rows,coords,scores


def summarize(coords,scores,signs,tau):
    detected=scores>=tau;n=len(scores);k=int(detected.sum());low,high=wilson(k,n)
    bits=(np.sign(coords)==signs).sum(1);aggregate=(np.sign(coords.mean(0))==signs)
    return dict(n=n,detected=k,tpr=k/n,tpr_ci95=[low,high],threshold=float(tau),
        mean_score=float(scores.mean()),score_quantiles={str(q):float(np.quantile(scores,q)) for q in [.01,.05,.1,.5,.9]},
        per_image_bit_accuracy=float(bits.mean()/len(signs)),exact_message_fraction=float(np.mean(bits==len(signs))),
        matching_bits_histogram={str(j):int(np.sum(bits==j)) for j in range(len(signs)+1)},
        aggregate_matched_bits=int(aggregate.sum()),aggregate_bit_accuracy=float(aggregate.mean()),
        target90_point_estimate_met=bool(k/n>=.9),target90_ci95_lower_met=bool(low>=.9))


def pair_identity(run,label,n):
    c=read_json(run/'config.json');generation=read_json(run/'images'/label/'generation.json')
    prompts=read_json(run/'prompts.json')['test']
    # Image filenames alone cannot establish pairing: check every relevant setting.
    return dict(prompts=prompts[:n],seed=c['seed'],generation={k:v for k,v in generation.items() if k!='checkpoint'})


def paired_difference(new,old,seed=20261026,repeats=4000):
    if len(new)!=len(old) or not len(new):raise ValueError('Unpaired counts')
    delta=np.asarray(new,dtype=float)-np.asarray(old,dtype=float)
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):boot.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
    return dict(tpr_difference_pp=float(delta.mean()*100),
        paired_bootstrap_ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
        resamples=repeats,seed=seed,limitation='Image resampling within one model seed; not training-seed uncertainty')


def verify_report(run):
    from campaign import verify, SOURCE, PROXY, ROOT
    verify(run.parent)
    c=read_json(run/'config.json');key=read_json(run/'carrier/key.json')
    signs=np.asarray(key['signs'],dtype=float)
    if len(signs)!=8 or not np.isin(signs,[-1,1]).all():raise ValueError('Expected frozen historical 8-bit key')
    for name,digest in read_json(run/'carrier/complete.json').items():
        if sha256(run/'carrier'/name)!=digest:raise ValueError('Carrier payload changed')
    threshold=read_json(run/'result/threshold_student.json');tau=threshold['threshold']
    if threshold!=read_json(SOURCE/'result/threshold_student.json') or key!=read_json(SOURCE/'carrier/key.json'):
        raise ValueError('Primary key/threshold changed')
    n=c['test_n'];rows,coords,scores=validated_rows(run,BRANCH,n,key)
    result=summarize(coords,scores,signs,tau);warnings=[]
    if not result['target90_point_estimate_met']:warnings.append('Student per-image TPR is below90%; target not achieved')
    if not result['target90_ci95_lower_met']:warnings.append('95% lower confidence bound does not establish TPR>=90%')
    per_image=[]
    for row,vec in zip(rows,coords):
        per_image.append(dict(file=row['file'],sha256=row['sha256'],score=row['score'],
            detected=bool(row['score']>=tau),matching_bits=int((np.sign(vec)==signs).sum()),
            bit_accuracy=float((np.sign(vec)==signs).mean())))
    controls={};comparisons={}
    clean=ROOT/'native_quality_20261025/clean_control'
    for name,path,label in [('unadapted',run,'unadapted'),('historical_clean_trained_7500',clean,BRANCH)]:
        if read_json(path/'carrier/key.json')!=key:raise ValueError('Control key mismatch')
        _,x,s=validated_rows(path,label,n,key)
        summary=summarize(x,s,signs,tau)
        summary['empirical_false_positive_rate']=summary.pop('tpr')
        summary['false_positive_ci95']=summary.pop('tpr_ci95')
        summary.pop('target90_point_estimate_met');summary.pop('target90_ci95_lower_met')
        controls[name]=summary
        if summary['empirical_false_positive_rate']>.01:
            warnings.append(name+': empirical false positives exceed calibration target1%; inspect CI')
    for name,path in [('projected7500',SOURCE),('projected10000',PROXY)]:
        if read_json(path/'carrier/key.json')!=key:raise ValueError('Comparison key mismatch')
        _,x,s=validated_rows(path,BRANCH,n,key)
        comparison=dict(metrics=summarize(x,s,signs,tau))
        if pair_identity(run,BRANCH,n)==pair_identity(path,BRANCH,n):
            comparison['paired']=paired_difference(scores>=tau,s>=tau)
        else:comparison['pairing_warning']='Generation protocol/prompt identity differs; unpaired descriptive comparison only'
        prior=read_json(path/'result/student_summary.json');new=read_json(run/'result/student_summary.json')
        quality=dict(student_lpips=new['lpips'],reference_lpips=prior['lpips'],
            reference='paired unadapted SD1.4 for LPIPS; clean teacher for FID/KID')
        for metric in ['frechet_inception_distance','kernel_inception_distance_mean','clip_score']:
            quality[metric]=new['utility'].get(metric);quality['reference_'+metric]=prior['utility'].get(metric)
        comparison['quality']=quality;comparisons[name]=comparison
        if new['lpips']>prior['lpips']:warnings.append('Student paired LPIPS worse than '+name)
        if new['utility']['frechet_inception_distance']>prior['utility']['frechet_inception_distance']:
            warnings.append('Student FID worse than '+name)
    teacher_tau=read_json(run/'result/threshold_teacher.json')['threshold']
    _,tc,ts=validated_rows(run,'teacher_wm_test',n,key)
    selection=read_json(run/'result/selection'/f'{BRANCH}.json')
    if selection['test_used_for_selection'] or selection['detector_used_for_selection']:
        raise ValueError('Checkpoint selection used forbidden test/detector evidence')
    milestones=[]
    for step in c['checkpoint_steps']:
        label=f'validation_{BRANCH}_step{step}'
        _,x,s=validated_rows(run,label,c['validation_generation_n'],key)
        milestones.append(dict(step=step,metrics=summarize(x,s,signs,tau),
            denoising_mse=read_json(run/'validation'/f'{label}.json')['denoising_mse']))
    report=dict(primary=result,teacher_test=summarize(tc,ts,signs,teacher_tau),
        clean_controls=controls,comparisons=comparisons,validation_milestones=milestones,
        selected_checkpoint=selection,student_quality=read_json(run/'result/student_summary.json')['utility'],
        teacher_validation=read_json(run/'result/refined_teacher.json'),warnings=warnings,
        primary_definition='single-image frozen semantic score >= frozen clean-student threshold',
        calibrated_target_fpr=.01,calibration_n=threshold['n'],key_bits=8,
        verifier_sha256=sha256(run/'carrier/complete.json'),threshold_sha256=sha256(run/'result/threshold_student.json'),
        limitations=read_json(run.parent/'plan.json')['limitations']+[
            'Historical clean-trained control has7500steps, not a matched15000step control.',
            'Aggregate key recovery is descriptive; it is not single-image TPR.',
            'FID/KID compare to clean teacher generations, not a natural-image population.'])
    out=run/'result';write_json(out/'verification.json',report)
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
    with (out/'validation_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','denoising_mse','tpr','bit_accuracy','score_p10'])
        writer.writeheader()
        for r in milestones:writer.writerow(dict(step=r['step'],denoising_mse=r['denoising_mse'],
            tpr=r['metrics']['tpr'],bit_accuracy=r['metrics']['per_image_bit_accuracy'],score_p10=r['metrics']['score_quantiles']['0.1']))
    text=(f'# Transfer90 result\n\nStudent detected: {result["detected"]}/{n} '
        f'({result["tpr"]:.1%}); Wilson95%CI {result["tpr_ci95"]}.\n\n'
        f'90% point target met: {result["target90_point_estimate_met"]}. '
        f'95% lower-bound target met: {result["target90_ci95_lower_met"]}.\n\n'
        f'Per-image bit accuracy: {result["per_image_bit_accuracy"]:.2%}. '
        f'Aggregate key matches: {result["aggregate_matched_bits"]}/8.\n\n'
        'Checkpoint selected by validation denoising MSE; fixed verifier/threshold.\n\n'
        'Warnings (report-only):\n\n'+''.join('- '+w+'\n' for w in warnings)+
        '\nLimitations:\n\n'+''.join('- '+w+'\n' for w in report['limitations']))
    (out/'SUMMARY.md').write_text(text)
    print(text)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    verify_report(p.parse_args().run.resolve())

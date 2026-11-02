"""Fixed-scale student comparisons, group attribution, then Pareto-grid search.

Isolated extension: never edits the active campaign or selects on test results.
All GPU stages are subprocesses; a detached locked supervisor persists the queue.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT/'src'
sys.path.insert(0, str(HARNESS))
from common import freeze, read_json, sha256, write_json, image_name
from group_search import (group_for_key, scale_state, coalitions, kernel_shap,
                          attribution_vector, proposal, shortlist, pareto_front)

QUEUE = ROOT/'queued_research_20261023'
BRANCH = 'lora_r32_conv_wm'
BASE_BRANCH = 'lora_r32_conv_baseline'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tag_for(vector):
    return 'g_' + hashlib.sha256(json.dumps(list(vector)).encode()).hexdigest()[:16]


def link_file(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not os.path.samefile(src, dst) and sha256(src) != sha256(dst):
            raise RuntimeError(f'Changed shared input: {dst}')
    else:
        os.link(src, dst)
    return str(dst)


def share(src, dst):
    src, dst = Path(src), Path(dst)
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=link_file, dirs_exist_ok=True)
    else:
        link_file(src, dst)


def safe_remove_derived(folder):
    """Only a reconstructible export beneath this extension may be removed."""
    folder = folder.resolve()
    if QUEUE.resolve() not in folder.parents or folder.name != 'merged_pipeline':
        raise ValueError('Refusing removal outside queue derived exports')
    if folder.is_symlink():
        raise ValueError('Refusing symlink export removal')
    shutil.rmtree(folder)


def release_export(folder, keep_adapter=True):
    proof = folder/'merge_test.json'
    if proof.exists():
        freeze(folder/'merge_audit.json', read_json(proof))
        if (folder/'merged_pipeline').exists():
            safe_remove_derived(folder/'merged_pipeline')
        proof.unlink()
    if not keep_adapter:
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            (folder/name).unlink(missing_ok=True)


def deduplicate_exports():
    """Replace byte-identical completed exports by hardlinks; retain all paths."""
    canonical = {}; records = []
    for base in (ROOT/'run/teacher_fp', ROOT/'parallel_scale085'):
        for proof in sorted(base.rglob('merge_test.json')):
            export = proof.parent/'merged_pipeline'
            for name, digest in read_json(proof).get('model_hashes', {}).items():
                file = export/name
                if export.resolve() not in file.resolve().parents or not file.is_file():
                    continue
                if file.stat().st_size < 1024**2:
                    continue
                previous = canonical.setdefault(digest, file)
                if previous == file or os.path.samefile(previous, file):
                    continue
                if sha256(previous) != digest or sha256(file) != digest:
                    raise RuntimeError('Completed export changed before deduplication')
                temporary = file.with_name(file.name+'.queue-link')
                temporary.unlink(missing_ok=True)
                os.link(previous, temporary); os.replace(temporary, file)
                records.append({'path':str(file), 'canonical':str(previous), 'sha256':digest})
    if records:
        with (QUEUE/'dedup.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':now(), 'files':records})+'\n')


def checkpoint(source):
    folder = source/'teacher_fp/candidates/kappa_2.0/checkpoint-3000'
    proof = read_json(folder/'checkpoint.json')
    if proof['step'] != 3000:
        raise RuntimeError('Unexpected source step')
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        if sha256(folder/name) != proof['files'][name]:
            raise RuntimeError('Source checkpoint changed')
    return folder


def initialize(source):
    from safetensors import safe_open
    src = checkpoint(source)
    with safe_open(str(src/'adapter_model.safetensors'), framework='pt') as f:
        groups = sorted({group_for_key(k) for k in f.keys()})
    if not 1 <= len(groups) <= 6:
        raise RuntimeError(f'Unsupported groups: {groups}')
    c = read_json(source/'config.json')
    expected = {'ranks':[32], 'adapter_scopes':['attention_conv'], 'train_n':1000,
                'max_train_steps':7500, 'checkpoint_steps':[2500,5000,7500],
                'teacher_objective':'threshold_cfg_v1', 'validation_n':256}
    if any(c[k] != v for k, v in expected.items()):
        raise RuntimeError('Source no longer matches reviewed student experiment')
    QUEUE.mkdir(parents=True, exist_ok=True)
    plan = {'source_run':str(source.resolve()), 'source_checkpoint':str(src),
            'source_config_sha256':sha256(source/'config.json'),
            'source_adapter_sha256':sha256(src/'adapter_model.safetensors'),
            'carrier_sha256':sha256(source/'carrier/complete.json'),
            'threshold_teacher_sha256':sha256(source/'result/threshold_teacher.json'),
            'prompts_sha256':sha256(source/'prompts.json'), 'groups':groups,
            'fixed_scales':[.65,.85], 'pilot_n':32, 'search_seed':20261023,
            'coalitions':2**len(groups), 'pareto_new_evaluations':16,
            'random_control_new_evaluations':16, 'full_validation_shortlist':3,
            'student':{k:c[k] for k in ('train_n','test_n','max_train_steps','checkpoint_steps',
                                       'ranks','adapter_scopes','batch_size','learning_rate','precision')},
            'max_parallel_gpu_workers':2, 'quality_gates_enforced':False,
            'selection_uses_test':False,
            'jobs':['fixed_scale_students_065_085','kernelshap_groups_and_student','pareto_grid_and_student'],
            'methods':'Adapted group attribution and Pareto-grid evolutionary search; not full NAS/LLM reproduction'}
    freeze(QUEUE/'plan.json', plan)
    files = [Path(__file__), ROOT/'group_search.py'] + list(HARNESS.glob('*.py'))
    files += [HARNESS/'vendor/provenance.json'] + list((HARNESS/'vendor/diffusers').glob('*.py'))
    freeze(QUEUE/'code_manifest.json', {str(p.relative_to(ROOT)):sha256(p) for p in files})
    return plan


def verify_plan():
    plan = read_json(QUEUE/'plan.json'); source = Path(plan['source_run'])
    for relative, digest in read_json(QUEUE/'code_manifest.json').items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError(f'Queue code changed: {relative}')
    for rel, key in [('config.json','source_config_sha256'), ('prompts.json','prompts_sha256'),
                     ('carrier/complete.json','carrier_sha256'),
                     ('result/threshold_teacher.json','threshold_teacher_sha256')]:
        if sha256(source/rel) != plan[key]:
            raise RuntimeError(f'Frozen source changed: {rel}')
    if sha256(checkpoint(source)/'adapter_model.safetensors') != plan['source_adapter_sha256']:
        raise RuntimeError('Source adapter changed')
    return plan


def prepare_run(run, pilot_n=None, shared_student=False):
    plan = read_json(QUEUE/'plan.json'); source = Path(plan['source_run'])
    run.mkdir(parents=True, exist_ok=True)
    c = read_json(source/'config.json')
    c.update(threshold_quality=False, train_clean_controls=False)
    prompts = read_json(source/'prompts.json')
    if pilot_n is not None:
        c['validation_n'] = pilot_n
        prompts['validation'] = prompts['validation'][:pilot_n]
    freeze(run/'config.json', c); freeze(run/'prompts.json', prompts)
    assets = read_json(source/'assets.json')
    assets['prompt_sha256'] = sha256(run/'prompts.json')
    freeze(run/'assets.json', assets)
    share(source/'carrier', run/'carrier')
    share(source/'result/threshold_teacher.json', run/'result/threshold_teacher.json')
    freeze(run/'lineage.json', {'source_run':str(source), 'source_config_sha256':plan['source_config_sha256'],
           'source_adapter_sha256':plan['source_adapter_sha256'], 'pilot_n':pilot_n,
           'changes':'external fixed adapter; wm-only branch; validation prefix only for pilot',
           'quality_gates_enforced':False, 'test_used_for_selection':False})
    # Exact prefix reuse; metadata reflects the new prompt manifest and full source lineage.
    clean = source/'images/teacher_clean_validation'; dest = run/'images/teacher_clean_validation'
    n = c['validation_n']
    entries = read_json(clean/'image_hashes.json')[:n]
    for entry in entries:
        link_file(clean/entry['file_name'], dest/entry['file_name'])
    generation = read_json(clean/'generation.json')
    generation.update(prompts_sha256=sha256(run/'prompts.json'), seeds=generation['seeds'][:n])
    freeze(dest/'generation.json', generation); freeze(dest/'image_hashes.json', entries)
    if shared_student:
        if not (source/'complete.json').exists():
            raise RuntimeError('Main run must complete before shared student controls are reused')
        selection=read_json(source/'teacher_fp/selected.json')
        baseline=next(r for r in selection['candidates'] if r.get('variant')=='baseline')
        baseline_adapter=(source/baseline['model']).parent/'adapter_model.safetensors'
        if sha256(baseline_adapter)!=plan['source_adapter_sha256']:
            raise RuntimeError('Main baseline student was not trained from the same kappa=2 teacher')
        for rel in ('result/threshold_student.json','result/threshold.json',
                    'images/teacher_clean_test','images/unadapted','scores/unadapted.json'):
            share(source/rel, run/rel)
        share(source/'images'/BASE_BRANCH, run/'images/baseline_student')
        share(source/'scores'/f'{BASE_BRANCH}.json', run/'scores/baseline_student.json')
        freeze(run/'baseline_lineage.json', {
            'selection':read_json(source/'result/selection'/f'{BASE_BRANCH}.json'),
            'student_complete_sha256':sha256(source/'students'/BASE_BRANCH/'complete.json'),
            'training_protocol_sha256':sha256(source/'students'/BASE_BRANCH/'protocol.json'),
            'teacher_scale':1., 'source_checkpoint':plan['source_checkpoint']})
    return run


def build_group_adapter(run, vector):
    from safetensors.torch import load_file, save_file
    plan = read_json(QUEUE/'plan.json'); source = Path(plan['source_checkpoint'])
    cfg = read_json(source/'adapter_config.json')
    if (cfg.get('use_dora') or cfg.get('use_rslora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern')
            or cfg['r'] != cfg['lora_alpha'] or cfg.get('bias','none') != 'none'):
        raise ValueError('Only ordinary alpha=rank adapters are supported')
    folder = run/'derived'/tag_for(vector); folder.mkdir(parents=True, exist_ok=True)
    recipe = {'groups':plan['groups'], 'vector':list(vector), 'source':str(source),
              'source_sha256':sha256(source/'adapter_model.safetensors'),
              'operation':'A unchanged; B multiplied per group in FP32'}
    freeze(folder/'recipe.json', recipe)
    result = scale_state(load_file(str(source/'adapter_model.safetensors')), plan['groups'], vector)
    target = folder/'adapter_model.safetensors'
    if not target.exists():
        temp = folder/'adapter.partial'; save_file(result,str(temp)); temp.replace(target)
    else:
        import torch
        old = load_file(str(target))
        if set(old) != set(result) or any(not torch.equal(old[k],v) for k,v in result.items()):
            raise RuntimeError('Derived adapter changed')
    freeze(folder/'adapter_config.json', cfg)
    return folder


def evaluate_vector(run, vector):
    import numpy as np
    from semantic_teacher import numerical_policy, generate_set, score
    from semantic_teacher_train import merge_teacher
    from semantic_report import teacher_quality, detection_metrics, score_rows
    numerical_policy()
    tag = tag_for(vector); label = f'candidate_{tag}_validation'
    target = run/'measurements'/f'{tag}.json'
    c = read_json(run/'config.json'); n = c['validation_n']
    if target.exists():
        row = read_json(target)
        score_rows(run,label,n)
        if row['vector'] != list(vector):
            raise RuntimeError('Vector cache mismatch')
        return row
    folder = build_group_adapter(run, vector); model = merge_teacher(run,folder)
    generate_set(run, model,'validation',label)
    score(run,[label]); q = teacher_quality(run,label)
    threshold = read_json(run/'result/threshold_teacher.json')['threshold']
    scores = [r['score'] for r in score_rows(run,label,n)]
    row = {'tag':tag,'vector':list(vector),'label':label,'lpips':q['paired_lpips'],
           'clip_drop':q['clip_drop'],'margin_p10':float(np.quantile(scores,.1)-threshold),
           'margin_min':min(scores)-threshold,'metrics':detection_metrics(run,label,threshold,n),
           'quality':q,'scores':scores,'n':n,'quality_gates_enforced':False,
           'test_used':False,'merge_proof':read_json(folder/'merge_test.json')}
    freeze(target,row)
    release_export(folder, keep_adapter=False)
    return row


def verify_measurement(run,row):
    """Check cached pixels and score records without allocating a GPU model."""
    label=row['label']
    scores=read_json(run/'scores'/f'{label}.json')
    if len(scores)!=row['n'] or [r['score'] for r in scores]!=row['scores']:
        raise RuntimeError('Cached measurement scores changed')
    for i,r in enumerate(scores):
        if r['file']!=image_name(i) or sha256(run/'images'/label/r['file'])!=r['sha256']:
            raise RuntimeError('Cached candidate image changed')
        if sha256(run/'images/teacher_clean_validation'/image_name(i))!=row['quality']['identity']['clean'][i]:
            raise RuntimeError('Cached clean reference changed')


def disk_wait():
    while shutil.disk_usage(QUEUE).free < 12*1024**3:
        write_json(QUEUE/'resource_wait.json', {'status':'waiting_disk','time':now(),'minimum_free_gib':12})
        print('Waiting for 12 GiB free disk; artifacts retained',flush=True)
        time.sleep(30)


def call(run, label, args):
    disk_wait()
    log = run/'queue_logs'/f'{label}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    print('STAGE',str(run),label,flush=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,*map(str,args)],check=True,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)


def own(run,label,operation,vector=None):
    args = [Path(__file__),'--operation',operation,'--run',run]
    if vector is not None:
        args += ['--vector',','.join(map(str,vector))]
    call(run,label,args)


def harness(run,label,file,*args):
    call(run,label,[HARNESS/file,'--run',run,*args])


def eval_many(run,vectors):
    def one(vector):
        tag = tag_for(vector)
        if not (run/'measurements'/f'{tag}.json').exists():
            own(run,tag,'evaluate',vector)
        row=read_json(run/'measurements'/f'{tag}.json')
        if row['vector']!=list(vector): raise RuntimeError('Cached vector mismatch')
        verify_measurement(run,row)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(one,vectors))


def generate_student_data(run,vector):
    from semantic_teacher import numerical_policy,generate_set,load_native
    from semantic_teacher_train import merge_teacher
    numerical_policy()
    folder = build_group_adapter(run,vector); model = merge_teacher(run,folder)
    pipe = load_native(model)
    for split in ('train','validation','test'):
        generate_set(run,model,split,f'teacher_wm_{split}',pipe)
    del pipe
    release_export(folder)


def student_report(run):
    import numpy as np
    import torch
    import lpips
    from semantic_report import detection_metrics,score_rows
    from semantic_teacher import numerical_policy,pixels
    numerical_policy()
    c = read_json(run/'config.json'); n = c['test_n']
    threshold = read_json(run/'result/threshold_student.json')['threshold']
    utility = read_json(run/'result/utility.json')['branches']
    perceptual = lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    rows = []
    for label in ('baseline_student',BRANCH):
        left = read_json(run/'images/unadapted/generation.json')
        right = read_json(run/'images'/label/'generation.json')
        if {k:v for k,v in left.items() if k!='checkpoint'} != {k:v for k,v in right.items() if k!='checkpoint'}:
            raise RuntimeError('Student LPIPS generation pairing changed')
        distances = []
        for start in range(0,n,c['score_batch']):
            files = [image_name(i) for i in range(start,min(n,start+c['score_batch']))]
            with torch.inference_mode():
                distances += perceptual(pixels([run/'images/unadapted'/f for f in files]),
                                        pixels([run/'images'/label/f for f in files])).flatten().cpu().tolist()
        scores = [r['score'] for r in score_rows(run,label,n)]
        rows.append({'label':label,'metrics':detection_metrics(run,label,threshold,n),
                     'lpips':float(np.mean(distances)),'lpips_per_image':distances,
                     'clip_drop':utility['unadapted']['clip_score']-utility[label]['clip_score'],
                     'utility':utility[label], 'scores':scores})
    base,candidate = rows
    paired = np.array(base['lpips_per_image'])-np.array(candidate['lpips_per_image'])
    rng=np.random.default_rng(20261023)
    means=np.concatenate([paired[rng.integers(0,n,(500,n))].mean(1) for _ in range(4)])
    bs=np.array(base['scores']); cs=np.array(candidate['scores'])
    result={'status':'evaluated','quality_gates_enforced':False,'results':rows,
            'lpips_gain':float(paired.mean()),'lpips_gain_ci95':np.quantile(means,[.025,.975]).tolist(),
            'lost_positive_images':int(((bs>=threshold)&(cs<threshold)).sum()),
            'tpr_difference':candidate['metrics']['detection_rate']-base['metrics']['detection_rate'],
            'selected_checkpoint':read_json(run/'result/selection'/f'{BRANCH}.json'),
            'teacher':read_json(run/'teacher_choice.json'),
            'threshold_student_sha256':sha256(run/'result/threshold_student.json'),
            'test_used_for_selection':False,'note':'One training seed; image bootstrap is not training-seed uncertainty.'}
    freeze(run/'result/student_comparison.json',result)
    freeze(run/'complete.json',{'status':'completed','report':'result/student_comparison.json'})


def train_student(name, vector, choice):
    run = prepare_run(QUEUE/'students'/name,shared_student=True)
    freeze(run/'teacher_choice.json',{'vector':list(vector),**choice,'test_used_for_selection':False})
    if (run/'complete.json').exists():
        return run
    own(run,'01_teacher_data','student_data',vector)
    harness(run,'02_validation_cache','validation.py','--stage','cache')
    harness(run,'03_student_train','student.py','--stage','train','--branch',BRANCH)
    harness(run,'04_validate','validation.py','--stage','evaluate','--branch',BRANCH)
    labels=[f'validation_{BRANCH}_step{s}' for s in read_json(run/'config.json')['checkpoint_steps']]
    harness(run,'05_validation_scores','semantic_teacher.py','--stage','score','--directories',*labels)
    harness(run,'06_validation_quality','utility.py','--split','validation','--labels',*labels)
    harness(run,'07_select_mse','validation.py','--stage','select','--branch',BRANCH)
    harness(run,'08_student_test','student.py','--stage','generate','--branch',BRANCH)
    harness(run,'09_test_scores','semantic_teacher.py','--stage','score','--directories',BRANCH,'teacher_wm_test')
    harness(run,'10_test_quality','utility.py','--labels','teacher_clean_test','teacher_wm_test','unadapted','baseline_student',BRANCH)
    own(run,'11_paired_report','student_report')
    return run


def validate_and_train(method, candidates):
    """Freeze full-validation choice before generating any method student/test."""
    run=prepare_run(QUEUE/'full_validation')
    n=len(read_json(QUEUE/'plan.json')['groups'])
    baseline=eval_many(run,[[1.]*n])[0]
    rows=eval_many(run,[r['vector'] for r in candidates])
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    eligible=[]
    for row in rows:
        row=dict(row)
        row['lost_positive_images']=sum(b>=threshold and v<threshold for b,v in zip(baseline['scores'],row['scores']))
        if baseline['metrics']['detection_rate']>=.99 and row['lost_positive_images']==0:
            eligible.append(row)
    selected=min(eligible,key=lambda r:(r['lpips'],-r['margin_p10'],r['tag'])) if eligible else None
    decision={'method':method,'candidates':rows,'selected':selected,'quality_gates_enforced':False,
              'selection':'no lost baseline positives, then minimum LPIPS; no quality stopping cutoff',
              'test_used':False}
    freeze(QUEUE/f'{method}_selection.json',decision)
    if selected is None:
        # Failed detection remains a published outcome; other queue jobs continue.
        return {'status':'no_detection_preserving_candidate','selection':str(QUEUE/f'{method}_selection.json')}
    vector=selected['vector']
    if all(v==1. for v in vector):
        return {'status':'completed','student_reused':str(Path(read_json(QUEUE/'plan.json')['source_run'])/'students'/BASE_BRANCH),
                'vector':vector,'note':'Search returned the existing scale-1 baseline; no duplicate training'}
    # Reuse an exact already-trained vector rather than spend another identical training seed.
    for existing in sorted((QUEUE/'students').glob('*')):
        if (existing/'complete.json').exists() and read_json(existing/'teacher_choice.json')['vector']==vector:
            return {'status':'completed','student_reused':str(existing),'vector':vector}
    student=train_student(method,vector,{'method':method,'selection_sha256':sha256(QUEUE/f'{method}_selection.json')})
    return {'status':'completed','student_run':str(student),'vector':vector}


def attribution_job():
    import numpy as np
    plan=read_json(QUEUE/'plan.json'); n=len(plan['groups'])
    pilot=prepare_run(QUEUE/'pilot',pilot_n=plan['pilot_n'])
    masks=coalitions(n)
    rows=eval_many(pilot,[[.85 if b else .65 for b in mask] for mask in masks])
    fits={key:kernel_shap(masks,[r[key] for r in rows]) for key in ('lpips','clip_drop','margin_p10')}
    # Resample paired image indices across every coalition, rather than treating
    # its 64 correlated image panels as independent observations.
    lp=np.array([r['quality']['lpips_per_image'] for r in rows])
    clip=np.array([np.array(r['quality']['clip_per_image_clean'])-np.array(r['quality']['clip_per_image_fp']) for r in rows])
    scores=np.array([r['scores'] for r in rows])
    threshold=read_json(pilot/'result/threshold_teacher.json')['threshold']
    rng=np.random.default_rng(plan['search_seed']); samples={k:[] for k in fits}
    for _ in range(200):
        ids=rng.integers(0,plan['pilot_n'],plan['pilot_n'])
        values={'lpips':lp[:,ids].mean(1),'clip_drop':clip[:,ids].mean(1),
                'margin_p10':np.quantile(scores[:,ids],.1,axis=1)-threshold}
        for k in fits: samples[k].append(kernel_shap(masks,values[k])['phi'])
    for k in fits: fits[k]['paired_image_bootstrap_ci95']=np.quantile(samples[k],[.025,.975],axis=0).tolist()
    vector,merit=attribution_vector(fits,n)
    report={'groups':plan['groups'],'fits':fits,'merit':merit,'proposed_vector':vector,
            'attribution_domain':[.65,.85],'pilot_n':plan['pilot_n'],
            'method':'KernelSHAP-inspired adapter group attribution; not a NAS reproduction',
            'note':'All coalitions measured. Image sampling uncertainty and cross-scale extrapolation remain.',
            'test_used':False}
    freeze(QUEUE/'attribution.json',report)
    candidate=next(r for r in rows if r['vector']==vector)
    return validate_and_train('kernelshap',[candidate])


def pareto_job():
    plan=read_json(QUEUE/'plan.json'); n=len(plan['groups'])
    pilot=prepare_run(QUEUE/'pilot',pilot_n=plan['pilot_n'])
    attribution=read_json(QUEUE/'attribution.json')
    initial=eval_many(pilot,[[1.]*n,[.85]*n,[.65]*n,attribution['proposed_vector']])
    initial=list({r['tag']:r for r in initial}.values())
    archives={mode:list(initial) for mode in ('pareto','random')}
    # Each policy starts with identical seeds and receives exactly 16 new oracle evaluations.
    # Cache reuse is reported but still counted against the policy's evaluation budget.
    for i in range(plan['pareto_new_evaluations']):
        vectors=[]
        for mode in ('pareto','random'):
            rows=archives[mode]; seen={tuple(r['vector']) for r in rows}
            seed=plan['search_seed']+i+(10000 if mode=='random' else 0)
            vector=proposal(rows,seen,n,seed,mode)
            freeze(QUEUE/'proposals'/f'{mode}_{i:02d}.json',{'mode':mode,'iteration':i,'vector':vector,'seed':seed})
            vectors.append(vector)
        measured=eval_many(pilot,vectors)
        for mode,row in zip(('pareto','random'),measured):
            archives[mode].append(row)
        write_json(QUEUE/'search_progress.json',{'round':i+1,'budget_per_policy':plan['pareto_new_evaluations']})
    freeze(QUEUE/'pareto_archive.json',{'archives':archives,
           'fronts':{m:pareto_front(v) for m,v in archives.items()},
           'new_evaluations_per_policy':plan['pareto_new_evaluations'],
           'method':'Pareto-grid guided crossover/mutation, inspired by MPaGE; no LLM synthesis',
           'test_used':False,'quality_gates_enforced':False})
    candidates=shortlist(archives['pareto'],plan['full_validation_shortlist'])
    freeze(QUEUE/'pareto_shortlist.json',{'vectors':[r['vector'] for r in candidates],'test_used':False})
    return validate_and_train('pareto',candidates)


def worker(operation,run,vector):
    verify_plan()
    if operation=='evaluate': evaluate_vector(run,vector)
    elif operation=='student_data': generate_student_data(run,vector)
    elif operation=='student_report': student_report(run)
    elif operation=='scale065': train_student('scale065',vector,{'method':'predeclared_fixed_scale','scale':.65})
    elif operation=='scale085': train_student('scale085',vector,{'method':'predeclared_fixed_scale','scale':.85})
    elif operation=='attribution': freeze(QUEUE/'kernelshap_complete.json',attribution_job())
    elif operation=='pareto': freeze(QUEUE/'pareto_complete.json',pareto_job())
    else: raise ValueError(operation)


def supervisor():
    import fcntl
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=verify_plan(); source=Path(plan['source_run'])
        status={'pid':os.getpid(),'state':'waiting_main','jobs':plan['jobs'],'started':now()}
        write_json(QUEUE/'status.json',status)
        while not (source/'complete.json').exists():
            deduplicate_exports()
            status.update(updated=now(),state='waiting_main',dependency=str(source/'complete.json'))
            write_json(QUEUE/'status.json',status)
            time.sleep(30)
        deduplicate_exports(); verify_plan()
        n=len(plan['groups'])
        try:
            status.update(state='running',job='fixed_scale_students_065_085',updated=now())
            write_json(QUEUE/'status.json',status)
            def fixed(scale):
                operation='scale065' if scale==.65 else 'scale085'
                own(QUEUE,operation,operation,[scale]*n)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(fixed,[.65,.85]))
            for operation in ('attribution','pareto'):
                status.update(job=operation,updated=now()); write_json(QUEUE/'status.json',status)
                own(QUEUE,operation,operation)
            status.update(state='completed',updated=now())
            write_json(QUEUE/'status.json',status)
        except Exception as exc:
            status.update(state='failed',error=str(exc),updated=now())
            write_json(QUEUE/'status.json',status)
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--initialize',type=Path);p.add_argument('--supervise',action='store_true')
    p.add_argument('--operation');p.add_argument('--run',type=Path);p.add_argument('--vector')
    a=p.parse_args()
    if a.initialize:
        print(json.dumps(initialize(a.initialize.resolve()),indent=2));return
    if a.supervise: supervisor();return
    vector=[float(v) for v in a.vector.split(',')] if a.vector else None
    worker(a.operation,a.run,vector)


if __name__=='__main__': main()

"""Fixed12500 policy: no validation selection or duplicate endpoint generation."""
from pathlib import Path
import shutil

def patch_campaign(text):
    if "stage('05_fixed12500'" in text:return text
    text=text.replace('checkpoint_steps=[7500,10000,11250,12500]','checkpoint_steps=[12500]')
    text=text.replace(",'validation_cache']", "]")
    text=text.replace("'key_select.py']","'key_select.py','fixed_endpoint.py']")
    start=text.index("            stage('02_validation'")
    end=text.index("            stage('06_test'",start)
    text=text[:start]+"            stage('05_fixed12500',ROOT/'fixed_endpoint.py','--branch',BRANCH)\n"+text[end:]
    start=text.index("            stage('07_endpoint'")
    end=text.index("            stage('10_summary'",start)
    text=text[:start]+"            stage('08_score',HARNESS/'semantic_teacher.py','--stage','score','--directories',BRANCH)\n            stage('09_quality',HARNESS/'utility.py','--labels',BRANCH)\n"+text[end:]
    text=text.replace('minimum validation denoising MSE at7500/10000/11250/12500; report selected and fixed12500','fixed12500 by user; no validation checkpoint evaluation')
    text=text.replace('validation denoising MSE only; also report fixed12500 endpoint','fixed12500 by user; no validation checkpoint evaluation')
    text=text.replace("'selection':'validation MSE only; fixed12500 also reported'","'selection':'fixed12500 by user'")
    return text

def apply_policy(root):
    root=Path(root);student=root/'student12500';p=student/'campaign.py'
    p.write_text(patch_campaign(p.read_text()))
    shutil.copy2(Path(__file__),student/'fixed_endpoint.py')
    p=student/'key_summary.py';text=p.read_text()
    text=text.replace("[(branch,'student_summary.json'),('endpoint12500','endpoint12500_summary.json')]","[(branch,'student_summary.json')]")
    text=text.replace("'selection':'validation denoising MSE only'","'selection':'fixed12500 by user; validation selection skipped'")
    p.write_text(text)
    if root.name=='structure32_student12500':
        import json,time
        marker=root/'deployment/stages/11_validation_cache.json'
        if not marker.exists():
            marker.parent.mkdir(parents=True,exist_ok=True)
            marker.write_text(json.dumps({'skipped':True,'updated':time.time(),
                'reason':'user requested fixed12500; validation checkpoint evaluation/selection disabled'},indent=2)+'\n')

def select(run,branch):
    from common import freeze,read_json
    from student import checkpoint_hashes
    folder=run/'students'/branch;cp=folder/'snapshot-12500'
    hashes=checkpoint_hashes(cp);record=read_json(folder/'complete.json')['checkpoints']
    if not hashes:raise RuntimeError('Missing fixed12500 snapshot')
    for name,digest in hashes.items():
        if record.get(str((cp/name).relative_to(folder)))!=digest:raise RuntimeError('Fixed checkpoint hash mismatch')
    freeze(run/'result/selection'/f'{branch}.json',{'rule':'fixed12500 explicitly requested by user; no validation comparison',
        'selected_step':12500,'checkpoint_hashes':hashes,'test_used_for_selection':False,
        'detector_used_for_selection':False,'utility_used_for_selection':False,'validation_evaluation_skipped':True})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--branch');p.add_argument('--apply',type=Path)
    a=p.parse_args()
    if a.apply:apply_policy(a.apply)
    else:
        import bootstrap
        select(a.run,a.branch)

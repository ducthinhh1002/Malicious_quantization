"""Clone only source wrappers and immutable data references; rank4, fresh20000."""
from pathlib import Path
import json,shutil,subprocess,sys,os

ROOT=Path(__file__).resolve().parent
OLD=Path('/var/lib/structure32_student_manipulations')
SOURCE=Path('/var/lib/structure32_student12500/runs/structure32')
sys.path.insert(0,'/var/lib/structure32_student12500/harness')
from common import read_json,freeze,sha256

def main():
    assert not (ROOT/'prepared.json').exists(),'Already prepared; do not overwrite a live experiment'
    expected='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'
    assert sha256(SOURCE/'teacher_fp/refined/adapter_model.safetensors')==expected
    assert len(read_json(SOURCE/'carrier/key.json')['signs'])==32
    assert read_json(SOURCE/'teacher_fp/refined/adapter_config.json')['r']==32
    plan=read_json(OLD/'study_plan.json')
    plan.update(student_rank=4,steps=20000,new_initialization='clean SD1.4; fresh rank4 LoRA; no teacher or student LoRA used for initialization',
        rank32_reference=str(OLD/'continuation20k'),student_source='same teacher32bit/rank32 original refine600 images',
        source_control_student='rank32 seven manipulated branches at20000',
        full_state_every=1250,amp_skip_policy='report attempted/applied/skipped separately; fail if more than1 skipped update',
        rank32_plan_sha256=sha256(OLD/'study_plan.json'))
    freeze(ROOT/'study_plan.json',plan)
    names=['rotation15','crop75','drop96','gaussian005','brightness07','blur15','flip']
    source_files=read_json(OLD/'branches/rotation15/runs/structure32_student12500_seed0/code_manifest.json')
    for name in names:
        src=OLD/'branches'/name;dest=ROOT/'branches'/name;dest.mkdir(parents=True,exist_ok=True)
        for filename in source_files:
            shutil.copy2(src/filename,dest/filename)
        shutil.copy2(src/'sitecustomize.py',dest/'sitecustomize.py')
        # The manipulation settings and pixel mapping must be bit-identical.
        for filename in ['channels.py','manipulation.json','pixel_index.json','schedule.py','resume_state.py']:
            assert sha256(src/filename)==sha256(dest/filename)
        f=dest/'campaign.py';s=f.read_text()
        s=s.replace('12501','20001').replace('12500','20000')
        s=s.replace('/var/lib/structure32_student20000/runs/structure32',str(SOURCE))
        s=s.replace("BRANCH='lora_r32_conv_wm'","BRANCH='lora_r4_conv_wm'")
        s=s.replace('c.update(max_train_steps=20000,checkpoint_steps=[20000])',
            'c.update(ranks=[4],max_train_steps=20000,checkpoint_steps=[20000])')
        s=s.replace("'rank':32,'scope':'attention_conv'","'rank':4,'scope':'attention_conv'")
        s=s.replace('rank32 attention_conv','rank4 attention_conv')
        # The schedule itself is unchanged; correct prose changed by step rename.
        s=s.replace('cosine to2e-5 at20000','cosine to2e-5 at12500')
        s=s.replace('cosine to2e-5 by20000','cosine to2e-5 by12500')
        s=s.replace('absolute7500to20000cosine_floor0.2','absolute7500to12500cosine_floor0.2')
        s=s.replace('all10 complete state checkpoints','all16 complete state checkpoints')
        s=s.replace("'training_manipulation':read_json(ROOT/'manipulation.json'),\n        'schedule':",
            "'training_manipulation':read_json(ROOT/'manipulation.json'),\n        'augmentation_code_sha256':sha256(ROOT/'channels.py'),\n        'schedule':")
        s=s.replace("rows[-1]['applied']!=expected or rows[-1]['skipped']!=0",
            "rows[-1]['applied']+rows[-1]['skipped']!=expected or rows[-1]['skipped']>(0 if smoke else 1)")
        f.write_text(s)
        for filename in ['fixed_endpoint.py','key_summary.py','acceptance.py']:
            f=dest/filename;f.write_text(f.read_text().replace('12500','20000'))
        subprocess.run([sys.executable,'-m','compileall','-q',str(dest)],check=True)
        subprocess.run([sys.executable,str(dest/'campaign.py'),'--initialize'],cwd=dest,check=True)
        q=dest/'runs/structure32_student20000_seed0';r=q/'student20000'
        c=read_json(r/'config.json');identity=read_json(q/'resume_identity.json')
        assert c['ranks']==[4] and c['max_train_steps']==20000 and c['num_bits']==32
        assert identity['rank']==4 and identity['scope']=='attention_conv'
        assert identity['training_manipulation']==read_json(src/'manipulation.json')
        assert identity['training_images_manifest_sha256']==sha256(SOURCE/'images/teacher_wm_train/image_hashes.json')
        print('PREPARED',name,'bits32 rank4 steps20000',flush=True)
    # Use the same four-GPU-slot scheduler, updated only for the new queue name.
    text=(OLD/'launch_study.py').read_text().replace('12500','20000')
    (ROOT/'launch_study.py').write_text(text)
    freeze(ROOT/'prepared.json',{'branches':names,'teacher_sha256':expected,'student_rank':4,'steps':20000})

if __name__=='__main__':main()

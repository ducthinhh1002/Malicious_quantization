"""Run on .21: isolated wrappers, original assets shared read-only, no retraining teacher."""
from pathlib import Path
import os,sys,json,shutil,subprocess,hashlib
ROOT=Path(__file__).resolve().parent
ORIGINAL=Path('/var/lib/structure32_student12500')
SOURCE=ORIGINAL/'runs/structure32'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ORIGINAL/'harness'))
from common import sha256,read_json,freeze
from channels import SETTINGS,pixel_digest

def prepare():
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    expected='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'
    assert sha256(SOURCE/'teacher_fp/refined/adapter_model.safetensors')==expected
    assert len(read_json(SOURCE/'carrier/key.json')['signs'])==32
    assert read_json(SOURCE/'teacher_fp/refined/adapter_config.json')['r']==32
    assert read_json(SOURCE/'data_complete.json')['adapter_sha256']==expected
    torch_threads=__import__('torch');torch_threads.set_num_threads(4)
    entries=read_json(SOURCE/'images/teacher_wm_train/image_hashes.json');assert len(entries)==5000
    lookup={}
    for i,row in enumerate(entries):
        f=SOURCE/'images/teacher_wm_train'/row['file_name'];assert sha256(f)==row['sha256']
        with Image.open(f) as im:
            assert im.size==(512,512);digest=pixel_digest(to_tensor(im.convert('RGB')))
        if digest in lookup:raise RuntimeError('Duplicate image pixels need explicit per-row lookup')
        lookup[digest]=i
    freeze(ROOT/'pixel_index.json',lookup)
    freeze(ROOT/'study_plan.json',{'source':str(SOURCE),'teacher_adapter_sha256':expected,
        'teacher_refine_steps':600,'teacher_robust1200_used':False,'bits':32,'student_rank':32,
        'student_base':'clean pinned SD1.4','steps':12500,'train_n':5000,'test_n':1000,
        'source_images_manifest_sha256':sha256(SOURCE/'images/teacher_wm_train/image_hashes.json'),
        'settings':SETTINGS,'deterministic_per_source_image':True,
        'training':'ordinary denoising MSE on manipulated image targets; no owner/detector loss',
        'pixel_channels':'float tensor preserved except declared RGB8 rotation/brightness/blur',
        'validation_selection':False,'loss_log':'each updates.jsonl','full_state_every':1250,
        'parallel_slots':{'gpu0':2,'gpu1':2},'per_process_cap_gib':40,
        'source_control_student':str(ORIGINAL/'student12500/runs/structure32_student12500_seed0/student12500'),
        'key_metrics':'BA, exact32/32, acceptance>=24/32 for1,8,32,64,256; theoretical FPR conditional0.350018%'})
    files=['schedule.py','resume_state.py','driver.py','campaign.py','tests.py','key_metrics.py','key_summary.py','key_select.py','fixed_endpoint.py']
    for name,setting in SETTINGS.items():
        branch=ROOT/'branches'/name;branch.mkdir(parents=True,exist_ok=True)
        for n in files:shutil.copy2(ORIGINAL/'student12500'/n,branch/n)
        for n in ['bootstrap.py','sitecustomize.py','channels.py','acceptance.py']:
            shutil.copy2(ROOT/n,branch/n)
        shutil.copy2(ROOT/'acceptance_metrics.py',branch/'acceptance_metrics.py')
        if not (branch/'pixel_index.json').exists():os.link(ROOT/'pixel_index.json',branch/'pixel_index.json')
        freeze(branch/'manipulation.json',{'name':name,'setting':setting,
            'source_teacher_sha256':expected,'seed_rule':'20261029+i rotation/blur;20261030+i others',
            'pixel_lookup_sha256':sha256(ROOT/'pixel_index.json'),'fixed_across_epochs':True})
        p=branch/'driver.py';s=p.read_text()
        s=s.replace('44*2**30','40*2**30')
        s=s.replace("    return source.replace(marker,marker+'        train_dataloader.set_epoch(epoch)\\n')", """    source=source.replace(marker,marker+'        train_dataloader.set_epoch(epoch)\\n')
    marker2='            transforms.ToTensor(),\\n            transforms.Normalize([0.5], [0.5]),'
    if source.count(marker2)!=1:raise RuntimeError('Pinned image preprocessing changed')
    return source.replace(marker2,'            transforms.ToTensor(),\\n            transforms.Lambda(WM_CHANNEL),\\n            transforms.Normalize([0.5], [0.5]),')""")
        s=s.replace("ns={'__name__':'structure12500_pinned_trainer','__file__':str(a.trainer)}",
            "from channels import ManipulatedTrainingImage\n    channel=ManipulatedTrainingImage(Path(__file__).resolve().parent)\n    ns={'__name__':'structure12500_pinned_trainer','__file__':str(a.trainer),'WM_CHANNEL':channel}")
        if 'transforms.Lambda(WM_CHANNEL)' not in s:raise RuntimeError('Missing channel insertion')
        p.write_text(s)
        p=branch/'campaign.py';s=p.read_text()
        s=s.replace("SOURCE=BASE/'runs/structure32'",'SOURCE=Path('+repr(str(SOURCE))+')')
        s=s.replace("'fixed_endpoint.py']","'fixed_endpoint.py','channels.py','manipulation.json','pixel_index.json','acceptance.py','acceptance_metrics.py']",1)
        a=s.index('def share(');b=s.index('\ndef initialize',a)
        s=s[:a]+'''def share(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve(),target_is_directory=source.is_dir())
    elif target.resolve()!=source.resolve():raise RuntimeError('Unexpected shared artifact')

'''+s[b:]
        s=s.replace("'precision':'fp16','rank':32,'scope':'attention_conv',", "'precision':'fp16','rank':32,'scope':'attention_conv',\n        'training_manipulation':read_json(ROOT/'manipulation.json'),")
        s=s.replace("'training_gpu_cap_gib':44","'training_gpu_cap_gib':40")
        s=s.replace('train from scratch with unchanged Structure teacher images','train from clean SD1.4 with fixed manipulated Structure refine600 image targets')
        s=s.replace("'student_steps':12500,","'student_steps':12500,'training_manipulation':read_json(ROOT/'manipulation.json'),")
        s=s.replace("required=50000 if stage in ['00_smoke','01_train'] else 55000","required=24000")
        s=s.replace("            stage('10_summary',Path(__file__),'--operation','summary')", "            stage('10_summary',Path(__file__),'--operation','summary')\n            stage('11_acceptance32',ROOT/'acceptance.py','--branch',BRANCH)")
        p.write_text(s)
        subprocess.run([sys.executable,'-m','compileall','-q',str(branch)],check=True)
        subprocess.run([sys.executable,str(p),'--initialize'],cwd=branch,check=True)
        print('PREPARED',name,flush=True)
    freeze(ROOT/'prepared.json',{'branches':list(SETTINGS),'teacher_sha256':expected})

if __name__=='__main__':prepare()

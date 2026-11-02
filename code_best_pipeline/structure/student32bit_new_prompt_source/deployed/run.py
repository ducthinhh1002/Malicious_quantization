"""Frozen Structure32 refine600 -> new prompt-source images -> fresh rank32 SD1.4."""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,random,re,unicodedata,fcntl
ROOT=Path(__file__).resolve().parent
OLD=Path('/var/lib/structure32_student12500/runs/structure32')
HARNESS=Path('/var/lib/structure32_student12500/harness')
SOURCE=ROOT/'source'
sys.path.insert(0,str(HARNESS))
from common import read_json,write_json,freeze,sha256
EXPECTED='2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8'

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())

def prepare():
    import pyarrow.parquet as pq
    from transformers import CLIPTokenizer
    assert sha256(OLD/'teacher_fp/refined/adapter_model.safetensors')==EXPECTED
    assert len(read_json(OLD/'carrier/key.json')['signs'])==32
    assert sha256(ROOT/'inputs/train.parquet')=='0bf624e8a81f85a97af6924e406b7e504daf9e3cbf5d893dadd347e2e0759d31'
    assert sha256(ROOT/'inputs/eval.parquet')=='e2df7c8d0b1b7bff9bd0afcf61b841181876920ab55c23d1791c6be99b5e9e25'
    SOURCE.mkdir(exist_ok=True)
    assets=read_json(OLD/'assets.json')
    tokens=[CLIPTokenizer.from_pretrained(assets[key],subfolder='tokenizer',local_files_only=True) for key in ['teacher_clean','student']]
    normalize=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
    old=[r['text'] for rows in read_json(OLD/'prompts.json').values() for r in rows]
    seen={normalize(s) for s in old};prefix=[set(),set()]
    for t,store in zip(tokens,prefix):
        store.update(tuple(x) for x in t(old,truncation=True,max_length=77)['input_ids'])
    blocked=re.compile(r'\b(nude|naked|nsfw|porn\w*|erotic\w*|sex\w*|nipples?|penis|vagina|loli|topless|fetish\w*)\b',re.I)
    rejected={'duplicate_text':0,'duplicate_effective_tokens':0,'short_or_filtered':0}
    def pick(file,n,offset):
        rows=pq.read_table(ROOT/'inputs'/file).column('Prompt').to_pylist()
        order=list(range(len(rows)));random.Random(20261101+offset).shuffle(order);result=[]
        for index in order:
            text=rows[index].strip()
            if len(text.split())<5 or blocked.search(text):rejected['short_or_filtered']+=1;continue
            norm=normalize(text)
            if norm in seen:rejected['duplicate_text']+=1;continue
            ids=[tuple(t(text,truncation=True,max_length=77)['input_ids']) for t in tokens]
            if any(k in store for k,store in zip(ids,prefix)):
                rejected['duplicate_effective_tokens']+=1;continue
            seen.add(norm)
            for k,store in zip(ids,prefix):store.add(k)
            result.append({'text':text,'image_id':offset+len(result),'caption_id':offset+len(result),
                           'source_row':index,'source_file':file})
            if len(result)==n:return result
        raise RuntimeError('Insufficient distinct prompts')
    new=pick('train.parquet',6000,2000000)
    prompts={'train':new[:5000],'validation':new[5000:],'test':pick('eval.parquet',1000,4000000),'calibration':[]}
    freeze(SOURCE/'prompts.json',prompts)
    assets['prompt_sha256']=sha256(SOURCE/'prompts.json')
    assets['new_prompt_dataset']={'repo':'Gustavosta/Stable-Diffusion-Prompts','revision':'f40484704ca5bdeb389483123ba017897f1bfcdc'}
    freeze(SOURCE/'assets.json',assets)
    c=read_json(OLD/'config.json');c.update(ranks=[32],max_train_steps=20000,checkpoint_steps=[20000])
    freeze(SOURCE/'config.json',c)
    for name in ['carrier','teacher_fp/refined','teacher_fp/calibration_target.json','result/threshold.json','result/threshold_teacher.json','result/threshold_student.json']:
        link(OLD/name,SOURCE/name)
    freeze(SOURCE/'teacher_choice.json',{'method':'fixed_original_refine600_new_prompt_source',
        'source_checkpoint':str(OLD/'teacher_fp/refined'),'source_adapter_sha256':EXPECTED,
        'num_bits':32,'refinement_steps':600,'teacher_is_unchanged':True,'teacher_retrained':False})
    freeze(ROOT/'plan.json',{'teacher_sha256':EXPECTED,'teacher_training_prompt_source':'original COCO',
        'new_prompt_dataset':assets['new_prompt_dataset'],'new_counts':{k:len(v) for k,v in prompts.items()},
        'old_coco_prompts_excluded':len(old),'deduplication':'NFKC/casefold/whitespace text; effective77-token IDs of BOTH teacher and student tokenizers; across all new splits and old COCO splits',
        'rejections':rejected,'selection_seed':20261101,'student':'fresh clean SD1.4,rank32 attention_conv,batch4,seed0,20000steps',
        'train_images':'unmanipulated output of unchanged teacher refine600',
        'evaluation':'new heldout official eval-source prompts; no COCO test reused',
        'carrier_calibration':'original COCO carrier/key and thresholds frozen; not fitted to new prompts',
        'interpretation':'prompt-source distribution shift; no claim all prompts are outside SD pretraining distribution'})
    freeze(ROOT/'prepared.json',{'completed':time.time()})

def generate_worker(kind):
    from semantic_teacher import load_native,generate_set
    if kind=='train':
        model=OLD/'teacher_fp/refined/merged_pipeline';pipe=load_native(model)
        generate_set(SOURCE,model,'train','teacher_wm_train',pipe)
    else:
        model=OLD/'teacher_fp/refined/merged_pipeline';pipe=load_native(model)
        for split in ['validation','test']:generate_set(SOURCE,model,split,'teacher_wm_'+split,pipe)
        del pipe
        import gc,torch;gc.collect();torch.cuda.empty_cache()
        model=read_json(SOURCE/'assets.json')['teacher_clean'];pipe=load_native(model)
        for split in ['validation','test']:generate_set(SOURCE,model,split,'teacher_clean_'+split,pipe)
        del pipe;gc.collect();torch.cuda.empty_cache()
        from student import load_pipeline,generate_images
        pipe,cp=load_pipeline(SOURCE,'unadapted');generate_images(SOURCE,'unadapted',pipe,cp)

def student_prepare():
    ref=Path('/var/lib/structure32_student12500/student12500');dest=ROOT/'student';dest.mkdir(exist_ok=True)
    if (dest/'runs/structure32_student20000_seed0/code_manifest.json').exists():
        return dest
    files=['bootstrap.py','schedule.py','resume_state.py','driver.py','campaign.py','tests.py','key_metrics.py','key_summary.py','key_select.py','fixed_endpoint.py']
    for name in files:shutil.copy2(ref/name,dest/name)
    link(HARNESS,ROOT/'harness')
    shutil.copy2(ROOT/'acceptance.py',dest/'acceptance.py');shutil.copy2(ROOT/'acceptance_metrics.py',dest/'acceptance_metrics.py')
    f=dest/'campaign.py';s=f.read_text().replace('12501','20001').replace('12500','20000')
    s=s.replace("SOURCE=BASE/'runs/structure32'","SOURCE=BASE/'source'")
    s=s.replace("'fixed_endpoint.py']","'fixed_endpoint.py','acceptance.py','acceptance_metrics.py']",1)
    # Share source data read-only rather than making additional copies.
    a=s.index('def share(');b=s.index('\ndef initialize',a)
    s=s[:a]+'''def share(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve(),target_is_directory=source.is_dir())

'''+s[b:]
    s=s.replace('cosine to2e-5 at20000','cosine to2e-5 at12500').replace('cosine to2e-5 by20000','cosine to2e-5 by12500').replace('absolute7500to20000cosine_floor0.2','absolute7500to12500cosine_floor0.2')
    s=s.replace('all10 complete state checkpoints','all16 complete state checkpoints')
    s=s.replace("rows[-1]['applied']!=expected or rows[-1]['skipped']!=0","rows[-1]['applied']+rows[-1]['skipped']!=expected")
    s=s.replace("            stage('10_summary',Path(__file__),'--operation','summary')","            stage('10_summary',Path(__file__),'--operation','summary')\n            stage('11_acceptance32',ROOT/'acceptance.py','--branch',BRANCH)")
    f.write_text(s)
    for name in ['fixed_endpoint.py','key_summary.py']:
        f=dest/name;f.write_text(f.read_text().replace('12500','20000'))
    subprocess.run([sys.executable,str(dest/'campaign.py'),'--initialize'],cwd=dest,check=True)
    queue=dest/'runs/structure32_student20000_seed0'
    assert read_json(queue/'resume_identity.json')['rank']==32
    assert read_json(queue/'student20000/config.json')['max_train_steps']==20000
    return dest

def supervise():
    with (ROOT/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**kw):write_json(ROOT/'status.json',{'stage':stage,'updated':time.time(),'pid':os.getpid(),**kw})
        if not (ROOT/'prepared.json').exists():status('prepare_prompts');prepare()
        children=[]
        for gpu,kind in [(1,'train'),(0,'controls')]:
            e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
            log=(ROOT/(kind+'_generation.log')).open('ab')
            c=subprocess.Popen([sys.executable,str(Path(__file__)),'generate',kind],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append((kind,c,log))
        status('teacher_generation',workers=[{'kind':n,'pid':c.pid} for n,c,l in children])
        for name,c,log in children:
            code=c.wait();log.close()
            if code:raise RuntimeError(name+' generation failed')
        e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']='1'
        status('score_frozen_teacher_and_control')
        with (ROOT/'source_scores.log').open('a') as log:
            subprocess.run([sys.executable,str(HARNESS/'semantic_teacher.py'),'--run',str(SOURCE),'--stage','score','--directories','unadapted','teacher_wm_test'],env=e,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        freeze(SOURCE/'data_complete.json',{'adapter_sha256':EXPECTED})
        freeze(SOURCE/'complete.json',{'role':'new prompt images ready; unchanged teacher','completed':time.time()})
        dest=student_prepare();status('student_smoke_then_train20000')
        e.update(PYTHONPATH=str(dest),BITMARK_RUNTIME_ROOT='/var/lib/vllm/runtime',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (ROOT/'student_supervisor.log').open('a') as log:
            subprocess.run([sys.executable,str(dest/'campaign.py'),'--supervise'],env=e,cwd=dest,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('completed')

if __name__=='__main__':
    try:
        if len(sys.argv)>1:generate_worker(sys.argv[2])
        else:supervise()
    except Exception as exc:
        write_json(ROOT/'status.json',{'stage':'failed','error':repr(exc),'updated':time.time()});raise

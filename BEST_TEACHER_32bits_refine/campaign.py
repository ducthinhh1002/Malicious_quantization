"""Original Full Structure, 32-bit payload -> independent 12500-step SD1.4 student."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
import runtime
from runtime import ROOT,BASE,HARNESS,SOURCE,RUN,STATE,read_json,write_json,freeze,sha256,share


def initialize():
    import torch
    from semantic_carrier import choose_key,SemanticCarrier
    old=BASE/'runs/k1_from_clean'
    c=read_json(old/'config.json')
    c.update(num_bits=32,stable_candidate_count=32,teacher_train_steps=3000,
        teacher_lr=1e-4,teacher_checkpoint_every=500,teacher_micro_batch=4,teacher_batch=4,
        teacher_precision='bf16',teacher_kappas=[1.],cache=str(BASE/'runtime/cache'))
    freeze(SOURCE/'config.json',c);share(old/'assets.json',SOURCE/'assets.json')
    share(old/'prompts.json',SOURCE/'prompts.json')
    for label in ['teacher_clean_train','teacher_clean_validation','teacher_clean_test',
        'teacher_clean_calibration','unadapted','unadapted_calibration']:
        share(old/'images'/label,SOURCE/'images'/label)
    carrier=SOURCE/'carrier';carrier.mkdir(parents=True,exist_ok=True)
    for name in ['pca.pt','stability.json']:share(old/'carrier'/name,carrier/name)
    stable=read_json(carrier/'stability.json')
    pcs,signs=choose_key([r['pc'] for r in stable[:32]],32,c['key_seed'])
    pca=torch.load(carrier/'pca.pt',map_location='cpu',weights_only=True)
    prov=read_json(old/'carrier/provenance.json')
    prov.update(config_sha256=sha256(SOURCE/'config.json'),
        reuse='Identical clean-only PCA/stability; select32 bits from top32 stable PCs',
        source_provenance_sha256=sha256(old/'carrier/provenance.json'))
    key={'key_seed':c['key_seed'],'num_bits':32,'selected_pc_indices':pcs,'signs':signs,
        'sigmas':pca['variance'][pcs].sqrt().tolist(),'basis_sha256':sha256(carrier/'pca.pt'),
        'carrier_fit_prompt_sha256':prov['train_prompt_sha256'],
        'index_convention':'zero-based inclusive PC range',
        'selection_rule':'correlation descending, sign retention descending, PC ascending; RNG sample without replacement'}
    freeze(carrier/'key.json',key);freeze(carrier/'provenance.json',prov)
    freeze(carrier/'complete.json',{n:sha256(carrier/n) for n in ['pca.pt','key.json','stability.json','provenance.json']})
    SemanticCarrier.load(carrier)
    freeze(ROOT/'plan.json',{'method':'original Full Structure with32-bit payload',
        'reference_payload_bits':8,'num_bits':32,'stable_candidate_count':[16,32],
        'reuse':'clean-only PCA, all clean RGB/captions, pinned SD2.1/SD1.4 models; hardlinks',
        'teacher':'fresh clean initialization k1 3000steps lr1e-4 then original Structure600steps lr1e-5',
        'student':'fresh clean SD1.4 rank32 attention_conv batch4;12500steps; identical cooldown schedule',
        'primary_metrics':'group bit accuracy and exact full-key recovery for1,8,32,64,256 images',
        'calibration':'recompute new32bit clean thresholds solely for original teacher loss; not reported as TPR',
        'scientific_stopping_gates':False,'single_seed':0})


def calibration():
    from semantic_report import freeze_thresholds
    from semantic_threshold import frozen_calibration
    freeze_thresholds(SOURCE);frozen_calibration(SOURCE)


def k1():
    from k1_original import train_one
    train_one(SOURCE,1.,SOURCE/'teacher_fp/candidates/kappa_1.0')


def baseline():
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import load_native,generate_set,numerical_policy
    numerical_policy()
    model=merge_teacher(SOURCE,SOURCE/'teacher_fp/candidates/kappa_1.0')
    generate_set(SOURCE,model,'train','baseline_wm_train',load_native(model))


def refinement_init():
    c=read_json(SOURCE/'config.json')
    c.update(teacher_train_steps=600,teacher_lr=1e-5,teacher_checkpoint_every=200)
    freeze(RUN/'config.json',c)
    for rel in ['assets.json','prompts.json','carrier','result/threshold.json',
        'result/threshold_student.json','result/threshold_teacher.json','teacher_fp/calibration_target.json',
        'images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_test',
        'images/baseline_wm_train','images/unadapted','scores/unadapted.json']:
        share(SOURCE/rel,RUN/rel)
    cp=SOURCE/'teacher_fp/candidates/kappa_1.0'
    freeze(RUN/'teacher_choice.json',{'method':'full_structure32','teacher_kappa':1.,
        'source_checkpoint':str(cp),'source_adapter_sha256':sha256(cp/'adapter_model.safetensors'),
        'initial_effective_scale':1.,'refinement_steps':600,'num_bits':32,
        'quality_stopping_gates':False,'test_used_for_selection':False,'dataset_condition':'watermarked'})
    freeze(RUN/'refine_spec.json',read_json(BASE/'evidence/refine_spec.json'))
    freeze(RUN.parent/'code_manifest.json',{str(p.relative_to(ROOT)):sha256(p)
        for p in (ROOT/'core').glob('*.py')})


def generate():
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import load_native,generate_set,numerical_policy
    numerical_policy()
    folder=RUN/'teacher_fp/refined';model=merge_teacher(RUN,folder);pipe=load_native(model)
    for split in ['train','validation','test']:generate_set(RUN,model,split,'teacher_wm_'+split,pipe)
    freeze(RUN/'data_complete.json',{'adapter_sha256':sha256(folder/'adapter_model.safetensors')})


def report():
    from key_metrics import aggregate_metrics
    from semantic_report import teacher_quality
    key=read_json(RUN/'carrier/key.json')['signs']
    rows=read_json(RUN/'scores/teacher_wm_test.json')
    freeze(RUN/'result/teacher_key_metrics.json',aggregate_metrics([r['coordinates'] for r in rows],key))
    freeze(RUN/'result/teacher_quality.json',teacher_quality(RUN,'teacher_wm_validation'))
    freeze(RUN/'complete.json',{'role':'Structure32 teacher/data ready','completed':time.time()})


def stage(name,script,*args):
    marker=STATE/'stages'/(name+'.json')
    if marker.exists():return
    while True:
        free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free>65000:break
        write_json(STATE/'status.json',{'state':'waiting_for_resources','stage':name,'free_gpu_mib':free});time.sleep(30)
    write_json(STATE/'status.json',{'state':'running','stage':name,'pid':os.getpid(),'updated':time.time()})
    log=STATE/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:subprocess.run([sys.executable,str(script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    freeze(marker,{'completed':time.time(),'log':str(log)})


def supervise():
    STATE.mkdir(exist_ok=True)
    with (STATE/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            me=Path(__file__)
            stage('00_initialize',me,'initialize')
            stage('01_calibration_scores',HARNESS/'semantic_teacher.py','--run',SOURCE,'--stage','score',
                '--directories','teacher_clean_calibration','unadapted_calibration','unadapted')
            stage('02_freeze_loss_calibration',me,'calibration')
            stage('03_k1_3000',me,'k1')
            stage('04_baseline_images',me,'baseline')
            stage('05_refinement_initialize',me,'refinement-init')
            stage('06_structure_smoke',ROOT/'core/structure_margin_refine.py','--run',RUN,'--smoke')
            stage('07_structure600',ROOT/'core/structure_margin_refine.py','--run',RUN)
            stage('08_teacher_images',me,'generate')
            stage('09_teacher_scores',HARNESS/'semantic_teacher.py','--run',RUN,'--stage','score',
                '--directories','teacher_wm_validation','teacher_wm_test')
            stage('10_teacher_report',me,'report')
            stage('11_validation_cache',HARNESS/'validation.py','--run',RUN,'--stage','cache')
            stage('12_student_initialize',ROOT/'student12500/campaign.py','--initialize')
            stage('13_student_tests',ROOT/'student_tests.py')
            stage('14_student12500',ROOT/'student12500/campaign.py','--supervise')
            write_json(STATE/'status.json',{'state':'completed','updated':time.time()})
        except Exception as e:
            write_json(STATE/'status.json',{'state':'failed','error':repr(e),'updated':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','calibration','k1','baseline','refinement-init','generate','report','supervise']);a=p.parse_args()
    globals()[a.action.replace('-','_')]()

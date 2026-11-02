"""Evaluate .23 fullFT endpoint using unchanged rank32 pipeline and heldout images."""
from pathlib import Path
import json,time,os,sys,subprocess,importlib.util
OLD=Path('/var/lib/structure32_newprompts_rank32');P=Path('/var/lib/structure32_dit_sweep_20261101/pixart512_fullft')
def status(stage,**kw):(P/'evaluation_status.json').write_text(json.dumps({'stage':stage,'updated':time.time(),**kw}))
try:
    status('queued_waiting_for_fullft_training_on_23')
    while not (P/'result_ready.json').exists():time.sleep(30)
    result=P/'result_from_23'
    code=(OLD/'cross_arch_jobs.py').read_text()
    code=code.replace("SOURCE=ROOT/'coco_disjoint_source'","SOURCE=Path('/var/lib/structure32_newprompts_rank32/coco_disjoint_source')")
    code=code.replace("p=ROOT/'cross_arch'/name;p.mkdir(parents=True,exist_ok=True);return p","return ROOT")
    a=code.index("            load_adapter(pipe.transformer,p/'adapter')",code.index('def generate('))
    b=code.index("        folder=p/'images'/label",a)
    code=code[:a]+'''            from safetensors.torch import load_file
            expected={}
            for f in (p/'result_from_23/transformer').glob('*.safetensors'):expected.update(load_file(str(f)))
            pipe.transformer.load_state_dict(expected,strict=True)
            for n,t in pipe.transformer.state_dict().items():torch.testing.assert_close(t.cpu(),expected[n].to(t.dtype),atol=0,rtol=0)
            del expected
'''+code[b:]
    code=code.replace("'model_revision':MODELS[name]['revision']","'training_mode':'full transformer finetune','model_revision':MODELS[name]['revision']")
    code=code.replace("for label in ['unadapted','student']:","for label in ['student']:",1)
    (P/'images').mkdir(exist_ok=True)
    control=P/'images/unadapted'
    if not control.exists():control.symlink_to((OLD/'cross_arch/pixart512/images/unadapted').resolve(),target_is_directory=True)
    (P/'evaluate_endpoint.py').write_text(code)
    for n in ['model_identity.json']:
        target=OLD/'cross_arch/pixart512'/n
        # Existing transport identity points to the same model snapshot on .21.
        assert json.loads((P/n).read_text())==json.loads(target.read_text())
    import shutil
    shutil.copy2(OLD/'acceptance_metrics.py',P/'acceptance_metrics.py')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(P),PYTHONUNBUFFERED='1')
    for stage in ['generate','evaluate']:
        marker=P/(stage+'.complete.json')
        if marker.exists():continue
        status(stage)
        with (P/(stage+'.log')).open('a') as log:subprocess.run(['/var/lib/vllm/env/bin/python',str(P/'evaluate_endpoint.py'),'pixart512',stage],env=env,cwd=P,check=True,stdout=log,stderr=subprocess.STDOUT)
        marker.write_text(json.dumps({'completed':time.time()}))
    status('completed')
except Exception as exc:status('failed',error=repr(exc));raise

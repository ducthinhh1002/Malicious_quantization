"""DiffusionDB prompts -> native SERUM/GROW teacher -> existing SD1.4 r4 trainer."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import unicodedata
import urllib.request

PROJECT = Path('/var/lib/baseline_mentor35')
BASE = PROJECT / 'run'
WORK = BASE / 'ood_diffusiondb_20261111'
REVISION = 'fb620fbe49fa4420e0734bd9c0df11f51176b61f'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def prepare():
    import pyarrow.parquet as pq
    from transformers import CLIPTokenizer
    if (WORK / 'prompts.json').exists():
        protocol = read(WORK / 'protocol.json')
        assert sha(WORK / 'prompts.json') == protocol['prompt_sha256']
        return
    metadata = WORK / 'source/metadata.parquet'
    # A concurrent download is allowed to finish before data preparation.
    deadline = time.monotonic() + 1800
    while metadata.stat().st_size != 194548652:
        if time.monotonic() > deadline:
            raise RuntimeError('DiffusionDB metadata download incomplete; inspect source/download.log')
        time.sleep(10)
    assets = read(PROJECT / 'assets.json')
    tokenizers = [CLIPTokenizer.from_pretrained(assets[k], subfolder='tokenizer', local_files_only=True)
                  for k in ['teacher_path', 'student_path']]
    excluded = set()
    excluded_tokens = [set(), set()]
    manifests = [PROJECT / 'benchmark/shared/data/prompts.json',
                 Path('/var/lib/structure32_student12500/runs/structure32/prompts.json')]
    old_rows = []
    for path in manifests:
        for rows in read(path).values():
            if isinstance(rows, list):
                old_rows.extend(r for r in rows if isinstance(r, dict) and 'text' in r)
    old_rows.extend(read(PROJECT / 'calibration.json'))
    for row in old_rows:
        excluded.add(normalized(row['text']))
    for k, tokenizer in enumerate(tokenizers):
        encoded = tokenizer([r['text'] for r in old_rows], truncation=True, max_length=77)['input_ids']
        excluded_tokens[k].update(tuple(t) for t in encoded)
    table = pq.ParquetFile(metadata)
    names = table.schema.names
    columns = ['prompt', 'image_name'] + (['prompt_nsfw'] if 'prompt_nsfw' in names else [])
    candidates = []
    for batch in table.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            text = row['prompt'].strip()
            if not 20 <= len(text) <= 500 or not text.isascii():
                continue
            if float(row.get('prompt_nsfw') or 0) > .1:
                continue
            key = normalized(text)
            if key in excluded:
                continue
            tokens = [tuple(t(text, truncation=True, max_length=77)['input_ids']) for t in tokenizers]
            if any(v in excluded_tokens[k] for k, v in enumerate(tokens)):
                continue
            excluded.add(key)
            for k, v in enumerate(tokens):
                excluded_tokens[k].add(v)
            candidates.append(dict(text=text, source_image=row['image_name']))
            if len(candidates) == 14000:
                break
        if len(candidates) == 14000:
            break
    assert len(candidates) >= 7000
    random.Random(20261111).shuffle(candidates)
    selected = candidates[:7000]
    for i, row in enumerate(selected):
        row['image_id'] = 81000000 + i
    prompts = dict(train=selected[:5000], validation=selected[5000:6000],
                   test=selected[6000:], calibration=[])
    write(WORK / 'prompts.json', prompts)
    write(WORK / 'protocol.json', dict(dataset='poloclub/diffusiondb', revision=REVISION,
        metadata_sha256=sha(metadata), prompt_sha256=sha(WORK/'prompts.json'),
        counts={k:len(v) for k,v in prompts.items()}, sample_seed=20261111,
        filtering='ASCII 20..500 chars; prompt_nsfw<=0.1 if available; unique normalized and effective77-token teacher/student prompts; exclude historical/newCOCO and calibration',
        candidate_selection='first14000 eligible rows in metadata order, seeded shuffle, first7000',
        teacher='existing native SD2.1 SERUM or GROW', student='clean SD1.4 LoRA r4 attention_conv',
        steps=12500, batch=4, seed=0, manipulation='none',
        threshold='original clean teacher threshold fixed; OOD clean test FPR measured separately',
        metrics=dict(serum='native TPR at frozen threshold + measured OOD clean FPR',grow='native 16-bit exact M-ACC and BA'),
        pretrained_teacher_training_membership='not fully auditable for public pretrained SERUM; excludes available local COCO manifests'))
    print('PREPARED_OOD', read(WORK/'protocol.json'), flush=True)


def generation(method):
    import torch
    from common import save_images, manifest
    from benchmark.scripts import mentor35
    from benchmark.scripts.additional_clean import verify_images
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = mentor35.setup(BASE, method)
    prompts = read(WORK/'prompts.json')
    folder = WORK/method
    batch = 1 if method == 'grow' else 8
    for split, marked in [('train', True), ('validation', True), ('test', True), ('test', False)]:
        rows = prompts[split]
        label = 'teacher_wm_' + split if marked else 'teacher_clean_test'
        images = folder/'images'/label
        images.mkdir(parents=True, exist_ok=True)
        if verify_images(images, rows):
            continue
        for start in range(0,len(rows),batch):
            ids = list(range(start,min(start+batch,len(rows))))
            if not all((images/f'sample_{i:05d}.png').exists() for i in ids):
                texts = [rows[i]['text'] for i in ids]
                seeds = [rows[i]['image_id'] for i in ids]
                if marked:
                    output = model.generate(texts, seeds)
                else:
                    with torch.inference_mode():
                        output = model.pipe(texts, height=512, width=512, num_inference_steps=50,
                            guidance_scale=7.5, generator=[torch.Generator('cuda').manual_seed(s) for s in seeds]).images
                save_images(output,images,ids)
            write(folder/'status.json',dict(stage='teacher_generation',split=label,n=start+len(ids),total=len(rows),pid=os.getpid()))
            if (start+len(ids)) % 100 == 0:
                print('GENERATED',method,label,start+len(ids),'/',len(rows),flush=True)
        manifest(images,rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train(method):
    from benchmark.shared.student_training.integration import sd14
    from benchmark.scripts.additional_clean import verify_checkpoint
    assets = read(PROJECT/'assets.json')
    job = WORK/method/'student'
    data = WORK/method/'images/teacher_wm_train'
    v = sd14(job,assets['student_path'],data,4)
    v['config'].update(max_train_steps=12500,checkpoint_steps=[])
    link = job/'images/teacher_wm_train'
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():
        link.symlink_to(data,target_is_directory=True)
    assert link.resolve()==data.resolve()
    branch=job/'students/sd14_r4'
    identity=read(v['identity'])
    cmd=v['builder'](v['config'],v['assets'],job,v['branch'])
    cmd[cmd.index('--checkpointing_steps')+1]='1250'
    assert cmd[cmd.index('--max_train_steps')+1]=='12500'
    assert cmd[cmd.index('--rank')+1]=='4'
    checkpoints=sorted(branch.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    if checkpoints:
        verify_checkpoint(checkpoints[-1],identity)
        if int(checkpoints[-1].name.split('-')[-1])<12500:
            cmd+=['--resume_from_checkpoint',str(checkpoints[-1])]
    if not checkpoints or int(checkpoints[-1].name.split('-')[-1])<12500:
        write(job/'training_command.json',dict(command=cmd,prompt_sha256=sha(WORK/'prompts.json')))
        env=os.environ.copy()
        for key in ['BASELINE_TRAIN_ATTACK','BASELINE_PIXEL_INDEX']:
            env.pop(key,None)
        subprocess.run([sys.executable,str(v['helper']/'driver.py'),'--trainer',cmd[1],
                        '--identity-file',str(v['identity']),*cmd[2:]],cwd=PROJECT,env=env,check=True)
    verify_checkpoint(branch/'checkpoint-12500',identity)
    write(job/'trained.json',dict(steps=12500,rank=4))


def evaluate(method):
    import torch
    from benchmark.scripts import mentor35
    from benchmark.scripts.additional_clean import verify_checkpoint,verify_images
    from benchmark.shared.evaluation.student_images import generate
    from benchmark.shared.evaluation.native_metrics import evaluate as metric
    assets=read(PROJECT/'assets.json')
    folder=WORK/method
    job=folder/'student'
    output=job/'evaluation/12500'
    weights=output/'students/sd14_r4'
    weights.parent.mkdir(parents=True,exist_ok=True)
    cp=job/'students/sd14_r4/checkpoint-12500'
    verify_checkpoint(cp,read(job/'resume_identity.json'))
    if not weights.exists():
        weights.symlink_to(cp,target_is_directory=True)
    assert weights.resolve()==cp.resolve()
    images=output/'test_images'
    rows=read(WORK/'prompts.json')['test']
    if not verify_images(images,rows):
        generate('sd14','r4',assets['student_path'],output,rows,images)
    model=mentor35.setup(BASE,method)
    threshold=read(BASE/'teachers/serum/thresholds/clean.json') if method=='serum' else None
    results={}
    for label,source in [('teacher',folder/'images/teacher_wm_test'),
                         ('clean_teacher',folder/'images/teacher_clean_test'),('student',images)]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        values,bits=mentor35.coordinates(model,source,'clean')
        write(output/(label+'_coordinates.json'),values)
        results[label]=metric(method,values,getattr(model,'key',None),threshold,bits)
        if method == 'serum':
            results[label]['metric'] = 'TPR at frozen COCO clean-teacher 1% calibration threshold'
        if method == 'serum' and label == 'clean_teacher':
            results[label]['metric'] = 'empirical OOD clean-teacher FPR at frozen threshold'
        print('RESULT',method,label,results[label],flush=True)
    write(output/'results.json',dict(method=method,student='SD1.4 LoRA r4',steps=12500,
        dataset='DiffusionDB',prompt_sha256=sha(WORK/'prompts.json'),n=1000,results=results))


def run(method):
    import fcntl
    folder=WORK/method
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in ['generation','train','evaluate']:
            write(folder/'status.json',dict(stage=stage,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
            subprocess.run([sys.executable,'-u','-m','benchmark.scripts.ood_clean_20261111',stage,
                            '--method',method],cwd=PROJECT,check=True)
        write(folder/'status.json',dict(stage='completed'))


if __name__=='__main__':
    from benchmark.shared.process_title import configure
    configure()
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','generation','train','evaluate'])
    parser.add_argument('--method',choices=['serum','grow'])
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        globals()[args.action](args.method)

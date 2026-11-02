"""Prepare disjoint COCO prompts and frozen refine600 images for both DiT students."""
from pathlib import Path
import sys,json,time,random,re,unicodedata,urllib.request,zipfile,fcntl,os,concurrent.futures
import run as base
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'coco_disjoint_source'
base.SOURCE=SOURCE
from common import read_json,write_json,freeze,sha256

def status(stage,**kw):
    write_json(ROOT/'coco_data_status.json',{'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw})

def prepare():
    from transformers import CLIPTokenizer
    SOURCE.mkdir(exist_ok=True)
    assets=read_json(base.OLD/'assets.json')
    archive=ROOT/'inputs/annotations_trainval2014.zip'
    if not archive.exists():
        status('download_coco_annotations')
        tmp=archive.with_suffix('.partial')
        total=252872794;block=4*1024*1024
        def fetch(start):
            end=min(total,start+block)-1
            for attempt in range(5):
                try:
                    req=urllib.request.Request(assets['coco_url']+'&chunk='+str(start),headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=30) as src:
                        assert src.status==206 and src.headers['Content-Range']==f'bytes {start}-{end}/{total}'
                        data=src.read();assert len(data)==end-start+1
                    return start,data
                except Exception:
                    if attempt==4:raise
                    time.sleep(2)
        with tmp.open('wb') as out,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            out.truncate(total);done=0
            futures=[pool.submit(fetch,start) for start in range(0,total,block)]
            for future in concurrent.futures.as_completed(futures):
                start,data=future.result();out.seek(start);out.write(data);done+=len(data)
                status('download_coco_annotations',downloaded_bytes=done,total_bytes=total)
        assert sha256(tmp)==assets['coco_sha256']
        tmp.replace(archive)
    assert sha256(archive)==assets['coco_sha256']
    assert sha256(base.OLD/'teacher_fp/refined/adapter_model.safetensors')==base.EXPECTED
    old=[r for rows in read_json(base.OLD/'prompts.json').values() for r in rows]
    normalize=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
    seen={normalize(r['text']) for r in old};images={r['image_id'] for r in old}
    tokens=[CLIPTokenizer.from_pretrained(assets[k],subfolder='tokenizer',local_files_only=True) for k in ['teacher_clean','student']]
    prefixes=[set(tuple(x) for x in t([r['text'] for r in old],truncation=True,max_length=77)['input_ids']) for t in tokens]
    rejected={'image_id':0,'text':0,'effective_tokens':0}
    def pick(rows,n,seed):
        random.Random(seed).shuffle(rows);chosen=[]
        for row in rows:
            if row['image_id'] in images:rejected['image_id']+=1;continue
            text=row['caption'].strip();norm=normalize(text)
            if norm in seen:rejected['text']+=1;continue
            ids=[tuple(t(text,truncation=True,max_length=77)['input_ids']) for t in tokens]
            if any(x in p for x,p in zip(ids,prefixes)):rejected['effective_tokens']+=1;continue
            images.add(row['image_id']);seen.add(norm)
            for x,p in zip(ids,prefixes):p.add(x)
            chosen.append({'image_id':row['image_id'],'caption_id':row['id'],'text':text})
            if len(chosen)==n:return chosen
        raise RuntimeError('Insufficient disjoint COCO captions')
    with zipfile.ZipFile(archive) as z:
        train=pick(json.loads(z.read('annotations/captions_train2014.json'))['annotations'],5000,20261102)
        held=pick(json.loads(z.read('annotations/captions_val2014.json'))['annotations'],2000,20261103)
    prompts={'train':train,'validation':held[:1000],'test':held[1000:],'calibration':[]}
    freeze(SOURCE/'prompts.json',prompts)
    assets['prompt_sha256']=sha256(SOURCE/'prompts.json');assets.pop('new_prompt_dataset',None)
    freeze(SOURCE/'assets.json',assets)
    c=read_json(base.OLD/'config.json');c.update(ranks=[32],max_train_steps=20000,checkpoint_steps=[20000])
    freeze(SOURCE/'config.json',c)
    for name in ['carrier','teacher_fp/refined','teacher_fp/calibration_target.json','result/threshold.json','result/threshold_teacher.json','result/threshold_student.json']:
        base.link(base.OLD/name,SOURCE/name)
    freeze(SOURCE/'teacher_choice.json',{'source_checkpoint':str(base.OLD/'teacher_fp/refined'),'source_adapter_sha256':base.EXPECTED,'teacher_retrained':False,'num_bits':32})
    freeze(ROOT/'coco_prompt_audit.json',{'counts':{k:len(v) for k,v in prompts.items()},'old_rows_excluded':len(old),'original_image_overlap':0,'normalized_text_overlap':0,'effective_77_token_overlap':0,'cross_split_overlap':0,'rejections':rejected,'annotation_sha256':sha256(archive),'prompt_sha256':sha256(SOURCE/'prompts.json'),'seed_train':20261102,'seed_heldout':20261103,'interpretation':'same COCO dataset, disjoint prompts and image IDs; not a claim of out-of-distribution data'})
    freeze(ROOT/'coco_prepared.json',{'completed':time.time()})

if __name__=='__main__':
    with (ROOT/'coco_data.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (ROOT/'coco_prepared.json').exists():status('prepare_disjoint_coco');prepare()
            status('queued_waiting_for_sd14_completion')
            while read_json(ROOT/'status.json').get('stage')!='completed':time.sleep(30)
            status('generate_frozen_teacher_train5000');base.generate_worker('train')
            status('generate_heldout_and_clean_references');base.generate_worker('controls')
            freeze(ROOT/'coco_data_complete.json',{'completed':time.time(),'prompt_sha256':sha256(SOURCE/'prompts.json')})
            status('completed')
        except Exception as exc:status('failed',error=repr(exc));raise

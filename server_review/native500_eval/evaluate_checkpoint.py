"""Read-only sidecar validation of an in-flight native teacher snapshot."""
import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.source))
    sys.path.insert(1, str(a.source/'new_watermark'))
    import new_watermark.paths
    from new_watermark.paths import read_json, write_json, sha256
    from new_watermark.data import load_base, generate, verify_images
    from new_watermark.native_unet import load_reader, reader_features, keyed_scores, key_bits
    from student import load_adapter, checkpoint_hashes
    from report import wilson
    import numpy as np
    import torch
    from PIL import Image
    import lpips

    a.output.mkdir(parents=True, exist_ok=True)
    config = read_json(a.run/'config.json')
    prompts = read_json(a.run/'prompts.json')
    rows = prompts['pilot_eval']
    ids = {r['image_id'] for r in rows}
    texts = {r['text'].strip().casefold() for r in rows}
    for split in ('design_fit', 'teacher_train', 'meta_probe', 'calibration', 'pilot_train', 'train', 'validation', 'test'):
        assert not ids.intersection(r['image_id'] for r in prompts[split]), split
        assert not texts.intersection(r['text'].strip().casefold() for r in prompts[split]), split
    snapshot = a.run/'teacher/snapshot-500'
    weights = checkpoint_hashes(snapshot)
    detector = read_json(a.run/'result/detector.json')
    clean = a.run/'images/clean_pilot_eval'
    verify_images(clean, len(rows))
    generation = read_json(clean/'generation.json')
    assert generation['prompts'] == rows
    assert generation['seed'] == config['seed']
    assert generation['steps'] == config['inference_steps']
    assert generation['dtype'] == 'torch.float16'
    write_json(a.output/'config.json', {**config, 'inference_batch': 1})
    write_json(a.output/'assets.json', read_json(a.run/'assets.json'))
    write_json(a.output/'prompts.json', {'pilot_eval': rows})
    identity = {'snapshot': str(snapshot), 'hashes': weights,
                'detector_sha256': sha256(a.run/'result/detector.json'),
                'split': 'pilot_eval', 'n': len(rows), 'test_used': False,
                'reader_or_threshold_updated': False, 'train_meta_overlap': False,
                'clean_generation_sha256': sha256(clean/'generation.json'),
                'note': 'Development validation; batch 1 vs cached clean batch 4 may cause minor floating point differences.'}
    write_json(a.output/'protocol.json', identity)
    write_json(a.output/'status.json', {'status': 'waiting_for_memory'})
    while torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        print('Waiting for 18 GiB free VRAM', flush=True)
        time.sleep(30)
    total = torch.cuda.mem_get_info()[1]
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / total)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    pipe = load_base(a.output, dtype=torch.float32)
    load_adapter(pipe.unet, snapshot, pipe)
    pipe.unet.eval()
    # Evaluate the exact trained adapter representation. This validation does not
    # certify a fused export: that needs its own numerical equivalence audit.
    identity['representation'] = 'base + original trained adapter, not fused export'
    write_json(a.output/'protocol.json', identity)
    pipe.to(dtype=torch.float16)
    torch.cuda.empty_cache()
    write_json(a.output/'status.json', {'status': 'generating', 'n': len(rows)})
    generate(a.output, pipe, 'teacher500', 'pilot_eval', identity)
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    reader = load_reader(a.output)
    perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    records = []
    write_json(a.output/'status.json', {'status': 'scoring'})
    def pixels(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert('RGB'), dtype=np.float32)/255).permute(2,0,1)[None].cuda()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            name = f'sample_{i:06d}.png'
            x = pixels(a.output/'images/teacher500'/name)
            y = pixels(clean/name)
            fx, fy = reader_features(reader, x), reader_features(reader, y)
            sx, sy = keyed_scores(fx, detector['keys'])[0], keyed_scores(fy, detector['keys'])[0]
            target = key_bits(detector['keys'][:1], x.device)
            mse = float((x-y).square().mean())
            r = {'index': i, 'image_id': row['image_id'], 'psnr': -10*math.log10(max(mse,1e-12)),
                 'lpips': float(perceptual(x*2-1,y*2-1).mean()),
                 'bit_accuracy': float(((fx[:,1:]>0)==target.bool()).float().mean()),
                 'clean_bit_accuracy': float(((fy[:,1:]>0)==target.bool()).float().mean())}
            for j, key in enumerate(detector['keys']):
                r[f'teacher_score_{key}'] = float(sx[j]); r[f'clean_score_{key}'] = float(sy[j])
            records.append(r)
            if (i+1)%16 == 0: print(f'Scored {i+1}/{len(rows)}', flush=True)
    results = []
    for j,key in enumerate(detector['keys']):
        for branch in ('teacher','clean'):
            hits = sum(r[f'{branch}_score_{key}'] >= detector['thresholds'][j] for r in records)
            results.append({'branch': branch, 'key': key, 'correct_key': j==0,
                            'n': len(rows), 'detected': hits, 'rate': hits/len(rows),
                            'ci95': wilson(hits,len(rows)), 'threshold': detector['thresholds'][j]})
    summary = {'checkpoint': 500, 'protocol': identity, 'total_keyset_nominal_fpr': detector['total_keyset_nominal_fpr'],
               'nominal_per_key_fpr': detector['nominal_per_key_fpr'], 'detection': results,
               'means': {k: float(np.mean([r[k] for r in records])) for k in ('psnr','lpips','bit_accuracy','clean_bit_accuracy')}}
    assert checkpoint_hashes(snapshot) == weights, 'Source snapshot changed'
    assert sha256(a.run/'result/detector.json') == identity['detector_sha256'], 'Detector changed'
    with (a.output/'per_image.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    write_json(a.output/'summary.json', summary)
    write_json(a.output/'status.json', {'status':'complete'})
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()

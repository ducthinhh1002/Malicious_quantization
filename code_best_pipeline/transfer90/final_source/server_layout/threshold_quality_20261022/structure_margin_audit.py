"""Paired structural diagnostics on the same fixed test references.

This metric is also optimized by the new teacher, so improvements on it alone
are not independent validation. Existing LPIPS/FID/KID/CLIP and detection stay.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import research_queue as q
from common import read_json, freeze, sha256, image_name
from semantic_teacher import pixels
from structure_margin_objectives import ms_ssim


def audit(run):
    n=read_json(run/'config.json')['test_n']
    rows={}
    for label,reference in [('teacher_wm_test','teacher_clean_test'),(q.BRANCH,'unadapted')]:
        values=[]
        for start in range(0,n,8):
            ids=range(start,min(start+8,n))
            with torch.no_grad():
                a=(pixels([run/'images'/label/image_name(i) for i in ids])+1)/2
                b=(pixels([run/'images'/reference/image_name(i) for i in ids])+1)/2
                values+=ms_ssim(a,b).cpu().tolist()
        rows[label]={'reference':reference,'n':n,'mean_ms_ssim':float(np.mean(values)),
            'p10_ms_ssim':float(np.quantile(values,.1)),'per_image':values,
            'generation_hashes_sha256':sha256(run/'images'/label/'image_hashes.json')}
    freeze(run/'result/structural_diagnostics.json',{'branches':rows,
        'selection_input':False,'metric_used_in_teacher_training':True,
        'caveat':'Paired structural fidelity, not anatomy, aesthetics or independent preference.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    audit(p.parse_args().run)

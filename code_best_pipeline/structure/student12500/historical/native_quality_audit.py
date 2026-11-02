"""Additional diagnostics, not a learned quality score or an automatic human rating."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import research_queue as q
from common import read_json, freeze, image_name, sha256


def pixel_statistics(folder,n):
    clipping,saturation=[],[]
    for i in range(n):
        pixels=np.asarray(Image.open(folder/image_name(i)).convert('RGB'),dtype=np.float32)/255
        clipping.append(float(((pixels<=1/255)|(pixels>=254/255)).mean()))
        high,low=pixels.max(-1),pixels.min(-1)
        saturation.append(float(((high-low)/np.maximum(high,1e-6)).mean()))
    return {'clipped_channel_fraction_mean':float(np.mean(clipping)),
            'clipped_channel_fraction_p95':float(np.quantile(clipping,.95)),
            'mean_hsv_saturation':float(np.mean(saturation)),
            'interpretation':'Color/clipping diagnostics only; do not measure anatomical correctness or universal image quality'}


def audit(run):
    summary=read_json(run/'result/student_summary.json');config=read_json(run/'config.json')
    references={'unadapted':run/'images/unadapted',
                'student':run/'images'/q.BRANCH}
    result={'n':config['test_n'],'condition':read_json(run/'teacher_choice.json')['dataset_condition'],
            'pixel_statistics':{name:pixel_statistics(path,config['test_n']) for name,path in references.items()},
            'lpips_quantiles':{str(p):float(np.quantile(summary['lpips_per_image'],p)) for p in [.1,.5,.9,.95]},
            'summary_sha256':sha256(run/'result/student_summary.json'),'human_ratings_collected':False}
    freeze(run/'result/visual_diagnostics.json',result)
    # Precommit a reproducible panel. No cherry-picking or automatic labeling of defects.
    prompts=read_json(run/'prompts.json')['test'];rng=np.random.default_rng(20261025)
    ids=sorted(rng.choice(len(prompts),size=min(200,len(prompts)),replace=False).tolist())
    baseline=q.ROOT/'kappa1_manojb_5k/students/k1_evolution/images'/q.BRANCH
    panel,key=[],[]
    for i in ids:
        swap=bool(rng.integers(2));pair=[str(baseline/image_name(i)),str(run/'images'/q.BRANCH/image_name(i))]
        if swap:pair.reverse()
        panel.append({'index':i,'prompt':prompts[i]['text'],'A':pair[0],'B':pair[1],
                      'preference':None,'A_severe_defect':None,'B_severe_defect':None,
                      'rubric':'blindly rate face/hand/body/object defects, natural appearance and prompt adherence; tie allowed'})
        key.append({'index':i,'candidate':'A' if swap else 'B'})
    freeze(run/'result/blind_panel.json',{'seed':20261025,'samples':panel,'ratings_complete':False})
    freeze(run/'result/blind_panel_key.json',key)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    audit(parser.parse_args().run)

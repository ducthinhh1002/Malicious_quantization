"""Student evaluation: aggregate bit metrics, paired LPIPS and image utility."""
import numpy as np
import torch
from common import read_json,freeze,image_name
from key_metrics import aggregate_metrics


def summary(run,branch):
    import lpips
    from semantic_teacher import pixels
    key=read_json(run/'carrier/key.json')['signs'];assert len(key)==32
    model=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    for label,outname in [(branch,'student_summary.json')]:
        rows=read_json(run/'scores'/f'{label}.json')
        result={'key_recovery':aggregate_metrics([r['coordinates'] for r in rows],key)}
        distances=[]
        a=read_json(run/'images/unadapted/generation.json');b=read_json(run/'images'/label/'generation.json')
        assert {k:v for k,v in a.items() if k!='checkpoint'}=={k:v for k,v in b.items() if k!='checkpoint'}
        for start in range(0,1000,8):
            names=[image_name(i) for i in range(start,min(start+8,1000))]
            with torch.inference_mode():distances.extend(model(
                pixels([run/'images'/label/n for n in names]),pixels([run/'images/unadapted'/n for n in names])).flatten().cpu().tolist())
        result.update(lpips=float(np.mean(distances)),lpips_per_image=distances,
            utility=read_json(run/'result/utility.json')['branches'][label],
            selected_checkpoint=read_json(run/'result/selection'/f'{branch}.json') if label==branch else {'selected_step':20000},
            test_used_for_selection=False,primary_metrics=['bit_accuracy','full_key_recovery_rate'])
        freeze(run/'result'/outname,result)
    freeze(run/'complete.json',{'completed':__import__('time').time(),'selection':'fixed20000 by user; validation selection skipped'})

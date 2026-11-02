import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path('/root/workspace/serum')
def wilson(k,n):
    p=k/n; z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]

def main():
    cal_path=ROOT/'metrics/serum_threshold_1pct.json'
    cal=json.loads(cal_path.read_text()); threshold=np.float32(cal['threshold'])
    clean=np.load(ROOT/'metrics/teacher_clean_train_scores.npy')
    wm=np.load(ROOT/'metrics/teacher_wm_train_scores.npy')
    assert len(clean)==len(wm)==5000
    assert np.isfinite(clean).all() and np.isfinite(wm).all()
    fp=int((clean>=threshold).sum()); tp=int((wm>=threshold).sum())
    result={'evaluation':'teacher adapt_train; separate from clean calibration set',
            'threshold':float(threshold),'threshold_file_sha256':hashlib.sha256(cal_path.read_bytes()).hexdigest(),
            'model':cal['model'],'clean_count':len(clean),'watermarked_count':len(wm),
            'false_positives':fp,'true_positives':tp,'fpr':fp/len(clean),'tpr':tp/len(wm),
            'fpr_wilson_95':wilson(fp,len(clean)),'tpr_wilson_95':wilson(tp,len(wm)),
            'roc_auc':float(roc_auc_score(np.r_[np.zeros(len(clean)),np.ones(len(wm))],np.r_[clean,wm])),
            'clean_score_mean':float(clean.mean()),'watermarked_score_mean':float(wm.mean()),
            'scope':'image-level detection only; no student radioactivity evaluation'}
    (ROOT/'metrics/teacher_image_detection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

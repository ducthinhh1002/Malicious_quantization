"""Diagnostic clean-student calibration only; never overwrites the frozen threshold."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

r=Path('/root/workspace/serum')
clean=np.load(r/'metrics/C00_seed0_scores.npy')
wm=np.load(r/'metrics/C10_seed0_scores.npy')
assert len(clean)==len(wm)==5000
threshold=np.nextafter(np.sort(clean)[::-1][50],np.float32(np.inf),dtype=np.float32)
ranks=rankdata(np.concatenate([clean,wm]))
auc=(ranks[len(clean):].sum()-len(wm)*(len(wm)+1)/2)/(len(clean)*len(wm))
report={'scope':'Post-hoc diagnostic only: same C00 outputs set threshold and measure FPR; not an independent calibration/evaluation result. Original threshold remains frozen.',
        'diagnostic_C00_threshold':float(threshold),
        'diagnostic_C00_fpr':float(np.mean(clean>=threshold)),
        'diagnostic_C10_tpr':float(np.mean(wm>=threshold)),
        'C10_vs_C00_roc_auc':float(auc),
        'C00_mean_score':float(clean.mean()),'C10_mean_score':float(wm.mean())}
(r/'metrics/student_threshold_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

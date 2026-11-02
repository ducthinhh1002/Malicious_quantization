"""Compare paired student detections under the existing frozen threshold."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

root = Path('/root/workspace/serum')
threshold = np.float32(json.loads((root/'metrics/serum_threshold_1pct.json').read_text())['threshold'])
c10 = np.load(root/'metrics/C10_seed0_scores.npy')
assert len(c10) == 5000
result = {'threshold': float(threshold), 'C10_rate': float(np.mean(c10 >= threshold)), 'controls': {}}
for name in ('SD14_unadapted', 'C00_seed0'):
    path = root/f'metrics/{name}_scores.npy'
    if not path.exists():
        continue
    scores = np.load(path)
    assert len(scores) == len(c10)
    c = scores >= threshold
    w = c10 >= threshold
    wm_only = int(np.sum(w & ~c))
    control_only = int(np.sum(c & ~w))
    discordant = wm_only + control_only
    ci = binomtest(int(c.sum()), len(c)).proportion_ci(confidence_level=.95)
    result['controls'][name] = {
        'num_images': len(c), 'positives': int(c.sum()), 'rate': float(c.mean()),
        'rate_ci95': [ci.low, ci.high],
        'C10_minus_control_percentage_points': float((w.mean()-c.mean())*100),
        'C10_only_positive': wm_only, 'control_only_positive': control_only,
        'paired_exact_mcnemar_p_two_sided': binomtest(wm_only, discordant, .5).pvalue if discordant else 1.,
    }
result['interpretation_rule'] = 'Assess C10 against clean student controls; frozen teacher threshold is unchanged. Statistical significance alone does not establish a useful radioactivity baseline.'
(root/'metrics/baseline_controls.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

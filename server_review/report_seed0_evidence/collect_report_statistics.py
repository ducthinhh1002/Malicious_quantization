"""Read-only supplementary statistics for the seed-0 report."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

root = Path('/root/workspace/serum')
names = ['C00_seed0', 'C10_seed0', 'C01_step0.25_seed0', 'C11_step0.25_seed0', 'C01_step1.00_seed0', 'C11_step1.00_seed0']
scores = {n: np.load(root / 'metrics' / (n + '_scores.npy')).reshape(-1) for n in names}
clip = {n: np.load(root / 'metrics' / (n + '_clipscore.npy')).reshape(-1) for n in names}
assert all(len(x) == 5000 and np.isfinite(x).all() for x in [*scores.values(), *clip.values()])
out = {'n': 5000, 'rows': {}, 'paired': [], 'bootstrap_replicates': 1000, 'bootstrap_seed': 20261011}
for n in names:
    selection = json.loads((root / 'students' / n / 'validation_selection.json').read_text())
    out['rows'][n] = {'score_mean': float(scores[n].mean()), 'score_median': float(np.median(scores[n])), 'best': selection['best']}
def auc(pos, neg):
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
out['score_auc'] = {'raw_C10_vs_C00': auc(scores[names[1]], scores[names[0]]), 'mild_C11_vs_C01': auc(scores[names[3]], scores[names[2]]), 'strong_C11_vs_C01': auc(scores[names[5]], scores[names[4]])}
rng = np.random.default_rng(20261011)
for strength, clean, wm in [(0.25, names[2], names[3]), (1.0, names[4], names[5])]:
    interaction = (scores[wm] - scores[clean]) - (scores[names[1]] - scores[names[0]])
    clean_clip_diff = clip[clean] - clip[names[0]]
    boots = np.empty((1000, 2))
    for b in range(1000):
        idx = rng.integers(0, 5000, 5000)
        boots[b] = [interaction[idx].mean(), clean_clip_diff[idx].mean()]
    out['paired'].append({'strength': strength, 'delta_launder_mean_score': float(interaction.mean()), 'delta_launder_ci95': np.quantile(boots[:,0], [.025,.975]).tolist(), 'C01_minus_C00_clipscore': float(clean_clip_diff.mean()), 'clean_clipscore_difference_ci95': np.quantile(boots[:,1], [.025,.975]).tolist()})
print(json.dumps(out, indent=2))

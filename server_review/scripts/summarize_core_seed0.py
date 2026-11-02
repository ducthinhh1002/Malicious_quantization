"""Report uncertainty, paired effects, and the watermark/utility frontier."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path('/root/workspace/serum'); threshold=np.float32(json.loads((r/'metrics/serum_threshold_1pct.json').read_text())['threshold'])
names=['C00_seed0','C10_seed0']+[f'{c}_step{s}_seed0' for s in ['0.25','1.00'] for c in ['C01','C11']]
rng=np.random.default_rng(42000)
def ci(values):
    vals=np.array([np.mean(values[rng.integers(0,len(values),len(values))]) for _ in range(1000)])
    return [float(v) for v in np.quantile(vals,[.025,.975])]
rows=[]; wm={}; clip={}
for name in names:
    scores=np.load(r/f'metrics/{name}_scores.npy'); wm[name]=(scores>=threshold).astype(float)
    clip[name]=np.load(r/f'metrics/{name}_clipscore.npy')
    assert len(scores)==len(clip[name])==5000
    utility=json.loads((r/f'metrics/{name}_utility.json').read_text())
    selection=json.loads((r/f'students/{name}/validation_selection.json').read_text())
    rows.append({'student':name,'tpr':float(wm[name].mean()),'tpr_bootstrap_ci95':ci(wm[name]),
                 'clipscore':float(clip[name].mean()),'clipscore_bootstrap_ci95':ci(clip[name]),
                 'best_validation_mse':selection['best']['mse'],**utility['distribution_metrics']})
effects=[]
for step in ['0.25','1.00']:
    c11=f'C11_step{step}_seed0'; c01=f'C01_step{step}_seed0'
    reduction=wm['C10_seed0']-wm[c11]
    interaction=(wm['C10_seed0']-wm['C00_seed0'])-(wm[c11]-wm[c01])
    delta=clip[c11]-clip['C10_seed0']
    effects.append({'step':float(step),'C10_minus_C11_tpr':float(reduction.mean()),
                   'paired_tpr_difference_ci95':ci(reduction),
                   'watermark_effect_reduction_adjusted_for_clean_control':float(interaction.mean()),
                   'interaction_ci95':ci(interaction),
                   'C11_minus_C10_clipscore':float(delta.mean()),'paired_clipscore_difference_ci95':ci(delta)})
result={'seed':0,'scope':'Exploratory core run; C10 reproduction remains weak. One seed cannot establish stable attack success.',
        'threshold':float(threshold),'bootstrap_replicates':1000,'rows':rows,'paired_effects':effects}
(r/'metrics/core_seed0_summary.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(figsize=(7,5))
for row in rows:
    if row['student'].startswith(('C10','C11')):
        ax.scatter(row['clipscore'],100*row['tpr'])
        ax.annotate(row['student'],(row['clipscore'],100*row['tpr']),fontsize=8,xytext=(5,5),textcoords='offset points')
ax.set(xlabel='Prompt-image CLIPScore (higher is better)',ylabel='Student watermark detection (%)',title='Seed 0: watermark and utility (weak C10 baseline)')
ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(r/'metrics/core_seed0_frontier.png',dpi=180)
fig.savefig(r/'metrics/core_seed0_frontier.svg'); plt.close(fig)
gate=r/'metrics/c10_baseline_gate.json'
baseline_approved=gate.exists() and json.loads(gate.read_text()).get('pass',False)
decision={'repeat_three_seeds':'requires_baseline_reproduction' if not baseline_approved else 'requires_seed0_effect_and_utility_review',
          'mechanism_and_defense':'conditional on stable attack success across adaptation seeds',
          'baseline_pass_not_created_by_this_run':True}
(r/'metrics/core_followup_decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(result,indent=2),flush=True)

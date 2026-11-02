"""Update only completed author-audited clean controls and qualify legacy contrasts."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
E=R/'report_seed0_evidence'
report=R/'EXPERIMENT_REPORT_SEED0_2026-10-11_EN.md'
threshold=json.loads((E/'C00_seed0_author_audit.json').read_text())['threshold']['threshold']
names=['C00_seed0_author','C01_step1.00_seed0_author']
scores=[np.load(E/f'{n}_scores.npy').reshape(-1) for n in names]
clips=[np.load(E/f'{n}_clipscore.npy').reshape(-1) for n in names]
assert all(len(x)==5000 and np.isfinite(x).all() for x in scores+clips)
positive=[x>=np.float32(threshold) for x in scores]
assert [int(x.sum()) for x in positive]==[110,95]
rng=np.random.default_rng(20261011)
boot=[[],[],[],[]]
for _ in range(1000):
    idx=rng.integers(0,5000,5000)
    boot[0].append(positive[0][idx].mean());boot[1].append(positive[1][idx].mean())
    boot[2].append((positive[1][idx].astype(float)-positive[0][idx]).mean())
    boot[3].append((clips[1][idx]-clips[0][idx]).mean())
ci=[np.quantile(x,[.025,.975]).tolist() for x in boot]
delta=float((clips[1]-clips[0]).mean())
stats={'branch':'author detector, corrected clipping, clean controls only','bootstrap_seed':20261011,
    'bootstrap_resamples':1000,'n':5000,'threshold':threshold,
    'C00_rate':float(positive[0].mean()),'C00_rate_ci95':ci[0],
    'C01_strong_rate':float(positive[1].mean()),'C01_strong_rate_ci95':ci[1],
    'C01_minus_C00_rate':float(positive[1].mean()-positive[0].mean()),'paired_rate_difference_ci95':ci[2],
    'C01_minus_C00_clipscore':delta,'paired_clipscore_difference_ci95':ci[3],
    'relative_clipscore_change_pct':100*delta/float(clips[0].mean())}
(E/'author_clean_control_statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
text=report.read_text(encoding='utf-8')

def replace(old,new):
    global text
    assert text.count(old)==1,old[:100]
    text=text.replace(old,new)

replace('The trainer is now corrected for future runs; all results below remain pre-fix measurements. Its effect on TPR is unmeasured.',
    'The completed C00 and strong C01 rows below now use the corrected trainer and the author-provided SD2.1 detector. C10/C11 and mild C01 remain historical pre-fix measurements. The effect of clipping on radioactivity TPR remains unmeasured.')
replace('The core queue completed at 10:18 ICT (UTC+7).',
    'The original six-condition core queue completed at 10:18 ICT (UTC+7). This update replaces C00 and strong C01 with completed corrected reruns; author-key C10/C11 results are pending.')
replace('CtrlRegen+ preserved image-level watermark detectability:',
    '**Historical watermark-branch findings (unchanged):** CtrlRegen+ preserved image-level watermark detectability:')
replace('## 2. Experimental design and protocol',
    '**Updated clean controls:** C00 and strong C01 have author-detector positive rates of **2.20%** and **1.90%**, respectively. Strong C01 CLIPScore is **0.807053**, compared with **0.811319** for C00 (a **0.53%** decline); FID/KID are lower. These are clean-student false-positive rates, not radioactive TPR. Updated C00/C01 cannot be combined with historical C10/C11 to recompute attack effects because the detector and threshold differ.\n\n## 2. Experimental design and protocol')
replace('Four disjoint prompt splits were used:',
    '**C00/C01 rerun protocol:** The original teacher mirror and author-supplied cache (snapshot `5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06`) have identical UNet, VAE and text-encoder tensors and matching model, scheduler and tokenizer files. Clean raw and strong-regenerated train/validation images were therefore reused. C00 resumed the corrected clean run from step 8,500 to 20,000, including optimizer and RNG state; strong C01 trained fresh for 20,000 steps with corrected clipping. Both retain the original generation and MSE-selection settings. Audit uses the author-provided epoch-50 SD2.1 detector and its separately frozen threshold; all historical watermark rows retain the original local detector.\n\nFour disjoint prompt splits were used:')
replace('The core experiment comprises six students.', 'The original core experiment comprised six students.')
replace('## 4. Pilot and strength selection',
    '**Updated C00/C01 audit branch:** The author detector uses a separate frozen threshold, **tau = 0.06189660727977753**, calibrated to **50/5,000 false positives (1.00%)** on the same held-out clean calibration images. This threshold is shared by both updated clean controls. The author detector and calibration are distinct from the historical results above; a nominal 1% calibration FPR does not imply 1% FPR on the shifted student distribution.\n\n## 4. Pilot and strength selection')
replace('Confidence intervals are 95% bootstrap intervals based on 1,000 resamples.',
    'Confidence intervals are 95% bootstrap intervals based on 1,000 resamples. The C00 and strong C01 intervals were recomputed for their corrected reruns using bootstrap seed 20261011. Other rows and intervals remain historical. **Author-audited clean rows and local-detector historical watermark rows are not a matched 2x2 comparison.**')
replace('| C00 clean/raw | — | 46 | 0.92% | [0.68, 1.22]% | 20,000 | 0.149790 |',
    f'| C00 clean/raw (updated; author audit) | — | 110 | 2.20% | [{100*ci[0][0]:.2f}, {100*ci[0][1]:.2f}]% | 20,000 | 0.149788 |')
replace('| C01 clean/regen | 0.25 | 48 | 0.96% | [0.72, 1.24]% | 20,000 | 0.144479 |',
    '| C01 clean/regen (historical mild) | 0.25 | 48 | 0.96% | [0.72, 1.24]% | 20,000 | 0.144479 |')
replace('| C01 clean/regen | 1.00 | 41 | 0.82% | [0.58, 1.08]% | **19,500** | 0.146157 |',
    f'| C01 clean/regen (updated; author audit) | 1.00 | 95 | 1.90% | [{100*ci[1][0]:.2f}, {100*ci[1][1]:.2f}]% | **19,500** | 0.146157 |')
replace('| C00 | 0.812932 | 13.295 | 1.356 ± 0.207 |',
    '| C00 (updated) | 0.811319 | 13.520 | 1.468 ± 0.203 |')
replace('| C01 mild | 0.810667 | 13.558 | 1.534 ± 0.262 |',
    '| C01 mild (historical) | 0.810667 | 13.558 | 1.534 ± 0.262 |')
replace('| C01 strong | 0.807146 | 13.036 | 1.071 ± 0.179 |',
    '| C01 strong (updated) | 0.807053 | 13.043 | 1.067 ± 0.176 |')
replace('C01 strong declined by **0.005786**, approximately **0.71%**, relative to C00, with CI [−0.007054, −0.004462].',
    f'In the updated clean branch, C01 strong declined by **{abs(delta):.6f}**, approximately **{abs(stats["relative_clipscore_change_pct"]):.2f}%**, relative to updated C00, with paired CI [{ci[3][0]:.6f}, {ci[3][1]:.6f}].')
replace('### 5.3. Paired effects and clean controls',
    '### 5.3. Paired effects and clean controls\n\n**Historical analysis:** All numerical attack contrasts, clean-adjusted effects, score interactions, AUCs and the frontier in this subsection refer to the original six pre-fix students, including original C00 (0.92%) and strong C01 (0.82%). They are retained unchanged and do not incorporate the updated clean-control rows. Author-key C10/C11 are required before computing updated 2x2 effects.\n\nFor the updated clean controls alone, C01 minus C00 is **-0.30 percentage points**, with paired bootstrap CI **['+f'{100*ci[2][0]:.2f}, {100*ci[2][1]:.2f}'+'] percentage points**. This measures a change in false positives and does not demonstrate watermark removal.')
replace('## 6. Answers to the plan\'s research questions',
    '## 6. Answers to the plan\'s research questions\n\nThe watermark-transfer answers below remain interpretations of the historical matched experiment. Updated C00/C01 establish clean-control behavior and utility only; they cannot resolve Q2/Q3 until matched author-key C10/C11 are available.')
replace('## 8. Progress and remaining work',
    '## 8. Progress and remaining work\n\n**Clean-control update:** Corrected C00 and strong C01 have completed training, checkpoint selection, 5,000-image generation, author-detector auditing and CLIPScore/FID/KID evaluation. Mild C01 has not been rerun. The table below describes the original six-condition experiment; the matched author-key C10/C11 update remains pending.')
replace('Core measurements: [summary]',
    'Updated clean-control evidence: [C00 detector audit](report_seed0_evidence/C00_seed0_author_audit.json), [C00 utility](report_seed0_evidence/C00_seed0_author_utility.json), [strong C01 detector audit](report_seed0_evidence/C01_step1.00_seed0_author_audit.json), [strong C01 utility](report_seed0_evidence/C01_step1.00_seed0_author_utility.json), and [paired clean-control statistics](report_seed0_evidence/author_clean_control_statistics.json). Artifact provenance and backbone equivalence: [author checkpoint manifest](report_seed0_evidence/serum_author_manifest.json), [backbone comparison](report_seed0_evidence/backbone_comparison.json), and [C00 resume record](report_seed0_evidence/C00_resume_provenance.json).\n\nHistorical core measurements: [summary]')
assert '| C10 WM/raw | — | 162 | 3.24%' in text
assert '| C11 WM/regen | 1.00 | 127 | 2.54%' in text
assert '| C10 | 0.812940 | 14.601 | 2.353 ± 0.268 |' in text
assert '| C11 strong | 0.811133 | 13.982 | 1.723 ± 0.211 |' in text
report.write_text(text,encoding='utf-8')
print(json.dumps(stats,indent=2))

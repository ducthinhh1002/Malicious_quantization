from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
evidence = root / 'report_seed0_evidence'
audit = json.loads((evidence / 'C10_seed0_author_audit.json').read_text())
assert audit['detections'] == 221 and audit['n'] == 5000
scores = np.load(evidence / 'C10_seed0_author_scores.npy').reshape(-1)
clean = np.load(evidence / 'C00_seed0_author_scores.npy').reshape(-1)
tau = np.float32(audit['threshold']['threshold'])
wm, c = scores >= tau, clean >= tau
rng = np.random.default_rng(20261011)
boot, diff = [], []
for _ in range(1000):
    ix = rng.integers(0, 5000, 5000)
    boot.append(wm[ix].mean())
    diff.append(wm[ix].mean() - c[ix].mean())
ci = np.quantile(boot, [.025, .975]) * 100
dci = np.quantile(diff, [.025, .975]) * 100
stats = dict(n=5000, resamples=1000, bootstrap_seed=20261011,
             C10_rate=float(wm.mean()), C10_rate_ci95=(ci/100).tolist(),
             C10_minus_C00=float(wm.mean()-c.mean()), paired_difference_ci95=(dci/100).tolist())
(evidence/'author_raw_control_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
p = root / 'EXPERIMENT_REPORT_SEED0_2026-10-11_EN.md'
t = p.read_text(encoding='utf-8')
replacements = {
 'The completed C00 and strong C01 rows below': 'The completed C00, C10 and strong C01 rows below',
 'C10/C11 remain historical pre-fix measurements. The effect of clipping on radioactivity TPR remains unmeasured.': 'Only C11 remains pending in the corrected author-artifact branch; its reported numbers are historical pre-fix measurements. The isolated effect of clipping remains unmeasured because the watermark key and detector also changed.',
 'This update replaces C00 and strong C01 with completed corrected reruns; author-key C10/C11 results are pending.': 'This update includes corrected C00, C01 and C10 detector results. Author-key C11 and C10 utility results are pending. Updated through the C10 audit completed at 21:14 ICT.',
 'C10 reached only 3.24%, substantially below': 'The updated C10 reached only **4.42%**, substantially below',
 'Updated C00/C01 cannot be combined with historical C10/C11 to recompute attack effects because the detector and threshold differ.': 'Updated C10 detects **221/5,000 outputs (4.42%)**, versus **110/5,000 (2.20%)** for C00: a **2.22-percentage-point excess**. These updated rows cannot be combined with historical C11 to recompute attack effects because the detector and threshold differ.',
 'all historical watermark rows retain the original local detector.': 'historical watermark rows retain the original local detector. Updated C10 trained fresh on author-watermarked teacher images for 20,000 steps with corrected clipping; minimum unseen MSE selected step 20,000 (MSE 0.1516707777). It generated 5,000 disjoint evaluation outputs under the same student generation settings.',
 'The corrected author-key watermark conditions are still pending.': 'Corrected author-key C10 training, selection, generation and detector audit are complete; C11 and C10 utility are pending.',
 '**Updated C00/C01 audit branch:**': '**Updated C00/C01/C10 audit branch:**',
 'This threshold is shared by both updated clean controls.': 'This threshold is shared by all updated students, including C10.',
 'The C00 and strong C01 intervals were recomputed for their corrected reruns using bootstrap seed 20261011. Other rows and intervals remain historical. **Author-audited clean rows and local-detector historical watermark rows are not a matched 2x2 comparison.**': 'The C00, C01 and C10 intervals were recomputed for their corrected reruns using bootstrap seed 20261011. C11 remains historical. **The updated branch does not yet form a complete matched 2x2 comparison.**',
 '| C10 WM/raw | — | 162 | 3.24% | [2.74, 3.72]% | 20,000 | 0.146096 |': f'| C10 WM/raw (updated; author audit) | — | 221 | 4.42% | [{ci[0]:.2f}, {ci[1]:.2f}]% | 20,000 | 0.151671 |',
 '| C11 WM/regen | 1.00 |': '| C11 WM/regen (historical) | 1.00 |',
 '| C10 | 0.812940 | 14.601 | 2.353 ± 0.268 |': '| C10 (historical) | 0.812940 | 14.601 | 2.353 ± 0.268 |',
 '| C11 strong | 0.811133': '| C11 strong (historical) | 0.811133',
 'Strong regeneration reduced C11 CLIPScore by': '**Historical watermark utility comparison:** Strong regeneration reduced C11 CLIPScore by',
 'Author-key C10/C11 are required before computing updated 2x2 effects.': 'Author-key C11 is required before computing updated 2x2 effects. Updated C10 utility is pending; the historical C10 utility row must not be attributed to the updated model.',
 'The watermark-transfer answers below remain interpretations of the historical matched experiment. Updated C00/C01 establish clean-control behavior and utility only; they cannot resolve Q2/Q3 until matched author-key C10/C11 are available.': 'Q2 now uses the updated matched C00/C10 detector results. Laundering-effect answers refer only to the historical matched experiment; updated C11 is required to resolve Q3 and classify the new branch.',
 'A weak signal is present: C10 reaches 3.24%, compared with C00 at 0.92% and unadapted SD1.4 at 0.96%, giving a 2.32-percentage-point excess over C00.': 'The updated C10 reaches 4.42% versus updated C00 at 2.20%, a 2.22-percentage-point excess. This supports a weak detection signal associated with watermark-trained adaptation.',
 'Cases A/B have not been established. Strong reduces detection without collapse.': 'No Case A/B/C/D can yet be assigned to the updated branch because C11 is pending. Historically, strong reduced detection without observed collapse, a weak trend toward Case A.',
 'Current C10 is lower by **49.06 percentage points**.': 'Updated C10 is lower by **47.88 percentage points**. Author-provided artifacts and corrected clipping have not reproduced the published baseline. The increase from historical 3.24% to updated 4.42% is not an isolated clipping or detector improvement test, because multiple components changed.',
 'A post-hoc threshold derived from C00 gives C10 approximately': 'In the historical branch, a post-hoc threshold derived from C00 gave C10 approximately',
 'The locally trained key/detector differ from the exact artifacts used in the paper\'s radioactivity experiment. Prompt IDs and teacher samples also differ.': 'The historical key/detector were locally trained. The updated branch uses the author-provided epoch-50 key/detector, but identity with every artifact used in the paper\'s radioactivity run is not independently established. Prompt IDs and teacher samples also differ.',
 'The SD2.1 mirror has a pinned revision and recorded hashes, but equality with the paper\'s exact teacher weights has not been established by hash comparison.': 'The mirror matches the author-supplied SD2.1 cache in model tensors and configuration files. Teacher samples and the complete original radioactivity training recipe remain unmatched.',
 'Only one adaptation seed and one watermark key were evaluated.': 'Only one adaptation seed was evaluated; the historical local-key and updated author-key branches are different experimental configurations, not seed replication.',
 '**Clean-control update:** Corrected C00 and strong C01 have completed training, checkpoint selection, 5,000-image generation, author-detector auditing and CLIPScore/FID/KID evaluation. The table below describes the historical raw/strong subset; the matched author-key C10/C11 update remains pending.': '**Current update:** Corrected C00/C01 have completed all listed evaluations. Corrected C10 has completed training, checkpoint selection, 5,000-image generation and author-detector audit; utility is pending. C11 remains in preprocessing. The historical stage record below is retained separately from this incomplete updated branch.',
 'corrected author-key C10/C11 pending': 'corrected C10 completed; corrected C11 pending',
 'corrected C00/C01 completed separately': 'corrected C00/C01/C10 completed separately',
 'Obtain the original radioactivity recipe/artifacts and test remaining key/checkpoint, LoRA, and teacher differences': 'The author checkpoint/cache have now been obtained. Obtain the complete original radioactivity adaptation recipe and test remaining LoRA, dependency and teacher-sampling differences',
 '**Supported statement:** In the local configuration': '**Supported historical statement:** In the local configuration',
}
for old,new in replacements.items():
    assert old in t, old
    t=t.replace(old,new)
needle='Define the TPR reduction as'
t=t.replace(needle, f'**Updated raw baseline:** C10 minus C00 is **2.22 percentage points**, paired bootstrap CI **[{dci[0]:.2f}, {dci[1]:.2f}] percentage points**. This is a watermark-trained versus clean-trained output contrast, not an attack reduction. C11 is pending.\n\n'+needle)
t=t.replace('Updated clean-control evidence:', 'Updated C10 evidence: [detector audit](report_seed0_evidence/C10_seed0_author_audit.json), [matched raw summary](report_seed0_evidence/raw_baseline_summary.json), and [paired raw-control statistics](report_seed0_evidence/author_raw_control_statistics.json).\n\nUpdated clean-control evidence:')
t += '\n**Supported updated statement:** With author-provided artifacts and corrected clipping, C10 detection is 4.42% versus 2.20% for C00. Radioactivity remains weak relative to the cited paper baseline. No updated laundering-success or Pareto claim is supported before C11 and C10 utility are available.\n'
assert 'mild' not in t.lower()
p.write_text(t,encoding='utf-8')
print(json.dumps(stats,indent=2))

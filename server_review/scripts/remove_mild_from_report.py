from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'EXPERIMENT_REPORT_SEED0_2026-10-11_EN.md'
text = path.read_text(encoding='utf-8')
replacements = {
    'C10/C11 and mild C01 remain historical pre-fix measurements.': 'C10/C11 remain historical pre-fix measurements.',
    'This update replaces C00 and strong C01 with completed corrected reruns;': 'The reported comparison is restricted to raw and strong regeneration. This update replaces C00 and strong C01 with completed corrected reruns;',
    ' Mild regeneration showed no clear effect.': '',
    'historical six-student experiment': 'historical raw/strong experiment',
    'The original core experiment comprised six students. Raw C00 and C10 are shared across strength comparisons; C01 and C11 are trained separately at mild and strong settings. Each student produced 5,000 evaluation images, yielding **30,000 student outputs**. Regeneration of the full clean/watermarked training and validation sets at both selected strengths yielded **24,000 images**.': 'This report includes four conditions: raw C00/C10 and strong C01/C11 (strength 1.00). The historical runs for these four conditions produced **20,000 student outputs**, 5,000 per student; strong regeneration produced **12,000 training/validation images**, 6,000 per teacher branch. Lower-strength conditions from the original broader experiment are excluded from this report. The corrected author-key watermark conditions are still pending.',
    'The pilot included 500 clean and 500 watermarked images, each processed at four strengths. Similarity values below are means over the **500 watermarked images**; clean controls were evaluated in parallel.': 'The historical pilot included 500 clean and 500 watermarked images and evaluated four strengths. Only raw and the selected strong setting are reported below. Similarity values are means over the **500 watermarked images**; clean controls were evaluated in parallel.',
    '**Mild = 0.25** maximized `CLIP + DINO − LPIPS` among acceptable lower-strength candidates. **Strong = 1.00**': '**Strong = 1.00**',
    'All numerical attack contrasts, clean-adjusted effects, score interactions, AUCs and the frontier in this subsection refer to the original six pre-fix students,': 'All numerical attack contrasts, clean-adjusted effects, score interactions and AUCs in this subsection refer to the historical matched raw/strong students,',
    '2.32 percentage points for raw, 2.24 for mild, and 1.72 for strong.': '2.32 percentage points for raw and 1.72 for strong.',
    'For mild, Δ = **−0.0000418**, CI [−0.002093, 0.001946]. For strong,': 'For strong,',
    'The frontier contains three operating points: raw, mild, and strong. It is not a continuous downstream strength sweep.': 'The reported comparison contains two operating points: raw and strong. It does not establish a continuous downstream strength–utility frontier.',
    'Mild shows no clear evidence. Strong reduces': 'Strong reduces',
    'Mild has no demonstrated effect in this seed; strong reduces': 'Strong reduces',
    'Case D cannot be generalized across strengths/seeds.': 'Case D cannot be generalized across operator settings or seeds.',
    ' Mild C01 has not been rerun.': '',
    'The table below describes the original six-condition experiment;': 'The table below describes the historical raw/strong subset;',
    '| 1–2: pilot and mild/strong selection | Completed using preservation metrics |': '| 1–2: pilot and strong selection | Completed using preservation metrics; lower-strength conditions excluded from this report |',
    '| 3: 2×2 datasets | Completed at mild/strong; automated pairing audits completed |': '| 3: 2×2 datasets | Historical raw/strong datasets completed; automated pairing audits completed |',
    '| 4–5: LoRA training and checkpoint selection | Completed for six students, seed 0 |': '| 4–5: LoRA training and checkpoint selection | Historical four reported students completed, seed 0; corrected author-key C10/C11 pending |',
    '| 6–7: 5k outputs per student and frozen-threshold audit | Completed; 30,000 images |': '| 6–7: 5k outputs per student and frozen-threshold audit | Historical reported subset completed: 20,000 images; corrected C00/C01 completed separately |',
    'repeat raw and mild/strong conditions': 'repeat raw and strong conditions',
    'only raw/mild/strong strength comparisons have been completed.': 'only the raw/strong comparison is included here. A complete strength sweep is not established by this report.',
}
for old, new in replacements.items():
    assert old in text, old
    text = text.replace(old, new)
text = '\n'.join(line for line in text.splitlines() if not (
    line.startswith('| 0.25 |') or line.startswith('| 0.50 |') or line.startswith('| 0.75 |')
    or 'historical mild)' in line or line.startswith('| C11 WM/regen | 0.25 |')
    or line.startswith('| C01 mild ') or line.startswith('| C11 mild |')
    or line.startswith('| Mild |')
    or line.startswith('![C10 and C11 watermark–utility frontier')
)) + '\n'
assert 'mild' not in text.lower()
for value in ['110 | 2.20%', '95 | 1.90%', '162 | 3.24%', '127 | 2.54%', '0.811319', '0.807053']:
    assert value in text, value
path.write_text(text, encoding='utf-8')
print('Updated report: removed lower-strength results and obsolete frontier figure; retained raw/strong results and branch qualifications.')

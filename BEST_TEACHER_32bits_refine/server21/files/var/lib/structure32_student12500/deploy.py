"""Run once remotely to wire immutable reused assets and original source modules."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent;BASE=Path('/var/lib/vllm')
for name in ['harness','runtime']:
    if not (ROOT/name).exists():(ROOT/name).symlink_to(BASE/name,target_is_directory=True)
for name in ['k1_original.py']:
    if not (ROOT/name).exists():shutil.copy2(BASE/name,ROOT/name)
folder=ROOT/'student12500';folder.mkdir(exist_ok=True)
for p in (BASE/'student12500').glob('*.py'):
    text=p.read_text()
    if p.name=='campaign.py':
        text=text.replace("BASE/'runs/nash_structure'","BASE/'runs/structure32'")
        text=text.replace('nash_structure_student12500_seed0','structure32_student12500_seed0')
        text=text.replace('full_structure_nash_student12500','full_structure32_student12500')
        start=text.index('def summary():');end=text.index('\ndef supervise():',start)
        text=text[:start]+"def summary():\n    from key_summary import summary as key_report\n    key_report(RUN,BRANCH)\n\n"+text[end:]
        text=text.replace("'campaign.py','tests.py']","'campaign.py','tests.py','key_metrics.py','key_summary.py','key_select.py']")
        text=text.replace("stage('05_select',HARNESS/'validation.py'","stage('05_select',ROOT/'key_select.py'")
        text=text.replace("'goal_tpr':.90,'goal_comparison':'> .90, not a guaranteed outcome',","'primary_metrics':['bit_accuracy','full_key_recovery_rate'],")
        text=text.replace("'LR decay may not reach90%'","'32bit exploratory single-seed experiment'")
        text=text.replace('training_gpu_cap_gib\':44','training_gpu_cap_gib\':44')
    (folder/p.name).write_text(text)
for name in ['key_metrics.py','key_summary.py','key_select.py']:shutil.copy2(ROOT/name,folder/name)
test=(BASE/'student_tests.py').read_text().replace('nash_structure_student12500_seed0','structure32_student12500_seed0')
(ROOT/'student_tests.py').write_text(test)
# Top-level bootstrap is for student_tests.py; child campaign/driver use their own.
(ROOT/'bootstrap.py').write_text("from runtime import ROOT,HARNESS,read_json,write_json,freeze,sha256\n")
from fixed_endpoint import apply_policy
apply_policy(ROOT)
print('DEPLOYED',ROOT)

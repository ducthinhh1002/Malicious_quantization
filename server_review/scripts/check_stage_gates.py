import argparse,json
from pathlib import Path
ROOT=Path('/root/workspace/serum')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--strength',type=float); a=p.parse_args()
    integrity=json.loads((ROOT/'manifests/teacher_integrity_v2.json').read_text())
    assert integrity['ok'], 'Data audit failed'
    cal=json.loads((ROOT/'metrics/serum_threshold_1pct.json').read_text())
    assert cal['model']==json.loads((ROOT/'manifests/serum_frozen.json').read_text())
    # Created only after selecting C10 by validation MSE and scoring its eval outputs.
    baseline=json.loads((ROOT/'metrics/c10_baseline_gate.json').read_text())
    assert baseline['pass'] and baseline['threshold']==cal['threshold'], 'C10 baseline not credible'
    assert baseline['selection_criterion']=='minimum unseen denoising MSE'
    if a.strength is not None:
        selection=json.loads((ROOT/'metrics/pilot_selection.json').read_text())
        assert selection['criterion']=='image preservation only'
        assert a.strength in (selection['mild'],selection['strong']), 'Strength not selected by pilot'
        for step in (0.25,0.5,0.75,1.0):
            for kind in ('clean','wm'):
                assert (ROOT/'metrics/pilot'/f'{kind}_{step:.2f}.json').exists(), 'Missing pilot metrics'
    print('Stage gates passed')
if __name__=='__main__': main()

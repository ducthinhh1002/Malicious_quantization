"""Record a reviewed baseline/pilot decision with evidence; never auto-select by detector."""
import argparse,json,math
from pathlib import Path
ROOT=Path('/root/workspace/serum')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline-student',type=Path)
    p.add_argument('--baseline-scores',type=Path); p.add_argument('--mild',type=float)
    p.add_argument('--strong',type=float); p.add_argument('--reason',required=True)
    a=p.parse_args()
    if a.baseline_student:
        assert a.baseline_scores is not None
        selection=json.loads((a.baseline_student/'validation_selection.json').read_text())
        result=json.loads(Path(str(a.baseline_scores)+'.result.json').read_text())
        assert result['num_images']==5000
        n=result['num_images']; rate=result['fraction_above']; z=1.96
        lower=(rate+z*z/(2*n)-z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n)))/(1+z*z/n)
        assert lower>0.01, '95% confidence interval reaches null; investigate baseline first'
        record={'pass':True,'threshold':result['threshold'],'tpr':result['fraction_above'],
                'selection_criterion':selection['criterion'],'student':str(a.baseline_student),
                'scores':str(a.baseline_scores),'review_reason':a.reason,'wilson_95_lower':lower}
        out=ROOT/'metrics/c10_baseline_gate.json'
    else:
        assert a.mild in (.25,.5,.75,1.) and a.strong in (.25,.5,.75,1.) and a.mild<a.strong
        for k in ('clean','wm'):
            for s in (.25,.5,.75,1.):
                assert (ROOT/'metrics/pilot'/f'{k}_{s:.2f}.json').exists()
        record={'mild':a.mild,'strong':a.strong,'criterion':'image preservation only','reason':a.reason}
        out=ROOT/'metrics/pilot_selection.json'
    out.write_text(json.dumps(record,indent=2)); print(out)
if __name__=='__main__':main()

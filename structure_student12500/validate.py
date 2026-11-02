"""Validate a newly initialized reproduction and issue its test receipt."""
import subprocess
import sys
import bootstrap
from bootstrap import ROOT,sha256,write_json

def main():
    import campaign
    from verify_bundle import main as check_bundle
    check_bundle()
    manifest=campaign.QUEUE/'code_manifest.json'
    if not manifest.exists():raise RuntimeError('Configure source artifacts and run campaign.py --initialize first')
    campaign.verify()
    log=campaign.QUEUE/'packaged_code_tests.log'
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','unittest','tests'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Tests failed; see '+str(log))
    write_json(campaign.QUEUE/'tests.json',{'passed':True,'manifest_sha256':sha256(manifest),
        'log_sha256':sha256(log),'scope':'packaged code resume tests; see log for CUDA availability/skips'})
    print('PASS; campaign launch receipt written:',campaign.QUEUE/'tests.json')

if __name__=='__main__':main()

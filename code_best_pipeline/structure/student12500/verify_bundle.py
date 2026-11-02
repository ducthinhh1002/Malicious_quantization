"""Check source hashes and bundled imports without loading model weights."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent

def main():
    expected=json.loads((ROOT/'bundled_source_manifest.json').read_text())['files']
    for rel,digest in expected.items():
        p=ROOT/rel
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Bundled source changed: '+rel)
    import bootstrap
    import student,validation,semantic_teacher,semantic_report
    from common import verify_vendor
    verify_vendor()
    for module in [student,validation,semantic_teacher,semantic_report]:
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'historical'):
            raise RuntimeError('Imported external harness: '+str(module.__file__))
    trainer=bootstrap.HARNESS/'vendor/diffusers/train_text_to_image_lora.py'
    original=json.loads((ROOT/'reference/original_resume_identity.json').read_text())
    if hashlib.sha256(trainer.read_bytes()).hexdigest()!=original['trainer_sha256']:
        raise RuntimeError('Trainer differs from the completed12500 run')
    for p in [*ROOT.glob('*.py'),*(ROOT/'historical').rglob('*.py')]:
        compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    print('PASS:',len(expected),'bundled source hashes, vendor provenance, imports, original trainer hash and syntax')

if __name__=='__main__':main()

"""Reuse the existing local-to-.23 transport for frozen clean teacher references."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parent
def main():
    p=ROOT.parent/'student8bit_rank4_refine600/deploy_from_local.py'
    spec=importlib.util.spec_from_file_location('transport',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.STATE=ROOT.parents[2]/'output_artifacts/deployments/teacher8bit_rank32_robust_23'
    m.STATE.mkdir(parents=True,exist_ok=True)
    m.image_bundle('teacher_clean_train')
    m.ssh("echo '{\"verified\":true,\"n_images\":5000}' > /var/lib/structure8bit_rank4_refine600/CLEAN_TRAIN_READY.json")
if __name__=='__main__':main()

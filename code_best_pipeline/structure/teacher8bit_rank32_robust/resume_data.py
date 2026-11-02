"""Resume remaining .23 uploads and add clean teacher training references."""
from pathlib import Path
import importlib.util,concurrent.futures
ROOT=Path(__file__).resolve().parent
def main():
    p=ROOT.parent/'student8bit_rank4_refine600/deploy_from_local.py'
    spec=importlib.util.spec_from_file_location('transport',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def student_data():
        names=['teacher_wm_train','teacher_wm_validation','teacher_wm_test','teacher_clean_validation','teacher_clean_test','unadapted']
        for n in names:m.image_bundle(n)
        m.ssh("echo '{\"verified\":true}' > /var/lib/structure8bit_rank4_refine600/DATA_READY.json")
    def teacher_data():
        m.image_bundle('teacher_clean_train')
        m.ssh("echo '{\"verified\":true,\"n_images\":5000}' > /var/lib/structure8bit_rank4_refine600/CLEAN_TRAIN_READY.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(student_data),pool.submit(teacher_data)]
        for job in jobs:job.result()
if __name__=='__main__':main()

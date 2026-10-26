"""Run the frozen trainer unchanged, auditing its replay at update7500."""
import argparse
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trainer',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    args,trainer_args=parser.parse_known_args()
    import torch
    from safetensors.torch import load_file
    from common import sha256,write_json
    namespace=runpy.run_path(str(args.trainer),run_name='student10000_frozen_trainer')
    globals_=namespace['main'].__globals__
    original=globals_['save_native_adapter']
    def save_and_audit(model,directory):
        result=original(model,directory)
        if Path(directory).name=='snapshot-7500':
            actual=Path(directory)/'adapter_model.safetensors'
            a=load_file(str(actual));b=load_file(str(args.reference))
            if a.keys()!=b.keys():raise RuntimeError('Replay adapter keys differ')
            if any(not torch.isfinite(x).all() for x in a.values()):raise RuntimeError('Nonfinite replay weights')
            equal=all(torch.equal(a[k],b[k]) for k in a)
            max_diff=max(float((a[k]-b[k]).abs().max()) for k in a)
            write_json(args.proof,{'step':7500,'tensor_exact_equal':equal,'max_abs_difference':max_diff,
                'actual_sha256':sha256(actual),'reference_sha256':sha256(args.reference),
                'optimizer_continuous_from_clean_initialization':True,
                'original_optimizer_state_available':False,
                'interpretation':'Exact weight replay verified' if equal else 'Same-config seeded replay has numerical divergence; compare as a rerun, not exact original continuation'})
            print(f'REPLAY_7500 exact_equal={equal} max_abs_difference={max_diff}',flush=True)
        return result
    globals_['save_native_adapter']=save_and_audit
    sys.argv=[str(args.trainer),*trainer_args]
    namespace['main']()


if __name__=='__main__':main()

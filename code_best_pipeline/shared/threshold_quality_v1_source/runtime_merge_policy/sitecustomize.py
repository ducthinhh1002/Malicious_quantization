"""Runtime fix for fresh merge subprocesses; frozen experiment files stay intact.

Only the two campaign data operations need this bootstrap. Their merge helper
otherwise runs before semantic_teacher.load_native applies numerical_policy.
Training and generation already set this same full-FP32 backend policy.
"""
from pathlib import Path
import sys


def applies(argv):
    return (bool(argv) and Path(argv[0]).name in ('transfer_campaign.py', 'minhnqn_campaign.py')
            and '--operation' in argv
            and argv.index('--operation') + 1 < len(argv)
            and argv[argv.index('--operation') + 1] == 'data')


if applies(sys.argv):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('MERGE_NUMERICAL_POLICY: deterministic FP32; matmul/cudnn TF32 disabled',
          file=sys.stderr, flush=True)

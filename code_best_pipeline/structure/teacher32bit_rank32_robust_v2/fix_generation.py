"""Resume postprocessing with the same strict numerical policy as training."""
import bootstrap
import semantic_teacher_train
from semantic_teacher import numerical_policy
original_merge = semantic_teacher_train.merge_teacher
def strict_merge(*args, **kwargs):
    numerical_policy()
    return original_merge(*args, **kwargs)
semantic_teacher_train.merge_teacher = strict_merge
import run_teacher
original_stage = run_teacher.stage
def recovered_stage(name, args):
    args = [str(Path(__file__).resolve()) if a == str(run_teacher.ROOT/'run_teacher.py') else a for a in args]
    return original_stage(name, args)
from pathlib import Path
run_teacher.stage = recovered_stage
if __name__ == '__main__':
    try:
        run_teacher.main()
    except Exception as exc:
        run_teacher.status('failed', error=repr(exc))
        raise

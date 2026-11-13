"""Label only this benchmark and its children; do not rename unrelated jobs."""
import os
from pathlib import Path

TITLE='VLLM::Enginecore'
def configure():
    from setproctitle import setproctitle
    setproctitle(TITLE)
    os.environ['BASELINE_PROCESS_TITLE']=TITLE
    hooks=str(Path(__file__).resolve().parents[1]/'runtime_hooks')
    paths=os.environ.get('PYTHONPATH','').split(os.pathsep)
    if hooks not in paths:os.environ['PYTHONPATH']=os.pathsep.join([hooks,*filter(None,paths)])

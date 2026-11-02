"""Start the reviewed campaign on VM4; refuse duplicate or overwritten runs."""
import datetime
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
if (root/'run').exists() or (root/'launcher.json').exists():
    raise SystemExit('Existing campaign: inspect logs and use exact --resume; refusing duplicate launch')
with (root/'launcher.log').open('x') as stream:
    child=subprocess.Popen(['bash',str(root/'launch_vm4.sh')],cwd=root,stdin=subprocess.DEVNULL,
                           stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'run':str(root/'run'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'launcher.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))

from pathlib import Path
import os,json
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
pin=json.loads((ROOT/'inputs/server_protocol.json').read_text())
snapshot_download('deepseek-ai/Janus-Pro-1B',revision=pin['model_revision'],local_dir=str(ROOT/'model'))
code=Path(os.environ.get('JPEG_CODE_DIR',str(ROOT.parent/'jpeg'))).expanduser().resolve()
source=Path(os.environ.get('JPEG_TEACHER_RUN',str(code/'run'))).expanduser().resolve()
for name in ['teacher_fp/refined/checkpoint-1200/adapter_model.safetensors','carrier/complete.json','prompts.json','images/teacher_jpeg600_test/image_hashes.json']:
    if not (source/name).exists():raise FileNotFoundError(source/name)
protocol=dict(pin,teacher_run=str(source),teacher_sha256=__import__('hashlib').sha256((source/'teacher_fp/refined/checkpoint-1200/adapter_model.safetensors').read_bytes()).hexdigest(),reproduction='new training can produce different adapter hash; reader must come from SAME teacher run')
protocol['prompt_sha256']=__import__('hashlib').sha256(Path(os.environ.get('JANUS_PROMPTS',str(ROOT/'inputs/prompts.json'))).read_bytes()).hexdigest()
protocol['carrier_sha256']=__import__('hashlib').sha256((source/'carrier/complete.json').read_bytes()).hexdigest()
protocol['portable_job_sha256']=__import__('hashlib').sha256((ROOT/'job.py').read_bytes()).hexdigest()
p=ROOT/'protocol.json'
if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Run protocol changed; choose a fresh work directory')
p.write_text(json.dumps(protocol,indent=2))

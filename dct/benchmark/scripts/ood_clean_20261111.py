from pathlib import Path
import os,sys,json,hashlib,subprocess
PROJECT=Path(__file__).resolve().parents[2]
WORK=PROJECT/"run"

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def train(method):
    from benchmark.shared.student_training.integration import sd14
    from benchmark.scripts.additional_clean import verify_checkpoint
    assets = read(PROJECT/'assets.json')
    job = WORK/method/'student'
    data = WORK/method/'images/teacher_wm_train'
    v = sd14(job,assets['student_path'],data,4)
    v['config'].update(max_train_steps=12500,checkpoint_steps=[])
    link = job/'images/teacher_wm_train'
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():
        link.symlink_to(data,target_is_directory=True)
    assert link.resolve()==data.resolve()
    branch=job/'students/sd14_r4'
    identity=read(v['identity'])
    cmd=v['builder'](v['config'],v['assets'],job,v['branch'])
    cmd[cmd.index('--checkpointing_steps')+1]='1250'
    assert cmd[cmd.index('--max_train_steps')+1]=='12500'
    assert cmd[cmd.index('--rank')+1]=='4'
    checkpoints=sorted(branch.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    if checkpoints:
        verify_checkpoint(checkpoints[-1],identity)
        if int(checkpoints[-1].name.split('-')[-1])<12500:
            cmd+=['--resume_from_checkpoint',str(checkpoints[-1])]
    if not checkpoints or int(checkpoints[-1].name.split('-')[-1])<12500:
        write(job/'training_command.json',dict(command=cmd,prompt_sha256=sha(WORK/'prompts.json')))
        env=os.environ.copy()
        for key in ['BASELINE_TRAIN_ATTACK','BASELINE_PIXEL_INDEX']:
            env.pop(key,None)
        subprocess.run([sys.executable,str(v['helper']/'driver.py'),'--trainer',cmd[1],
                        '--identity-file',str(v['identity']),*cmd[2:]],cwd=PROJECT,env=env,check=True)
    verify_checkpoint(branch/'checkpoint-12500',identity)
    write(job/'trained.json',dict(steps=12500,rank=4))

"""One-command BitMark→LoRA experiment, with matched full-finetuning controls."""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import HERE, freeze, read_json, sha256, verify_vendor, write_json, runtime_root, require_local_runtime_path
from student import branches, training_command


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranks", type=int, nargs="+", default=[4])
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--calibration-n", type=int, default=5000)
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--precision", choices=["fp16", "bf16", "no"], default="fp16")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--inference-batch", type=int, default=4)
    p.add_argument("--fpr", type=float, default=.01)
    p.add_argument("--lora-only", action="store_true", help="Omit matched full controls; inference about LoRA weakness remains limited")
    p.add_argument("--paper-full-ema", action="store_true", help="Also run the paper-style full+EMA positive control")
    p.add_argument("--train-clean-controls", action="store_true", help="Also train matched clean LoRA/full controls to isolate generic adaptation effects")
    p.add_argument("--skip-utility", action="store_true", help="Explicitly omit CLIP/FID/KID (reported as unavailable)")
    p.add_argument("--cache", type=Path, default=runtime_root() / "cache")
    p.add_argument("--output", type=Path, help="Create a NEW run directory here")
    p.add_argument("--resume", type=Path, help="Resume exact recorded config from an existing run; do not supply experiment overrides")
    p.add_argument("--min-free-gb", type=float, default=70)
    p.add_argument("--plan-only", action="store_true", help="Print protocol only; default without this flag executes the experiment")
    p.add_argument("--analyze", type=Path, help="Recompute reports from an existing complete run; no GPU required")
    a = p.parse_args(argv)
    if a.resume and a.output:
        p.error("--resume and --output are mutually exclusive")
    numeric = [a.train_n, a.calibration_n, a.test_n, a.epochs, a.batch_size,
               a.inference_batch, a.inference_steps, *a.ranks]
    if min(numeric) <= 0 or len(set(a.ranks)) != len(a.ranks):
        p.error("Counts/ranks must be positive and ranks unique")
    if (not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or not 0 < a.fpr < 1 or a.seed < 0
            or not math.isfinite(a.min_free_gb) or a.min_free_gb < 0):
        p.error("Invalid learning rate/FPR/seed/disk limit")
    if a.calibration_n * a.fpr < 1:
        p.error("Use at least 1/FPR clean calibration samples")
    if a.test_n < 2:
        p.error("Use at least two test images")
    return a


def config_from_args(a):
    ignored = {"output", "resume", "plan_only", "analyze", "min_free_gb"}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ignored}
    return config


def stage(run, label, command):
    log = run / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print(f"\n[{label}] log: {log}\n{' '.join(map(str, command))}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    with log.open("a") as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env, cwd=HERE, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
    status = {"stage": label, "exit_code": code, "seconds": time.monotonic()-start,
              "finished": datetime.datetime.now().astimezone().isoformat()}
    with (run / "stages.jsonl").open("a") as f:
        f.write(json.dumps(status) + "\n")
    if code:
        raise RuntimeError(f"Stage {label} failed (exit={code}). See {log}; resume with --resume {run}")


def preflight(a):
    for path in [a.cache, a.resume or a.output or runtime_root() / 'runs']:
        require_local_runtime_path(path)
    import torch
    import flash_attn
    if not torch.cuda.is_available():
        raise RuntimeError("BitMark official Infinity needs CUDA. Run this experiment on the GPU server.")
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("Official Infinity/FlashAttention requires an Ampere-or-newer GPU")
    # Exercise actual CUDA kernels before model downloads or long generation.
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    result = flash_attn.flash_attn_func(q, q, q)
    if not torch.isfinite(result).all():
        raise RuntimeError("FlashAttention CUDA smoke check returned nonfinite values")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name()}; torch={torch.__version__}", flush=True)
    for path in [a.cache, (a.output or runtime_root() / "runs").parent]:
        probe = path.resolve()
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free / 1024**3
        if free < a.min_free_gb:
            raise RuntimeError(f"Only {free:.1f} GiB free on {probe}; requested minimum {a.min_free_gb:g}. "
                               "Choose --cache/--output on a larger disk or set an explicit --min-free-gb.")
    return {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "cuda_runtime": torch.version.cuda, "python": sys.version,
            "note": "Matching seeds does not promise bitwise equality across GPU/software versions."}


def main(argv=None):
    a = parse_args(argv)
    if a.analyze:
        from report import summarize
        summarize(a.analyze.resolve())
        return
    verify_vendor()
    c = config_from_args(a)
    if a.resume:
        # Reject hidden overrides rather than silently ignoring user arguments.
        supplied = argv if argv is not None else sys.argv[1:]
        allowed = {"--resume", "--min-free-gb", "--plan-only"}
        if any(x.split("=")[0] not in allowed for x in supplied if x.startswith("--")):
            raise ValueError("--resume loads the frozen config; only --min-free-gb/--plan-only may accompany it")
        c = read_json(a.resume.resolve() / "config.json")
        a.cache = Path(c["cache"])
        a.output = a.resume.resolve()
    if a.plan_only:
        print(json.dumps({"config": c, "students": branches(c),
              "baseline": "unadapted SD2.1", "teacher": "official Infinity-2B / BitMark delta=2, all scales, G=01,10",
              "training": "official Diffusers trainers; fixed final epoch; clean-trained controls optional",
              "expected_updates_per_student": ((c['train_n']+c['batch_size']-1)//c['batch_size'])*c['epochs'],
              "detection": "official tokenizer+detector, frozen clean threshold; |z| primary + signed-z secondary",
              "execute": "bash run_bitmark_lora.sh (omit --plan-only)"}, indent=2))
        return
    hardware = preflight(a)
    if a.resume:
        run = a.resume.resolve()
    else:
        run = a.output.resolve() if a.output else runtime_root() / "runs" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True, exist_ok=False)
    with (run / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(run / "config.json", c)
        source_files = {str(p.relative_to(HERE)): sha256(p) for p in HERE.glob("*.py")}
        source_files["vendor/provenance.json"] = sha256(HERE / "vendor/provenance.json")
        source_files["requirements.txt"] = sha256(HERE / "requirements.txt")
        freeze(run / "code_manifest.json", source_files)
        versions = {p: importlib.metadata.version(p) for p in (
            "torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "flash-attn",
            "datasets", "pyarrow", "numpy", "scipy", "torch-fidelity", "huggingface-hub")}
        freeze(run / "environment.json", {"packages": versions, "hardware": hardware})
        print(f"Run: {run}\nDefault mode EXECUTES training and evaluation, not plan-only.", flush=True)
        py = sys.executable
        def call(file, label, *args):
            stage(run, label, [py, str(HERE / file), "--run", str(run), *args])
        call("assets.py", "01_assets")
        call("teacher.py", "02_teacher", "--stage", "generate")
        call("teacher.py", "03_teacher_scores", "--stage", "score", "--directories",
             "teacher_clean_calibration", "teacher_clean_test", "teacher_wm_test")
        call("report.py", "04_freeze_threshold", "--calibrate-only")
        call("student.py", "05_unadapted", "--stage", "generate", "--branch", "unadapted")
        for b in branches(c):
            call("student.py", f"train_{b['name']}", "--stage", "train", "--branch", b["name"])
            call("student.py", f"generate_{b['name']}", "--stage", "generate", "--branch", b["name"])
        call("teacher.py", "06_student_scores", "--stage", "score", "--directories",
             "unadapted", *[b["name"] for b in branches(c)])
        # Always publish detector results before optional heavyweight utility evaluation.
        call("report.py", "07_radioactivity")
        if not c["skip_utility"]:
            call("utility.py", "08_utility")
            call("report.py", "09_final_report")
        write_json(run / "complete.json", {"status": "completed", "result": str(run / "result")})
        print(f"\nFinished. Review: {run / 'result/README.md'}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    main()

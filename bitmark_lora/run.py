"""One-command BitMark→LoRA experiment, with matched full-finetuning controls."""
import argparse
import contextlib
import traceback
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
from zoneinfo import ZoneInfo
from pathlib import Path

from common import HERE, freeze, read_json, sha256, verify_vendor, write_json, runtime_root, require_local_runtime_path, results_root
from student import branches, training_command


def parse_args(argv=None, method="bitmark"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--calibration-n", type=int, default=5000)
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--epochs", type=int, help="Legacy budget override; cannot accompany --max-train-steps")
    p.add_argument("--max-train-steps", type=int, default=None, help="Shared optimizer budget; default 7500")
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[2500, 5000, 7500])
    p.add_argument("--adapter-scopes", nargs="+", choices=["attention", "attention_conv"],
                   default=["attention", "attention_conv"])
    p.add_argument("--validation-n", type=int, default=256)
    p.add_argument("--validation-generation-n", type=int, default=128)
    p.add_argument("--validation-draws", type=int, default=2, help="Fixed noise/timestep draws per validation image")
    p.add_argument("--watermark-delta", type=float, default=2., help="Official BitMark embedding strength; separate-run ablation")
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
    p.set_defaults(watermark_method=method)
    if method == "serum":
        p.description = "SERUM SD2.1 diffusion teacher -> SD1.4 LoRA/full adaptation"
        p.set_defaults(train_n=5000, validation_n=1000, batch_size=8, learning_rate=1e-5)
        p.add_argument("--serum-alpha", type=float, default=.5)
        p.add_argument("--serum-checkpoint", type=Path, help="Optional local copy of pinned author epoch-50 checkpoint")
        p.add_argument("--serum-cache-archive", type=Path, help="Optional local copy of pinned author SD2.1 cache ZIP")
        p.add_argument("--teacher-batch", type=int, default=4)
        p.add_argument("--score-batch", type=int, default=16)
        p.add_argument("--full-learning-rate", type=float, default=5e-6)
    a = p.parse_args(argv)
    if method == "serum":
        if not math.isfinite(a.serum_alpha) or not 0 < a.serum_alpha < 1:
            p.error("SERUM alpha must lie strictly between 0 and 1")
        if min(a.teacher_batch, a.score_batch) < 1 or not math.isfinite(a.full_learning_rate) or a.full_learning_rate <= 0:
            p.error("Invalid SERUM batch size/full learning rate")
    if a.epochs is not None and a.max_train_steps is not None:
        p.error("Choose --epochs or --max-train-steps, not both")
    if a.epochs is not None and a.epochs <= 0:
        p.error("--epochs must be positive")
    if a.batch_size <= 0:
        p.error("--batch-size must be positive")
    if a.max_train_steps is None:
        a.max_train_steps = (math.ceil(a.train_n / a.batch_size) * a.epochs if a.epochs else 7500)
    if any(x <= 0 for x in a.checkpoint_steps):
        p.error("Checkpoint steps must be positive")
    a.checkpoint_steps = sorted(set(x for x in a.checkpoint_steps if x <= a.max_train_steps) | {a.max_train_steps})
    if a.validation_generation_n > a.validation_n or a.validation_generation_n < 2:
        p.error("Require 2 <= validation-generation-n <= validation-n")
    if len(set(a.adapter_scopes)) != len(a.adapter_scopes):
        p.error("Adapter scopes must be unique")
    if a.resume and a.output:
        p.error("--resume and --output are mutually exclusive")
    numeric = [a.train_n, a.calibration_n, a.test_n, a.max_train_steps, a.validation_n, a.validation_draws, a.batch_size,
               a.inference_batch, a.inference_steps, *a.ranks]
    if min(numeric) <= 0 or len(set(a.ranks)) != len(a.ranks):
        p.error("Counts/ranks must be positive and ranks unique")
    if (not math.isfinite(a.watermark_delta) or a.watermark_delta <= 0 or not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or not 0 < a.fpr < 1 or a.seed < 0
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
    require_local_runtime_path(a.cache)
    if getattr(a, "watermark_method", "bitmark") == "serum":
        from serum_teacher import check_imports as check_model_imports
    else:
        from check_runtime import check_model_imports
    try:
        check_model_imports()
    except Exception as exc:
        raise RuntimeError(
            f"Existing Python environment cannot import pipeline models: {sys.executable}. "
            "See the chained import error above for the missing/incompatible dependency. "
            "Select a working existing interpreter with SERUM_PYTHON/ BITMARK_PYTHON; no automatic installation."
        ) from exc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    major, _ = torch.cuda.get_device_capability()
    if getattr(a, "watermark_method", "bitmark") == "bitmark":
        import flash_attn
        if major < 8:
            raise RuntimeError("Official Infinity/FlashAttention requires an Ampere-or-newer GPU")
        q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        result = flash_attn.flash_attn_func(q, q, q)
        if not torch.isfinite(result).all():
            raise RuntimeError("FlashAttention CUDA smoke check returned nonfinite values")
    else:
        x = torch.randn(16,16,device="cuda")
        if not torch.isfinite(x @ x).all():
            raise RuntimeError("CUDA matrix smoke check returned nonfinite values")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name()}; torch={torch.__version__}", flush=True)
    for path in [a.cache, (a.output or results_root())]:
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


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, text):
        self.log.write(text)
        return self.terminal.write(text)
    def flush(self):
        self.log.flush()
        self.terminal.flush()
    def isatty(self):
        return False


@contextlib.contextmanager
def persistent_log(run):
    with (run / "run.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=log)
                raise


def main(argv=None, method="bitmark"):
    a = parse_args(argv, method)
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
        a.watermark_method = c.get("watermark_method", "bitmark")
        a.cache = Path(c["cache"])
        a.output = a.resume.resolve()
    carrier = c.get("watermark_method", "bitmark")
    teacher_file = "serum_teacher.py" if carrier == "serum" else "teacher.py"
    assets_file = "serum_assets.py" if carrier == "serum" else "assets.py"
    if a.plan_only:
        print(json.dumps({"config": c, "students": branches(c),
              "baseline": "unadapted SD1.4" if carrier == "serum" else "unadapted SD2.1",
              "teacher": (f"author SD2.1 / SERUM alpha={c['serum_alpha']}" if carrier == "serum" else
                          f"official Infinity-2B / BitMark delta={c.get('watermark_delta',2)}, all scales, G=01,10"),
              "training": "fixed step budget; minimum held-out denoising MSE, earlier step breaks ties; detector/utility excluded",
              "results_root": str(results_root()),
              "expected_updates_per_student": c.get("max_train_steps") or math.ceil(c["train_n"]/c["batch_size"])*c["epochs"],
              "detection": ("author SERUM detector; separate frozen teacher and student clean-calibration thresholds"
                            if carrier == "serum" else "official tokenizer+detector; |z| primary + signed-z secondary"),
              "execute": "bash run_serum.sh" if carrier == "serum" else "bash run_bitmark_lora.sh"}, indent=2))
        return
    if a.resume:
        run = a.resume.resolve()
    else:
        run = a.output.resolve() if a.output else results_root() / datetime.datetime.now(ZoneInfo("Asia/Vientiane")).strftime(f"run_{carrier}_%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True, exist_ok=False)
    # Drive FUSE may not support flock; synchronize same-host launches on local storage.
    import hashlib
    lock_dir = runtime_root() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / (hashlib.sha256(str(run).encode()).hexdigest() + ".lock")
    with lock_path.open("w") as lock, persistent_log(run):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Persistent run: {run}", flush=True)
        hardware = preflight(a)
        freeze(run / "config.json", c)
        source_files = {str(p.relative_to(HERE)): sha256(p) for p in HERE.glob("*.py")}
        source_files["vendor/provenance.json"] = sha256(HERE / "vendor/provenance.json")
        source_files["requirements.txt"] = sha256(HERE / "requirements.txt")
        freeze(run / "code_manifest.json", source_files)
        versions = {p: importlib.metadata.version(p) for p in (
            "torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
            "datasets", "pyarrow", "numpy", "scipy", "torch-fidelity", "huggingface-hub")}
        if carrier == "bitmark":
            versions["flash-attn"] = importlib.metadata.version("flash-attn")
        freeze(run / "environment.json", {"packages": versions, "hardware": hardware})
        print(f"Run: {run}\nDefault mode EXECUTES training and evaluation, not plan-only.", flush=True)
        py = sys.executable
        def call(file, label, *args):
            stage(run, label, [py, str(HERE / file), "--run", str(run), *args])
        call(assets_file, "01_assets")
        call(teacher_file, "02_teacher", "--stage", "generate")
        call(teacher_file, "03_teacher_scores", "--stage", "score", "--directories",
             "teacher_clean_calibration", "teacher_clean_test", "teacher_wm_test")
        if carrier == "serum":
            call("student.py", "03b_student_calibration_images", "--stage", "generate", "--branch", "unadapted",
                 "--split", "calibration")
            call(teacher_file, "03c_student_calibration_scores", "--stage", "score", "--directories", "unadapted_calibration")
        call("report.py", "04_freeze_threshold", "--calibrate-only")
        if c.get("checkpoint_steps"):
            call("validation.py", "04b_validation_cache", "--stage", "cache")
        call("student.py", "05_unadapted", "--stage", "generate", "--branch", "unadapted")
        for b in branches(c):
            call("student.py", f"train_{b['name']}", "--stage", "train", "--branch", b["name"])
            if c.get("checkpoint_steps"):
                from validation import validation_label
                call("validation.py", f"validate_{b['name']}", "--stage", "evaluate", "--branch", b["name"])
                labels = [validation_label(b["name"], step) for step in c["checkpoint_steps"]]
                call(teacher_file, f"validation_scores_{b['name']}", "--stage", "score", "--directories", *labels)
                if not c["skip_utility"]:
                    call("utility.py", f"validation_quality_{b['name']}", "--split", "validation", "--labels", *labels)
                call("validation.py", f"select_{b['name']}", "--stage", "select", "--branch", b["name"])
        # Final test is generated only after every branch has frozen its selection.
        for b in branches(c):
            call("student.py", f"generate_{b['name']}", "--stage", "generate", "--branch", b["name"])
        call(teacher_file, "06_student_scores", "--stage", "score", "--directories",
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

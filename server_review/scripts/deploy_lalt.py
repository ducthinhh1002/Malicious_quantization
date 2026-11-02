"""Deploy an isolated LALT long-training snapshot and launch verification then diagnostics."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys

import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from new_watermark.learning_channel.run import source_manifest


def main():
    tag = "lalt_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    remote = "/root/logs/" + tag
    old = "/root/logs/lcf_candidate_20261017_191721Z"
    result_root = "/root/logs/results/" + tag
    script = remote + ".sh"
    record = dict(folder=remote, session=tag, smoke=result_root + "_smoke",
        diagnostic=result_root + "_diagnostic", script=script, log=remote + ".log",
        tests_log=remote + "_tests.log", smoke_log=remote + "_smoke.log", gpu_check=result_root + "_gpu_check.json")
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "output_artifacts/vm4_known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("171.232.252.207", username="root",
                   key_filename=str(Path.home() / ".ssh/id_ed25519_vm4"), timeout=20)

    def command(text):
        _, stdout, stderr = client.exec_command(text, timeout=60)
        output, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error + output)
        return output

    try:
        command(f"test ! -e {shlex.quote(remote)} && cp -a -- {shlex.quote(old)} {shlex.quote(remote)}")
        files = source_manifest()
        for name in ("new_watermark/learning_channel/README.md",):
            files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        with client.open_sftp() as sftp:
            for name, digest in files.items():
                sftp.put(str(ROOT / name), remote + "/" + name)
                with sftp.open(remote + "/" + name, "rb") as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise RuntimeError("Upload checksum mismatch: " + name)
            body = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export NEW_WATERMARK_PYTHON=/root/logs/gate_env/bin/python
export LCF_CACHE=/root/logs/runtime/cache
export LCF_ARTIFACT_ROOT=/root/logs/runtime/lcf_artifacts
export LCF_IMAGE_ROOT=/root/logs/output_image/lcf
export OMP_NUM_THREADS=8
cd {shlex.quote(remote)}
exec > {shlex.quote(record['log'])} 2>&1
"$NEW_WATERMARK_PYTHON" -m unittest new_watermark.learning_channel.test_learning_channel -v > {shlex.quote(record['tests_log'])} 2>&1
bash -n run_learning_channel_gate.sh run_learning_channel_diagnostics.sh run_latent_transport.sh
bash run_latent_transport.sh --profile smoke --output {shlex.quote(record['smoke'])} > {shlex.quote(record['smoke_log'])} 2>&1
"$NEW_WATERMARK_PYTHON" -u -m new_watermark.learning_channel.transport_gpu_check --run {shlex.quote(record['smoke'])} --output {shlex.quote(record['gpu_check'])}
bash run_latent_transport.sh --output {shlex.quote(record['diagnostic'])}
'''
            with sftp.open(script, "w") as stream:
                stream.write(body)
            sftp.chmod(script, 0o700)
        command("bash -n " + shlex.quote(script))
        command("tmux new-session -d -s " + shlex.quote(tag) + " bash " + shlex.quote(script))
        record.update(source_hashes=files, launched_at_utc=datetime.now(timezone.utc).isoformat(),
            old_run_unchanged=True, long_run_includes_ordinary_students_if_teacher_gate_passes=True)
        folder = ROOT / "server_review/lalt_deployment"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "deployment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

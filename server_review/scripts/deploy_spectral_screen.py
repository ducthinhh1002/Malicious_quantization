"""Deploy a checksummed snapshot, test on the server, then smoke-gate the serial suite."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from new_watermark.learning_channel.run import source_manifest


def main():
    tag = "lalt_spectral_screen_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    snapshot = "/root/logs/" + tag
    reports = "/root/logs/results/" + tag
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "output_artifacts/vm4_known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("171.232.252.207", username="root",
                   key_filename=str(Path.home() / ".ssh/id_ed25519_vm4"), timeout=20)

    def command(body, timeout=120):
        _, stdout, stderr = client.exec_command(body, timeout=timeout)
        result, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status():
            raise RuntimeError(result + error)
        return result

    try:
        # Read availability once more immediately before launch. Never interrupt another job.
        status = command("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
        if int(status.strip()) < 24000:
            raise RuntimeError("Need at least 24000 MiB free for the spectral carrier screen")
        free_kib = int(command("df --output=avail /root/logs | tail -1").strip())
        if free_kib < 2 * 1024**2:
            raise RuntimeError("Need at least 2 GiB free for images/checkpoints; nothing deleted")
        command("test ! -e " + shlex.quote(snapshot) +
                " && cp -a /root/logs/lalt_stable_reader_20261018_111112Z " + shlex.quote(snapshot))
        hashes = source_manifest()
        with client.open_sftp() as sftp:
            for name, digest in hashes.items():
                sftp.put(str(ROOT / name), snapshot + "/" + name)
                with sftp.open(snapshot + "/" + name, "rb") as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise RuntimeError("Source upload mismatch: " + name)
            script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {shlex.quote(snapshot)}
exec > {shlex.quote(snapshot + '.log')} 2>&1
exec /root/logs/gate_env/bin/python -u -m new_watermark.learning_channel.spectral_transport_screen --output {shlex.quote(reports)}
"""
            with sftp.open(snapshot + ".sh", "w") as stream:
                stream.write(script)
        command("bash -n " + shlex.quote(snapshot + ".sh"))
        test_command = (
            "cd " + shlex.quote(snapshot) + " && OMP_NUM_THREADS=4 USE_TF=0 USE_FLAX=0 "
            "CUDA_VISIBLE_DEVICES='' /root/logs/gate_env/bin/python -m unittest "
            "new_watermark.learning_channel.test_learning_channel "
            "new_watermark.learning_channel.test_reader_stability "
            "new_watermark.learning_channel.test_transfer_diagnostic "
            "new_watermark.learning_channel.test_transfer_repair new_watermark.learning_channel.test_contrast_replay new_watermark.learning_channel.test_transfer_readout_audit new_watermark.learning_channel.test_spectral_transport -q")
        tests = command(test_command + " 2>&1", timeout=180)
        print(tests)
        local = ROOT / "server_review/lalt_spectral_screen" / tag
        local.mkdir(parents=True, exist_ok=False)
        record = dict(session=tag, snapshot=snapshot, reports=reports,
            log=snapshot+".log", source_hashes=hashes,
            server_tests=tests, modes=["spectral_carrier_screen"],
            student_retraining=False, fit_images=32, validation_images=64, angles=[.03,.06,.12,.24],
            reuse="Teacher/reader, clean latent cache, clean PNG datasets, clean students and clean query/calibration images")
        (local / "deployment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        command("tmux new-session -d -s " + shlex.quote(tag) + " bash " + shlex.quote(snapshot + ".sh"))
        print(json.dumps({k: v for k, v in record.items() if k not in ("source_hashes", "server_tests")}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

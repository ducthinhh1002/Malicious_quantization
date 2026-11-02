"""Read-only progress check and local evidence copy for the contrast/replay suite."""
from datetime import datetime, timezone
import json
from pathlib import Path
import paramiko
import shlex

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = sorted((ROOT / "server_review/lalt_contrast").glob("lalt_contrast_*"))[-1]
    deployment = json.loads((folder / "deployment.json").read_text())
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "output_artifacts/vm4_known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("171.232.252.207", username="root",
        key_filename=str(Path.home() / ".ssh/id_ed25519_vm4"), timeout=20)
    evidence = dict(checked_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with client.open_sftp() as sftp:
            for kind, remote in (("smoke", deployment["smoke_reports"]), ("full", deployment["reports"])):
                names = ["status.json", "outcomes.json", "expansion_selection.json"]
                for mode in ("contrast_cold", "contrast_replay"):
                    for step in (32, 64, 96):
                        names += [mode + "/" + pattern.format(step=step) for pattern in (
                            "learning_proxy_step{step}.json", "learning_teacher_validation_step{step}.json",
                            "preservation_step{step}.json")]
                    names += [mode + "/" + n for n in (
                        "stage_status.json", "learning_selection.json", "learning_teacher_baseline.json",
                        "learning_proxy_baseline.json", "learning_gradient_audit.json",
                        "objective_gradient_comparison.json", "learning_channel_training.jsonl")]
                for name in names:
                    try:
                        with sftp.open(remote + "/" + name, "rb") as stream:
                            content = stream.read()
                    except FileNotFoundError:
                        continue
                    target = folder / "latest" / kind / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    if name.endswith("training.jsonl"):
                        lines = content.decode().splitlines()
                        evidence[kind + "/" + name] = json.loads(lines[-1]) if lines else None
                    elif name == "status.json" or name.endswith("learning_selection.json"):
                        evidence[kind + "/" + name] = json.loads(content)
        for name, command in (
            ("gpu", "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"),
            ("wrapper_log_tail", "tail -8 " + shlex.quote(deployment["log"]))):
            _, stdout, stderr = client.exec_command(command, timeout=20)
            evidence[name] = stdout.read().decode()
            error = stderr.read().decode()
            if stdout.channel.recv_exit_status():
                evidence[name + "_error"] = error
        (folder / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

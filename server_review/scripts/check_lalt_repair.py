"""Read-only status collection for one recorded LALT repair deployment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shlex
import paramiko

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.deployment.read_text(encoding="utf-8"))
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "output_artifacts/vm4_known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("171.232.252.207", username="root",
                   key_filename=str(Path.home() / ".ssh/id_ed25519_vm4"), timeout=20)
    try:
        result = dict(checked_at=datetime.now(timezone.utc).isoformat(), session=record["session"], files={})
        folder = args.deployment.parent / "latest"
        folder.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            requests = [(record["smoke_reports"], "smoke", [
                "status.json", "stage_status.json", "gradient_smoke/learning_gradient_audit.json",
                "gradient_smoke/channel_audit.json", "gradient_smoke/learning_channel_training.jsonl",
                "gradient_smoke/repair_result.json"])]
            requests.append((record["reports"], "suite", [
                "status.json", "outcomes.json", "long_rank_expansion.json"] +
                [mode + "/" + name for mode in ("adaptive", "shared", "amplitude")
                 for name in ("stage_status.json", "repair_progress.json", "repair_result.json",
                              "learning_selection.json", "learning_gradient_audit.json")]))
            for remote, label, names in requests:
                for name in names:
                    try:
                        with sftp.open(str(PurePosixPath(remote) / name), "rb") as stream:
                            content = stream.read()
                    except FileNotFoundError:
                        continue
                    destination = folder / label / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    if name.endswith(".json") and "channel_audit" not in name:
                        result["files"][label + "/" + name] = json.loads(content)
            log_paths = [record["log"], record["smoke_reports"] + "/gradient_smoke.log"]
            log_paths += [record["reports"] + "/" + m + "/warmup.log"
                         for m in ("adaptive", "shared", "amplitude")]
            for index, path in enumerate(log_paths):
                try:
                    with sftp.open(path, "rb") as stream:
                        size = stream.stat().st_size
                        stream.seek(max(0, size - 5000))
                        tail = stream.read().decode(errors="replace")
                    (folder / f"log_{index}_tail.txt").write_text(tail, encoding="utf-8")
                except FileNotFoundError:
                    pass
        _, out, err = client.exec_command(
            "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv && "
            "tmux has-session -t " + shlex.quote(record["session"]), timeout=20)
        result["gpu"] = out.read().decode()
        result["session_alive"] = out.channel.recv_exit_status() == 0
        result["command_error"] = err.read().decode()
        (args.deployment.parent / "latest_status.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

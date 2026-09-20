"""Verify the final local collection copy, then acknowledge the armed server guard."""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from backup_run import REMOTE, HOST, DEADLINE, save


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def verify(outer):
    path = outer / "collection_export_manifest.json"
    manifest = read(path)
    assert manifest.get("scope") == "collection_only_user_stop"
    failures, size = [], 0
    for relative, expected in manifest["files"].items():
        file = outer / relative
        assert file.resolve().is_relative_to(outer.resolve())
        if not file.is_file() or file.is_symlink():
            failures.append(relative)
            continue
        h = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if file.stat().st_size != expected["bytes"] or h.hexdigest() != expected["sha256"]:
            failures.append(relative)
        size += file.stat().st_size
    return {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified_files": len(manifest["files"]) - len(failures),
            "verified_bytes": size, "failed_files": failures, "verified_epoch": time.time()}


def main(project):
    project = Path(project)
    records = project / "runs/v15/backup"
    outer = project / "runs/v15/remote/r2"
    status_path = records / "shutdown_coordination.json"
    save(status_path, {"stage": "waiting_for_final_backup", "epoch": time.time()})
    while time.time() < DEADLINE:
        if (read(records / "status.json").get("stage") == "completed"
                and read(outer / "finalization_summary.json").get("scope") == "collection_only_user_stop"):
            break
        time.sleep(10)
    else:
        save(status_path, {"stage": "backup_not_complete_before_deadline", "epoch": time.time()})
        return
    ack = verify(outer)
    save(records / "collection_export_verification.json", ack)
    if ack["failed_files"]:
        save(status_path, {"stage": "verification_failed", "details": ack})
        return
    code = ("import sys,json,hashlib,pathlib;"
            f"p=pathlib.Path({(REMOTE + '/r2')!r});"
            "a=json.load(sys.stdin);"
            "assert hashlib.sha256((p/'collection_export_manifest.json').read_bytes()).hexdigest()==a['manifest_sha256'];"
            "assert not a['failed_files'];"
            "tmp=p/'backup_verified_ack.tmp';"
            "tmp.write_text(json.dumps(a));tmp.replace(p/'backup_verified_ack.json');print('acknowledged')")
    command = ["/usr/bin/ssh", "-p", "33364", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
               "/root/autodl-tmp/moe-plasticity-v13-20260920/venv/bin/python -c " + shlex.quote(code)]
    result = subprocess.run(command, input=json.dumps(ack), capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or result.stdout.strip() != "acknowledged":
        save(status_path, {"stage": "acknowledgement_failed", "ssh_exit_code": result.returncode})
        return
    save(status_path, {"stage": "backup_verified_acknowledged_waiting_for_shutdown", "epoch": time.time(), **ack})
    failures = 0
    for _ in range(18):
        time.sleep(10)
        check = subprocess.run(["/usr/bin/ssh", "-p", "33364", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=5", HOST, "true"], capture_output=True, timeout=10)
        failures = failures + 1 if check.returncode != 0 else 0
        if failures >= 3:
            save(status_path, {"stage": "ssh_unreachable_after_shutdown_ack", "epoch": time.time(),
                     "backup_verified": True, "consecutive_failed_connections": failures,
                     "scope": "Server guard was armed with provider shutdown; SSH unavailable after acknowledgement. Billing state not independently queried."})
            return
    save(status_path, {"stage": "server_still_reachable_needs_check", "epoch": time.time()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--project", required=True)
    main(p.parse_args().project)

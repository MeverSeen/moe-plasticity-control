"""Finite local backup of the authorized run, ending at its existing deadline.

Wait for the initial transfer, then synchronize new artifacts without deletion.
This copies files only; it never starts GPU work or changes server billing/state.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REMOTE = "/root/autodl-tmp/moe-v15-offline-20260920"
HOST = "root@connect.nma1.seetacloud.com"
DEADLINE = 1789938900.


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def initial_running(pid):
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True)
    return result.returncode == 0 and "rsync -az --stats" in result.stdout and REMOTE in result.stdout


def checkpoint_audit(destination, cache):
    indexed = {}
    for metadata in destination.rglob("*.json"):
        if any(x in metadata.parts for x in (".rsync-partial", "__pycache__")):
            continue
        try:
            data = json.loads(metadata.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("file", "endpoint_adapter"):
            item = data.get(key)
            if not isinstance(item, dict) or not str(item.get("path", "")).startswith(REMOTE + "/"):
                continue
            relative = str(Path(item["path"]).relative_to(REMOTE))
            path = destination / relative
            if not path.is_file() or path.is_symlink():
                indexed[relative] = {"status": "missing"}
                continue
            stat = path.stat()
            old = cache.get(relative, {})
            if old.get("bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
                sha = old["sha256"]
            else:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        h.update(block)
                sha = h.hexdigest()
            indexed[relative] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                 "sha256": sha, "expected_sha256": item.get("sha256"),
                                 "status": "verified" if sha == item.get("sha256") else "mismatch"}
    return indexed


def main(project, initial_pid):
    project = Path(project)
    destination = project / "runs/v15/remote"
    records = project / "runs/v15/backup"
    records.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "started_epoch": time.time(),
              "authorization": "User explicitly replied 备份 after request covering data, logs and recoverable checkpoints.",
              "deadline_epoch": DEADLINE, "initial_transfer_pid": initial_pid,
              "destination": str(destination), "cycles": [], "stage": "waiting_for_initial_copy"}
    save(records / "status.json", status)
    while time.time() < DEADLINE and initial_running(initial_pid):
        time.sleep(15)
    cache = checkpoint_audit(destination, {})
    save(records / "checkpoints.json", cache)
    while time.time() < DEADLINE - 30:
        try:
            state = subprocess.run(["/usr/bin/ssh", "-p", "33364", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=15", HOST,
                f"cat {REMOTE}/r2/finalization_summary.json 2>/dev/null || true"],
                capture_output=True, text=True, timeout=30)
            finished = json.loads(state.stdout).get("status") == "report_written"
        except (ValueError, subprocess.TimeoutExpired):
            finished = False
        command = ["/usr/bin/rsync", "-az", "--stats", "--timeout=120",
                   "--partial-dir=.rsync-partial", "-e",
                   "ssh -p 33364 -o BatchMode=yes -o ConnectTimeout=15",
                   "--exclude=*.tmp", "--exclude=__pycache__", "--exclude=._*"]
        if not finished:
            command += ["--exclude=emergency.pt"]
        command += [f"{HOST}:{REMOTE}/", str(destination) + "/"]
        started = time.time()
        status.update(stage="copying_final" if finished else "copying_incremental", last_started_epoch=started)
        save(records / "status.json", status)
        try:
            with (records / "transfer.log").open("ab", buffering=0) as log:
                result = subprocess.run(command, stdout=log, stderr=log,
                                        timeout=max(1, DEADLINE - time.time()))
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = "deadline"
        cache = checkpoint_audit(destination, cache)
        save(records / "checkpoints.json", cache)
        status["cycles"].append({"started_epoch": started, "ended_epoch": time.time(),
                                  "exit_code": code, "server_finalized_before_copy": finished,
                                  "verified_checkpoints": sum(x["status"] == "verified" for x in cache.values())})
        if finished and code == 0 and all(x["status"] == "verified" for x in cache.values()):
            status.update(stage="completed", ended_epoch=time.time())
            save(records / "status.json", status)
            return
        status["stage"] = "waiting_for_new_outputs"
        save(records / "status.json", status)
        time.sleep(max(0, min(180, DEADLINE - time.time())))
    status.update(stage="bounded_stop", ended_epoch=time.time(),
                  note="Existing deadline reached; verified local files remain. No additional GPU runtime requested.")
    save(records / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--initial-pid", type=int, required=True)
    a = p.parse_args()
    main(a.project, a.initial_pid)

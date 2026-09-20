"""Bound only our child process group; never change instance billing/state."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if time.time() >= args.deadline - 180:
        raise SystemExit("Insufficient authorized time to start")
    proc = subprocess.Popen(command, start_new_session=True)
    receipt = {"pid": proc.pid, "supervisor_pid": os.getpid(), "started_epoch": time.time(),
               "deadline_epoch": args.deadline, "command": command}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    requested = False
    while proc.poll() is None:
        now = time.time()
        if now >= args.deadline - 180 and not requested:
            os.killpg(proc.pid, signal.SIGUSR1); requested = True
        if now >= args.deadline:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(2)
    receipt.update(ended_epoch=time.time(), exit_code=proc.wait(), checkpoint_requested=requested)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

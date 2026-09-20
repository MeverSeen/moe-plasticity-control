"""Optional earlier collection cap, preserving time for the locked main table.

This guard only signals the exact collection process recorded by its supervisor.
It does not change the environment, behaviors, targets, fitting, or hard deadline.
"""
import json
import os
import signal
import time
from common import run_root, save_json


def main():
    root = run_root()
    receipt = json.loads((root / "collection_receipt.json").read_text())
    deadline = receipt["started_epoch"] + 3.25 * 3600
    note = {"created_epoch": time.time(), "collection_stop_epoch": deadline,
            "collection_cap_hours": 3.25, "hard_deadline_unchanged": True,
            "reason": "Measured deterministic replay costs 92.21s/10 updates; fresh target calibration is required. Reserve remaining authorized time for offline fits and the full locked comparison, regardless of method outcomes.",
            "collection_pid": receipt["pid"], "status": "waiting"}
    save_json(root / "budget_reserve.json", note)
    while time.time() < deadline:
        current = json.loads((root / "collection_receipt.json").read_text())
        if "exit_code" in current:
            note.update(status="collection_ended_before_cap", ended_epoch=time.time())
            save_json(root / "budget_reserve.json", note)
            return
        time.sleep(min(10, deadline - time.time()))
    current = json.loads((root / "collection_receipt.json").read_text())
    if "exit_code" in current:
        note.update(status="collection_ended_before_cap", ended_epoch=time.time())
    else:
        pid = current["pid"]
        command = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        assert any(x.endswith(b"/v15_offline_control/collect.py") for x in command)
        assert str(root / "data").encode() in command
        os.kill(pid, signal.SIGUSR1)
        note.update(status="graceful_checkpoint_requested", requested_epoch=time.time())
    save_json(root / "budget_reserve.json", note)


if __name__ == "__main__":
    main()

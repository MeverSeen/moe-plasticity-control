"""User-authorized collection-only finalization and AutoDL instance shutdown."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import run_root, save_json, file_sha

SHUTDOWN = Path("/usr/bin/shutdown")
SHUTDOWN_SHA = "0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027"
TRANSIENT = {"collection_export_manifest.json", "finalization_summary.json",
             "shutdown_status.json", "backup_verified_ack.json", "shutdown_request.json",
             "shutdown_output.log", "shutdown_collection.log"}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def acknowledged(ack, manifest_sha, files):
    return (ack.get("manifest_sha256") == manifest_sha
            and ack.get("verified_files") == len(files)
            and ack.get("verified_bytes") == sum(f["bytes"] for f in files.values())
            and ack.get("failed_files") == [])


def finalize_collection(outer):
    data = outer / "data"
    summary = read(data / "collection_summary.json")
    records = [read(p) for p in sorted((data / "results/train").glob("*.json"))]
    records = [r for r in records if r.get("history_completed")]
    analyses = None
    if records:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze.py")),
                                 "--root", str(data), "--output", str(data / "analysis")],
                                capture_output=True, text=True, timeout=120)
        analyses = {"exit_code": result.returncode}
        (data / "analysis/collection_analysis.log").write_text(result.stdout + result.stderr)
    facts = {"scope": "collection_only_user_stop", "histories": len(records),
             "roots": sorted({r["root"]["id"] for r in records}),
             "decisions": sum(len(r["trace"]) for r in records),
             "actual_learner_updates": summary.get("native_cost", {}).get("updates"),
             "collection_status": summary.get("status", "missing_summary"),
             "collection_receipt": read(outer / "collection_receipt.json"),
             "analysis": analyses, "offline_training": "cancelled_by_user_before_start",
             "closed_loop_evaluation": "cancelled_by_user_before_start"}
    save_json(data / "collection_only_facts.json", facts)
    (outer / "结果解读.md").write_text(
        "# v15 采集阶段收尾\n\n用户要求采集结束后停止并关闭服务器，已取消后续离线控制器训练与闭环评价。\n\n"
        f"实际保存 {facts['histories']} 条完整训练分支、{len(facts['roots'])} 个训练根、"
        f"{facts['decisions']} 个控制决策。采集阶段总计 {facts['actual_learner_updates']} 次 learner 更新，"
        "包括参考校准、共享前缀、恢复检查和实际分支；不能把这些全算成控制器训练数据。\n\n"
        "原始观测、动作、监测 CE、参数变化、成本和可恢复检查点均保留。时间预算中断的历史单列，"
        "不伪装成任务失败。尚无新控制器训练或独立闭环效果结论。\n\n"
        "完整数量及状态见 data/collection_only_facts.json；训练池描述性统计见 data/analysis。\n")
    files = {}
    for path in sorted(outer.rglob("*")):
        if (not path.is_file() or path.is_symlink() or path.name in TRANSIENT
                or path.name.startswith("._") or path.name.endswith(".tmp")
                or "__pycache__" in path.parts):
            continue
        files[str(path.relative_to(outer))] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
    manifest = {"scope": "collection_only_user_stop", "created_epoch": time.time(), "files": files}
    save_json(outer / "collection_export_manifest.json", manifest)
    sha = file_sha(outer / "collection_export_manifest.json")
    save_json(outer / "finalization_summary.json", {"status": "report_written",
              "scope": "collection_only_user_stop", "manifest_sha256": sha, "facts": facts})
    return manifest, sha


def main():
    outer = run_root()
    plan = read(outer / "code/experiments/v15_offline_control/protocol.json")
    assert read(outer / "stop_after_collection.json").get("shutdown_after_collection_and_backup") is True
    assert SHUTDOWN.is_file() and file_sha(SHUTDOWN) == SHUTDOWN_SHA
    save_json(outer / "shutdown_status.json", {"stage": "waiting_for_collection", "epoch": time.time()})
    while "exit_code" not in read(outer / "collection_receipt.json"):
        if time.time() > plan["hard_deadline_epoch"]:
            raise RuntimeError("Collection has no terminal receipt at the hard deadline; inspect before shutdown")
        time.sleep(5)
    # The cancelled pipeline cannot start fitting after collection completes.
    assert read(outer / "pipeline_receipt.json").get("exit_code") is not None
    try:
        manifest, sha = finalize_collection(outer)
    except Exception as error:
        save_json(outer / "shutdown_status.json", {"stage": "finalization_failed", "error": repr(error)})
        raise
    until = min(time.time() + 1800, plan["hard_deadline_epoch"] - 30)
    save_json(outer / "shutdown_status.json", {"stage": "waiting_for_verified_backup", "epoch": time.time(),
              "manifest_sha256": sha, "backup_grace_deadline_epoch": until,
              "fallback": "Power off after at most 30 minutes; preserve remote disks, never release/delete instance."})
    verified = False
    while time.time() < until:
        if acknowledged(read(outer / "backup_verified_ack.json"), sha, manifest["files"]):
            verified = True
            break
        time.sleep(5)
    assert file_sha(SHUTDOWN) == SHUTDOWN_SHA
    request = {"requested_epoch": time.time(), "backup_verified": verified, "manifest_sha256": sha,
               "reason": "collection_ended_backup_verified" if verified else "collection_ended_backup_grace_expired",
               "authorization": "User: 采集结束后先停止，关闭服务器", "command": str(SHUTDOWN)}
    save_json(outer / "shutdown_request.json", request)
    save_json(outer / "shutdown_status.json", {"stage": "shutdown_requested", **request})
    os.sync()
    with (outer / "shutdown_output.log").open("ab", buffering=0) as log:
        result = subprocess.run([str(SHUTDOWN)], stdout=log, stderr=log, timeout=60)
    save_json(outer / "shutdown_status.json", {"stage": "shutdown_command_returned",
              "exit_code": result.returncode, **request})


if __name__ == "__main__":
    main()

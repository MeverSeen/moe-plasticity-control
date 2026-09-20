"""Write a factual run report after the supervised pipeline; no model selection."""
import json
import time
from pathlib import Path
import torch
from common import run_root, save_json, file_sha
from validate_run import validate


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def report(outer):
    root = outer / "data"
    pipeline = read(outer / "pipeline_receipt.json")
    collection = read(root / "collection_summary.json")
    evaluation = read(root / "evaluation_summary.json")
    dataset = read(root / "analysis/dataset_summary.json")
    fits = read(root / "offline/summary.json")
    selection = read(root / "selection_lock.json")
    reserve = read(outer / "budget_reserve.json")
    try:
        integrity = validate(outer, root / "analysis/validation.json")
        audit = {"status": "passed", "decisions": integrity["decisions"]}
    except Exception as error:
        audit = {"status": "not_passed", "reason": repr(error),
                 "meaning": "Missing or inconsistent artifacts require inspection; no automatic efficacy claim."}
        save_json(root / "analysis/validation.json", audit)
    facts = {"written_epoch": time.time(), "pipeline": pipeline,
             "collection_status": collection.get("status", "missing"),
             "evaluation_status": evaluation.get("status", "missing"),
             "training_histories": dataset.get("histories"),
             "training_roots": dataset.get("roots"), "training_decisions": dataset.get("decisions"),
             "collection_updates": collection.get("native_cost", {}).get("updates"),
             "evaluation_updates": evaluation.get("native_cost", {}).get("updates"),
             "offline_fits": fits.get("fits", []), "selection": selection,
             "evaluation_rows": evaluation.get("rows", read(root / "evaluation_rows.json") or []),
             "budget_reserve": reserve, "integrity": audit}
    save_json(root / "analysis/final_facts.json", facts)
    lines = ["# v15 固定轨迹上的奖励重标与离线控制", "",
             "本报告由已保存记录自动汇总；尚未作额外科学解释。缺失实验不填零，不记作 learner 失败。", "",
             f"采集状态：`{facts['collection_status']}`；评价状态：`{facts['evaluation_status']}`；完整性核验：`{audit['status']}`。", "",
             f"训练数据：{facts['training_histories']} 条完整分支，{facts['training_roots']} 个根，{facts['training_decisions']} 个控制决策。校准、恢复检查和共享前缀不计作新的训练历史。", "",
             f"实际 learner 更新：采集阶段 {facts['collection_updates']}，评价阶段 {facts['evaluation_updates']}。首轮停止与独立复现诊断另计，不在这两个数中。", "",
             "## 开发与确认实跑表", "",
             "F 是独立任务评价 CE 相对各自完成时的上升正部；更新数含共享 A 前缀的逻辑成本。不同根的 CE 不代表相同任务难度。", "",
             "| 阶段 | 根 | 方法 | 失败任务 | F↓ | 最终平均 CE↓ | 路径更新 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in facts["evaluation_rows"]:
        lines.append(f"| {row['phase']} | {row['root']} | {row['method']} | {row['failures']} | {row['F']:.6f} | {row['final_ce']:.6f} | {row['updates']} |")
    lines += ["", f"冻结选中的 IQL：`{selection.get('selected', '尚无完整开发选择')}`。选择依据为开发根的失败数、遗忘、绝对 CE、更新数；确认结果不进入选模。", "",
              "## 奖励与结论边界", "",
              "三种奖励共享同一批真实轨迹、相同观测与结构化动作，分别使用正部遗忘、旧任务绝对 CE、相对 A 后前缀的有效 LoRA BA 距离。BC 和 IQL 使用相同 actor 更新预算。", "",
              "距离小不等于遗忘小；奖励有差异也不等于离线控制器已学会利用差异。必须结合达标、独立遗忘、绝对表现和成本判断。单种子、4个训练根及已曝光留出任务池不能支持普遍有效或正式显著性结论。", "",
              "r2 采用固定 CUDA 工作区及确定性算法，重新校准目标；首轮恢复失败的轨迹不混入数据。相同动作重放的验证范围是本硬件/软件组合，不追溯消除 v14 的执行异常。", "",
              "原始过程和终点记录见 `data/results`，训练清单和检查点见 `data/offline`，成本见阶段 summary。通用能力、生成评分与逐任务 CE 保存在各确认 JSON 中；开启同一部署适配器评价，不进入训练奖励。", "",
              "采集提前收尾依据实测吞吐与完整评价所需时间，按耗时触发，不按结果优劣筛选历史。总授权硬截止不顺延。"]
    (outer / "结果解读.md").write_text("\n".join(lines) + "\n")
    files = {}
    for directory in (outer / "code", root / "results", root / "targets", root / "references", root / "analysis", root / "offline"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
                files[str(path.relative_to(outer))] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    save_json(outer / "artifact_manifest.json", {"files": files, "created_epoch": time.time(),
              "scope": "Source, results, calibration, analysis and offline artifacts. Heavy learner prefixes/adapters retain their per-file metadata; not rehashed here."})
    return facts


def main():
    torch.set_num_threads(1)
    outer = run_root()
    plan = read(outer / "code/experiments/v15_offline_control/protocol.json")
    deadline = plan["hard_deadline_epoch"]
    while time.time() < deadline - 90:
        if "exit_code" in read(outer / "pipeline_receipt.json"):
            break
        time.sleep(20)
    facts = report(outer)
    save_json(outer / "finalization_summary.json", {"status": "report_written", "written_epoch": time.time(),
               "pipeline_exit_code": facts["pipeline"].get("exit_code"), "integrity": facts["integrity"]})


if __name__ == "__main__":
    main()

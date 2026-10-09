"""Development-only analysis for v11 expert-level RL diagnostics.

Rows are rollout records, not independent observations.  Sampled policy draws
are averaged within root before any across-root summary or comparison.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CHECKPOINTS = (0, 3, 6, 24, 48)


def _finite(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _describe(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": int(len(values)), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max())}


def _matrix(row, field, k):
    matrix = row.get(field)
    if not isinstance(matrix, list) or len(matrix) != k or any(not isinstance(line, list) or len(line) != k for line in matrix):
        raise ValueError(f"root={row.get('root')} has invalid {field}")
    for learned in range(k):
        for tested in range(learned + 1):
            _finite(matrix[learned][tested], f"root={row.get('root')} {field}[{learned}][{tested}]")
    return matrix


def row_metrics(row):
    """Endpoint metrics, using each task's original diagonal as its reference."""
    for field in ("rowkind", "arm", "root", "mode", "tasks", "loss_matrix", "accuracy_matrix", "cost"):
        if field not in row:
            raise ValueError(f"evaluation lacks {field}")
    if row["rowkind"] not in ("learned", "baseline"):
        raise ValueError("rowkind must be learned or baseline")
    if row["mode"] not in ("greedy", "sample"):
        raise ValueError("mode must be greedy or sample")
    if row["rowkind"] == "learned":
        if not isinstance(row.get("policy_seed"), int) or not isinstance(row.get("epoch"), int):
            raise ValueError("learned rows need integer policy_seed and epoch")
    tasks = row["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be nonempty")
    k = len(tasks)
    loss, accuracy = _matrix(row, "loss_matrix", k), _matrix(row, "accuracy_matrix", k)
    diagonal = [float(loss[i][i]) for i in range(k)]
    final_loss = [float(loss[k - 1][i]) for i in range(k)]
    final_accuracy = [float(accuracy[k - 1][i]) for i in range(k)]
    deltas = [final_loss[i] - diagonal[i] for i in range(k - 1)]
    success = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or "success" not in task:
            raise ValueError(f"task {index} lacks success")
        success.append(bool(task["success"]))
    cost = row["cost"]
    if not isinstance(cost, dict):
        raise ValueError("cost must be object")
    target = _finite(row.get("target_loss"), "target_loss")
    diagonal_hit = [value <= target for value in diagonal]
    final_hit = [value <= target for value in final_loss]
    action_counts = defaultdict(int)
    action_total = 0
    events = row.get("events", [])
    if not isinstance(events, list): raise ValueError("events must be a list")
    for event in events:
        if not isinstance(event, dict): raise ValueError("event must be object")
        # Task 0 is the shared global_1 prefix; it was not a policy decision.
        if event.get("task") == 0:
            continue
        multipliers = event.get("multipliers", [])
        if not isinstance(multipliers, list): raise ValueError("event multipliers must be list")
        for layer in multipliers:
            if not isinstance(layer, list): raise ValueError("event multiplier layer must be list")
            for value in layer:
                value = _finite(value, "event multiplier")
                action_counts[f"{value:g}"] += 1; action_total += 1
    return {
        "rowkind": row["rowkind"], "arm": row["arm"], "root": row["root"],
        "policy_seed": row.get("policy_seed"), "epoch": row.get("epoch"), "mode": row["mode"], "draw": row.get("draw"),
        "sequence_success": bool(all(success)), "task_success_rate": float(np.mean(success)),
        "failed_tasks": int(k - sum(success)), "final_positive_forgetting": _mean([max(0., x) for x in deltas]),
        "final_signed_backward_change": _mean(deltas), "final_mean_loss": _mean(final_loss),
        "final_mean_accuracy": _mean(final_accuracy),
        "diagonal_task_reach_rate": float(np.mean(diagonal_hit)), "diagonal_all_reached": bool(all(diagonal_hit)),
        "diagonal_suffix_reach_rate": float(np.mean(diagonal_hit[1:])) if k > 1 else None,
        "diagonal_suffix_all_reached": bool(all(diagonal_hit[1:])) if k > 1 else None,
        "final_task_reach_rate": float(np.mean(final_hit)), "final_all_reached": bool(all(final_hit)),
        "final_suffix_reach_rate": float(np.mean(final_hit[1:])) if k > 1 else None,
        "final_suffix_all_reached": bool(all(final_hit[1:])) if k > 1 else None,
        "action_frequencies": {name: count / action_total for name, count in action_counts.items()} if action_total else {},
        "monitor_reward": _finite(row["monitor_reward"], "monitor_reward") if "monitor_reward" in row else None,
        "learner_updates": _finite(cost.get("learner_updates", 0), "cost.learner_updates"),
        "cost": {str(name): _finite(value, f"cost.{name}") for name, value in cost.items()},
    }


def _condition(metric):
    return (metric["rowkind"], metric["arm"], metric["policy_seed"], metric["epoch"], metric["mode"])


def _flatten_numeric(value, prefix=""):
    """Yield stable dotted/indexed names for nested diagnostic log values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, float(value)
    elif isinstance(value, dict):
        for name, item in value.items():
            # Raw per-slot probes stay in their training log; copying them into
            # summary.json creates thousands of pseudo-timeseries columns.
            if name == "per_probe":
                continue
            child = f"{prefix}.{name}" if prefix else str(name)
            yield from _flatten_numeric(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten_numeric(item, f"{prefix}[{index}]")


def aggregate_roots(rows):
    """Average all sampled draws inside each condition/root before summaries.

    Greedy is deliberately a separate condition.  A greedy rollout and a
    sampled rollout are two policy-evaluation modes, not exchangeable draws.
    """
    buckets = defaultdict(list)
    for row in rows:
        metric = row_metrics(row)
        draw = metric["draw"]
        if metric["mode"] == "sample" and not isinstance(draw, int):
            raise ValueError("sample rows require integer draw")
        if metric["mode"] == "greedy" and draw is not None:
            raise ValueError("greedy rows require draw=null")
        buckets[(_condition(metric), metric["root"])].append(metric)
    result = defaultdict(list)
    fields = ("sequence_success", "task_success_rate", "failed_tasks", "final_positive_forgetting",
              "final_signed_backward_change", "final_mean_loss", "final_mean_accuracy", "learner_updates", "monitor_reward")
    for (condition, root), replicas in buckets.items():
        draw_ids = [item["draw"] for item in replicas]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError(f"duplicate draw for root={root}, condition={condition}")
        costs = defaultdict(float)
        for item in replicas:
            for name, value in item["cost"].items(): costs[name] += value
        actions = defaultdict(float)
        for item in replicas:
            for name, value in item["action_frequencies"].items(): actions[name] += value / len(replicas)
        root_item = {field: _mean([None if item[field] is None else float(item[field]) for item in replicas]) for field in fields}
        for field in ("diagonal_task_reach_rate", "diagonal_all_reached", "diagonal_suffix_reach_rate", "diagonal_suffix_all_reached",
                      "final_task_reach_rate", "final_all_reached", "final_suffix_reach_rate", "final_suffix_all_reached"):
            root_item[field] = _mean([None if item[field] is None else float(item[field]) for item in replicas])
        root_item.update(root=root, draws=len(replicas), draw_ids=sorted(draw_ids, key=lambda x: -1 if x is None else x),
                         cost_sum=dict(sorted(costs.items())), action_frequencies=dict(sorted(actions.items())))
        result[condition].append(root_item)
    return {key: sorted(value, key=lambda item: repr(item["root"])) for key, value in result.items()}


def summarize(groups):
    fields = ("sequence_success", "task_success_rate", "failed_tasks", "final_positive_forgetting",
              "final_signed_backward_change", "final_mean_loss", "final_mean_accuracy", "learner_updates", "monitor_reward")
    output = {}
    for condition, roots in groups.items():
        cost_total = defaultdict(float)
        for root in roots:
            for name, value in root["cost_sum"].items(): cost_total[name] += value
        reach_fields = ("diagonal_task_reach_rate", "diagonal_all_reached", "diagonal_suffix_reach_rate", "diagonal_suffix_all_reached",
                        "final_task_reach_rate", "final_all_reached", "final_suffix_reach_rate", "final_suffix_all_reached")
        action_names = sorted({name for root in roots for name in root["action_frequencies"]})
        output[condition] = {
            "roots": len(roots), "draws_per_root": sorted({root.get("draws", root.get("policy_seeds", 1)) for root in roots}),
            "metrics": {field: _describe([root[field] for root in roots]) for field in fields + reach_fields},
            "action_frequencies": {name: _describe([root["action_frequencies"].get(name, 0.) for root in roots]) for name in action_names},
            "cost_sum_all_rows": dict(sorted(cost_total.items())), "root_values": roots,
        }
    return output


def aggregate_policy_seeds(groups):
    """Pool learned policy seeds inside a root after its draw aggregation."""
    pooled = defaultdict(list)
    for condition, roots in groups.items():
        kind, arm, seed, epoch, mode = condition
        key = (kind, arm, epoch, mode)
        for root in roots:
            pooled[(key, root["root"])].append(root)
    fields = ("sequence_success", "task_success_rate", "failed_tasks", "final_positive_forgetting",
              "final_signed_backward_change", "final_mean_loss", "final_mean_accuracy", "learner_updates", "monitor_reward")
    result = defaultdict(list)
    for (condition, root), replicas in pooled.items():
        costs = defaultdict(float)
        actions = defaultdict(float)
        for replica in replicas:
            for name, value in replica["cost_sum"].items(): costs[name] += value
            for name, value in replica["action_frequencies"].items(): actions[name] += value / len(replicas)
        item = {field: _mean([replica[field] for replica in replicas]) for field in fields}
        for field in ("diagonal_task_reach_rate", "diagonal_all_reached", "diagonal_suffix_reach_rate", "diagonal_suffix_all_reached",
                      "final_task_reach_rate", "final_all_reached", "final_suffix_reach_rate", "final_suffix_all_reached"):
            item[field] = _mean([replica[field] for replica in replicas])
        item.update(root=root, policy_seeds=len(replicas), cost_sum=dict(sorted(costs.items())),
                    action_frequencies=dict(sorted(actions.items())))
        result[condition].append(item)
    return {key: sorted(value, key=lambda item: repr(item["root"])) for key, value in result.items()}


def selection_reconstruction(policy_seed_groups):
    """Rebuild the protocol's greedy development ranking without calling it confirmation."""
    candidates = defaultdict(list)
    for condition, roots in policy_seed_groups.items():
        kind, arm, epoch, mode = condition
        if kind == "learned" and mode == "greedy" and epoch in CHECKPOINTS:
            candidates[arm].append((epoch, roots))
    output = {}
    for key, choices in candidates.items():
        ranked = []
        for epoch, roots in choices:
            failure = _mean([1. - root["sequence_success"] for root in roots])
            forgetting = _mean([root["final_positive_forgetting"] for root in roots])
            updates = _mean([root["learner_updates"] for root in roots])
            ranked.append({"epoch": epoch, "failure_fraction": failure, "positive_forgetting": forgetting,
                           "learner_updates": updates})
        ranked.sort(key=lambda item: (item["failure_fraction"], item["positive_forgetting"], item["learner_updates"], item["epoch"]))
        output[str(key)] = {"ranking": ranked, "selected_epoch": ranked[0]["epoch"] if ranked else None,
                                          "status": "development_only"}
    return output


def paired_checkpoint_deltas(policy_seed_groups):
    """Raw same-root development deltas versus epoch zero, without inference."""
    out = {}
    contexts = defaultdict(dict)
    for (kind, arm, epoch, mode), roots in policy_seed_groups.items():
        if kind == "learned": contexts[(arm, mode)][epoch] = {item["root"]: item for item in roots}
    for (arm, mode), epochs in contexts.items():
        base = epochs.get(0)
        if not base: continue
        for epoch, values in epochs.items():
            if epoch == 0: continue
            rows = []
            for root in sorted(set(base) & set(values), key=repr):
                rows.append({"root": root,
                             "positive_forgetting_delta": values[root]["final_positive_forgetting"] - base[root]["final_positive_forgetting"],
                             "final_accuracy_delta": values[root]["final_mean_accuracy"] - base[root]["final_mean_accuracy"],
                             "sequence_success_delta": values[root]["sequence_success"] - base[root]["sequence_success"],
                             "learner_updates_delta": values[root]["learner_updates"] - base[root]["learner_updates"]})
            out[f"{arm}:{mode}:epoch{epoch}_minus_epoch0"] = rows
    return out


def training_diagnostics(run):
    """Summarize every numeric training-log field without assigning causality."""
    directory = run / "training"
    output = {}
    if not directory.exists():
        return output
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        log, cost = record.get("log", []), record.get("cost", {})
        if not isinstance(log, list) or not isinstance(cost, dict):
            raise ValueError(f"invalid training record {path.name}")
        fields = defaultdict(list)
        for entry in log:
            if not isinstance(entry, dict): raise ValueError(f"non-object log entry in {path.name}")
            for name, value in _flatten_numeric(entry): fields[name].append(value)
        output[path.stem] = {
            "entries": len(log),
            "numeric_fields": {name: {"n": len(values), "first": values[0], "last": values[-1],
                                        "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
                               for name, values in sorted(fields.items())},
            "cost": {str(name): _finite(value, f"training.{path.name}.cost.{name}") for name, value in cost.items()},
        }
    return output


def _json_groups(summary):
    return {"|".join("null" if value is None else str(value) for value in key): value
            for key, value in sorted(summary.items(), key=lambda item: repr(item[0]))}


def write_report(summary, selection, training, protocol, out):
    lines = ["# v11 逐 expert RL 训练诊断", "", "本报告是 development-only 诊断，不构成最终确认、优越性或非劣结论。每个 sample draw 先在同一 root 内平均，learned 臂再在同一 root 内平均 policy seed，最后才跨 root 汇总；因此三个采样 draw 不会被当作三个独立样本。greedy 与 sample 是不同评估模式，不能把二者的差异解释为同一策略在同一模式下的误差。", "",
             "| 条件 | roots | 每 root draws | 在线整序列达标率 | 对角全任务达标 | 最终全任务达标 | monitor reward | 独立评估最终 F | 最终平均准确率 | 实际 learner 更新 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, item in sorted(summary.items(), key=lambda pair: repr(pair[0])):
        kind, arm, seed, epoch, mode = condition; metrics = item["metrics"]
        get = lambda field: metrics[field]["mean"]
        label = f"{kind}/{arm}/seed={seed}/epoch={epoch}/{mode}"
        pct = lambda field: "NA" if get(field) is None else f"{get(field):.1%}"
        num = lambda field: "NA" if get(field) is None else f"{get(field):.4f}"
        lines.append(f"| {label} | {item['roots']} | {item['draws_per_root']} | {pct('sequence_success')} | {pct('diagonal_all_reached')} | {pct('final_all_reached')} | {num('monitor_reward')} | {num('final_positive_forgetting')} | {pct('final_mean_accuracy')} | {num('learner_updates')} |")
    lines += ["", "F 使用每个任务学习完成时的独立评估 loss 对角线作固定基准；在序列末，对旧任务 CE 增量取正后平均。它不等于最终绝对 loss，后者在 `summary.json` 单独保留。失败任务没有删除。`action_frequencies` 只统计 tasks 1–4 的实际 events；共同 task 0 的固定全 1 前缀不计入控制动作频率。", "",
              "## 贪心开发排序描述", "", "下列排序描述 greedy 开发结果的字典序：先整序列失败比例，再 F，再更新数。它用于检查诊断过程，不能转化为确认性结果。"]
    for name, item in sorted(selection.items()):
        lines += ["", f"### {name}", "", f"排序第一 epoch：`{item['selected_epoch']}`。"]
        for rank in item["ranking"]:
            lines.append(f"- epoch {rank['epoch']}: failure={rank['failure_fraction']:.4f}, F={rank['positive_forgetting']:.6f}, updates={rank['learner_updates']:.2f}")
    lines += ["", "## 训练日志诊断", "", "日志的策略梯度与熵梯度范数、概率/argmax 汇总、reward 与其分项、重要性比率、参数变化和固定 probe 汇总只作为优化诊断：梯度不为零不自动代表有效学习，reward 变化也不自动代表策略学会。`summary.json` 保存这些聚合数值字段的首末值、均值、最小值和最大值；逐 slot probe 原始记录保留在 training 文件，不扩写为摘要列。"]
    if not training:
        lines.append("\n本次运行没有可读取的训练日志；未绘制学习曲线，也不以占位数据替代。")
    else:
        for name, item in sorted(training.items()):
            lines += ["", f"### {name}", "", f"记录 {item['entries']} 条；关键聚合字段已写入 summary.json。"]
    (out / "研究诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def plot_training(training, run, out):
    """Plot the four predeclared round-level diagnostics, never per-slot logs."""
    fields = ("probe.mean_probabilities[2]", "probe.argmax_changed_fraction",
              "policy_gradient_grad_norm", "entropy_grad_norm", "reward_std")
    series = {field: [] for field in fields}
    directory = run / "training"
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        log = json.loads(path.read_text(encoding="utf-8")).get("log", [])
        numeric = defaultdict(list)
        for entry in log:
            if isinstance(entry, dict):
                for name, value in _flatten_numeric(entry): numeric[name].append(value)
        for field in fields:
            if len(numeric[field]) > 1: series[field].append((path.stem, numeric[field]))
    if not any(series.values()): return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    panels = (("probe.mean_probabilities[2]", "Probe multiplier-1 probability"),
              ("probe.argmax_changed_fraction", "Probe argmax changed fraction"),
              (None, "Gradient norms"), ("reward_std", "GRPO reward standard deviation"))
    for axis, (field, title) in zip(axes.flat, panels):
        names = ("policy_gradient_grad_norm", "entropy_grad_norm") if field is None else (field,)
        for name in names:
            for stem, values in series[name]: axis.plot(range(1, len(values) + 1), values, label=f"{stem}:{name}")
        axis.set_title(title); axis.set_xlabel("Training round"); axis.legend(fontsize=6)
    fig.savefig(out / "training_curves.png", dpi=170); plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run / "locked_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("development_only") is not True:
        raise ValueError("v11 analysis requires locked_protocol.json with development_only=true")
    out = args.out or args.run / "analysis"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run / "evaluations.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list): raise ValueError("evaluations.json must be a list")
    groups = aggregate_roots(rows); summary = summarize(groups)
    seed_groups = aggregate_policy_seeds(groups); selection = selection_reconstruction(seed_groups)
    paired_deltas = paired_checkpoint_deltas(seed_groups)
    training = training_diagnostics(args.run)
    total_cost_path = args.run / "total_cost.json"
    total_cost = json.loads(total_cost_path.read_text(encoding="utf-8")) if total_cost_path.exists() else None
    result = {"protocol": protocol, "groups": _json_groups(summary), "development_greedy_selection": selection,
              "policy_seed_pooled_root_values": _json_groups(summarize(seed_groups)),
              "paired_raw_root_deltas_vs_epoch0": paired_deltas,
              "training_diagnostics": training, "total_cost": total_cost,
              "unit_of_summary": "root; sample draws, then learned policy seeds, are averaged inside root before cross-root summaries",
              "interpretation": "development-only diagnostic; no confirmation, win, or non-inferiority claim"}
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, selection, training, protocol, out)
    plot_training(training, args.run, out)


if __name__ == "__main__": main()

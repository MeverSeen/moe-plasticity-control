"""Analysis for v8 multilayer retention runs.

The unit of inference is a root (initialisation/data stream), never a policy
replica.  This module deliberately keeps unsuccessful runs in reach, cost, and
all-root descriptive summaries.  Audit data are reported separately from dev
data and are never used to select a method here.
"""
import argparse
import itertools
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def interval(values, seed=8103, samples=4000):
    """Exploratory root-level bootstrap interval (not a formal test)."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (samples, len(values)))].mean(1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [.025, .975])]}


def _matrix(row, name, k):
    matrix = row.get(name)
    if not isinstance(matrix, list) or len(matrix) != k:
        raise ValueError(f"root={row.get('root')} has invalid {name}")
    if any(not isinstance(line, list) or len(line) != k for line in matrix):
        raise ValueError(f"root={row.get('root')} has non-square {name}")
    # A lower-triangular endpoint matrix is complete evidence for every task
    # learned so far.  Missing past entries must not be silently dropped from F.
    for learned in range(k):
        for tested in range(learned + 1):
            value = matrix[learned][tested]
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"root={row.get('root')} has non-finite {name}[{learned}][{tested}]")
    return matrix


def row_metrics(row):
    """Return one history's metrics, anchored to its original diagonal losses."""
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("each result needs a nonempty tasks list")
    k = len(tasks)
    loss = _matrix(row, "loss_matrix", k)
    accuracy = _matrix(row, "accuracy_matrix", k)
    target = row.get("target_loss")
    if target is None:
        raise ValueError("each result needs target_loss")
    diagonal = []
    for j in range(k):
        value = loss[j][j]
        if value is None:
            raise ValueError(f"root={row.get('root')} lacks diagonal loss for task {j}")
        diagonal.append(float(value))

    trajectory = []
    for learned in range(k):
        old = [float(loss[learned][j]) - diagonal[j] for j in range(learned)]
        # The diagonal is intentionally the fixed, task-local reference: it is
        # never replaced by a later, partly-forgotten checkpoint.
        trajectory.append({"after_task": learned,
                           # F is a mean over old tasks.  At k=0 there is no
                           # old task, so it is deliberately null rather than
                           # a numerical zero comparable with later points.
                           "positive_forgetting": _mean([max(0., x) for x in old]),
                           "signed_backward_change": _mean(old),
                           "old_tasks_scored": len(old)})
    final_loss = [loss[k - 1][j] for j in range(k)]
    final_accuracy = [accuracy[k - 1][j] for j in range(k)]
    final_deltas = [float(final_loss[j]) - diagonal[j] for j in range(k - 1)]
    successes = [bool(t.get("success", False)) for t in tasks]
    costs = row.get("cost", {})
    return {
        "root": row.get("root"), "method": row.get("method"), "split": row.get("split"),
        "policy_seed": row.get("policy_seed"), "k": k,
        "sequence_success": bool(all(successes)),
        "sequence_ever_reached_rate": float(np.mean(successes)),
        "failed_tasks": int(k - sum(successes)),
        "initially_at_target": int(sum(float(t.get("initial_loss", np.inf)) <= float(target) for t in tasks)),
        "final_all_at_target": bool(all(x is not None and float(x) <= float(target) for x in final_loss)),
        "final_mean_loss": _mean(final_loss), "final_mean_accuracy": _mean(final_accuracy),
        "final_positive_forgetting": _mean([max(0., x) for x in final_deltas]),
        "final_signed_backward_change": _mean(final_deltas),
        "trajectory": trajectory,
        "loss_matrix": [[None if x is None else float(x) for x in line] for line in loss],
        "cost": {key: float(costs.get(key, 0)) for key in
                 ["learner_updates", "train_examples", "monitor_examples", "audit_examples", "route_examples",
                  "controller_calls", "seconds", "audit_seconds"]},
    }


def aggregate_roots(rows):
    """Average policy replicas within (method, split, root), retaining failure."""
    buckets = defaultdict(list)
    for row in rows:
        m = row_metrics(row)
        if m["root"] is None or not m["method"] or not m["split"]:
            raise ValueError("result requires root, method, and split")
        buckets[(m["split"], m["method"], m["root"])].append(m)
    grouped = defaultdict(list)
    for (split, method, root), replicas in buckets.items():
        keys = ["sequence_success", "sequence_ever_reached_rate", "failed_tasks", "initially_at_target",
                "final_all_at_target", "final_mean_loss", "final_mean_accuracy",
                "final_positive_forgetting", "final_signed_backward_change"]
        item = {key: _mean([float(r[key]) for r in replicas]) for key in keys}
        item.update(split=split, method=method, root=root, replicas=len(replicas),
                    all_replicas_success=all(r["sequence_success"] for r in replicas))
        item["cost"] = {key: _mean([r["cost"][key] for r in replicas]) for key in replicas[0]["cost"]}
        # A per-step root mean supports figures without treating replicas as n.
        item["trajectory"] = [{key: _mean([r["trajectory"][i][key] for r in replicas])
                               for key in ["positive_forgetting", "signed_backward_change"]}
                              for i in range(replicas[0]["k"])]
        matrices = np.asarray([[[np.nan if x is None else x for x in line] for line in r["loss_matrix"]]
                               for r in replicas], dtype=float)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            matrix = np.nanmean(matrices, axis=0)
        item["loss_matrix"] = [[None if np.isnan(x) else float(x) for x in line] for line in matrix]
        grouped[(split, method)].append(item)
    return {key: sorted(value, key=lambda x: x["root"]) for key, value in grouped.items()}


def summarize_group(roots):
    fields = ["sequence_success", "sequence_ever_reached_rate", "failed_tasks", "initially_at_target",
              "final_all_at_target", "final_mean_loss", "final_mean_accuracy",
              "final_positive_forgetting", "final_signed_backward_change"]
    out = {field: interval([r[field] for r in roots]) for field in fields}
    out["roots"] = len(roots)
    out["replicas_total"] = int(sum(r["replicas"] for r in roots))
    out["cost"] = {key: interval([r["cost"][key] for r in roots]) for key in roots[0]["cost"]} if roots else {}
    return out


def compare_groups(a, b):
    """Paired root comparison; common set requires every policy replica succeeds."""
    by_a, by_b = ({r["root"]: r for r in group} for group in (a, b))
    common_roots = sorted(set(by_a) & set(by_b))
    def diffs(field, predicate=lambda x, y: True):
        return [by_a[root][field] - by_b[root][field] for root in common_roots
                if predicate(by_a[root], by_b[root])]
    successful = lambda x, y: x["all_replicas_success"] and y["all_replicas_success"]
    return {
        "paired_roots": len(common_roots),
        "common_all_replica_success_roots": sum(successful(by_a[r], by_b[r]) for r in common_roots),
        "final_positive_forgetting_common_success": interval(diffs("final_positive_forgetting", successful)),
        "final_mean_loss_gap_common_success": interval(diffs("final_mean_loss", successful)),
        "final_positive_forgetting_all_roots": interval(diffs("final_positive_forgetting")),
        "final_mean_loss_gap_all_roots": interval(diffs("final_mean_loss")),
        "final_accuracy_gap_all_roots": interval(diffs("final_mean_accuracy")),
        "reach_gap_all_roots": interval(diffs("sequence_success")),
    }


def validate_protocol_coverage(rows, protocol):
    """Reject incomplete selected-result grids when the protocol specifies one."""
    expected_seeds = protocol.get("expected_policy_seeds")
    expected_roots = protocol.get("roots")
    if expected_seeds is None or expected_roots is None:
        return
    if not isinstance(expected_seeds, dict) or not isinstance(expected_roots, dict):
        raise ValueError("protocol expected_policy_seeds and roots must be objects")
    actual = defaultdict(list)
    for row in rows:
        split, method = row.get("split"), row.get("method")
        actual[(split, method, row.get("root"))].append(row.get("policy_seed"))
    expected_methods = set(expected_seeds)
    for split, roots in expected_roots.items():
        if not isinstance(roots, list):
            raise ValueError(f"protocol roots[{split}] must be a list")
        for method, seeds in expected_seeds.items():
            if not isinstance(seeds, list):
                raise ValueError(f"protocol expected_policy_seeds[{method}] must be a list")
            wanted = list(seeds)
            if len(wanted) != len(set(map(repr, wanted))):
                raise ValueError(f"protocol has duplicate expected seeds for {method}")
            for root in roots:
                got = actual.pop((split, method, root), [])
                if len(got) != len(set(map(repr, got))) or set(map(repr, got)) != set(map(repr, wanted)):
                    raise ValueError(f"incomplete result grid: split={split} method={method} root={root}; expected seeds={wanted}, got={got}")
    if actual:
        unexpected = sorted((str(key), values) for key, values in actual.items())
        raise ValueError(f"results contain roots, splits, or methods outside protocol: {unexpected}")


def write_report(summary, protocol, out):
    status = protocol.get("status", "unknown")
    lines = ["# v8 多层 MoE 保留实验报告", "", f"协议状态：`{status}`。CIFAR-100 类隔离，5 个 5-way 任务；固定样本路由与固定总旧记忆由 protocol/config 记录。",
             "", "开发集（dev）可以用于选择；最终 audit 在模型冻结后以 official test 评测，不进入选择。下表的‘最终全任务达标’是最终 loss 均不高于 target_loss，和训练时的在线监测 success 不同。", ""]
    for split, groups in summary["groups"].items():
        lines += [f"## {split}", "", "| 方法 | 根数 | 整条序列达标率 | 逐任务达标率 | 最终全任务达标率 | 最终平均准确率 | 最终正遗忘 F | 失败任务数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, s in groups.items():
            get = lambda key: s[key]["mean"]
            pct = lambda key: "NA" if get(key) is None else f"{get(key):.1%}"
            num = lambda key: "NA" if get(key) is None else f"{get(key):.4f}"
            lines.append(f"| {method} | {s['roots']} | {pct('sequence_success')} | {pct('sequence_ever_reached_rate')} | {pct('final_all_at_target')} | {pct('final_mean_accuracy')} | {num('final_positive_forgetting')} | {num('failed_tasks')} |")
    lines += ["", "F 是每个任务刚学会时的固定对角 loss 为基准，在序列末对旧任务 loss 增量取正部分的均值（j < k）；基准不会在后续任务后重置。k=0 没有旧任务，F 与 signed backward change 记为 null，不能与后续检查点比较。所有 cost 与失败历史都已纳入根级汇总。", "", "## 配对比较", "", "同一 root 内先平均 policy_seed。共同成功集合要求两种方法该 root 的所有 replica 都完成整段序列；all-root F 同时报告，避免因筛选成功历史而隐藏失败。bootstrap 区间只作探索性描述，不构成显著胜利或非劣结论。"]
    if status == "pilot":
        lines += ["", "本次为 pilot：结果仅用于检查流程、方差和效应量，不作显著胜利或非劣宣称。"]
    (out / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def plot(groups, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entries = [(split, method, roots) for (split, method), roots in groups.items()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    labels = [f"{s}:{m}" for s, m, _ in entries]
    # One representative aggregate matrix makes the scoring convention visible:
    # rows are checkpoints after learning k and columns are independently tested tasks j.
    chosen = next((x for x in entries if x[0] == "audit"), entries[0]) if entries else None
    if chosen and chosen[2]:
        values = np.asarray([[[np.nan if v is None else v for v in line] for line in r["loss_matrix"]] for r in chosen[2]])
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            matrix = np.nanmean(values, axis=0)
        image = axes[0].imshow(matrix, cmap="viridis")
        axes[0].set_title(f"Mean loss matrix: {chosen[0]}:{chosen[1]}")
        axes[0].set_xlabel("Test task j"); axes[0].set_ylabel("After learning task k")
        fig.colorbar(image, ax=axes[0], fraction=.05)
    for split, method, roots in entries:
        if not roots: continue
        trajectory = roots[0]["trajectory"]
        values = [_mean([root["trajectory"][i]["positive_forgetting"] for root in roots])
                  for i in range(len(trajectory))]
        axes[1].plot(range(1, len(trajectory) + 1), values, marker="o", label=f"{split}:{method}")
    axes[1].set_xlabel("Task learned"); axes[1].set_ylabel("Mean positive forgetting F (k=0 NA)"); axes[1].legend(fontsize=7)
    positions = np.arange(len(entries))
    axes[2].bar(positions - .18, [_mean([x["sequence_success"] for x in r]) for _, _, r in entries], width=.36, label="Sequence success")
    axes[2].bar(positions + .18, [_mean([x["sequence_ever_reached_rate"] for x in r]) for _, _, r in entries], width=.36, label="Task success")
    axes[2].set_xticks(range(len(entries)), labels, rotation=35, ha="right"); axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Reached target rate"); axes[2].legend(fontsize=7)
    fig.savefig(out / "results.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True, type=Path); parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty analysis directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    protocol_path = args.run / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {"status": "unknown"}
    validate_protocol_coverage(rows, protocol)
    root_groups = aggregate_roots(rows)
    nested = defaultdict(dict)
    for (split, method), roots in root_groups.items(): nested[split][method] = summarize_group(roots)
    comparisons = {split: {f"{a}_minus_{b}": compare_groups(root_groups[(split, a)], root_groups[(split, b)])
                           for a, b in itertools.combinations(sorted(methods), 2)}
                   for split, methods in nested.items()}
    result = {"protocol": {key: protocol.get(key, "unknown") for key in ["status", "config", "selection"]},
              "groups": nested, "comparisons": comparisons,
              "statistics": "Exploratory 95% root-paired bootstrap CI; policy replicas averaged inside root. No formal non-inferiority test."}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(result, protocol, args.out); plot(root_groups, args.out)


if __name__ == "__main__": main()

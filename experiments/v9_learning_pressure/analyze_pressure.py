"""Analyse v9 learning-pressure rollouts without touching dataset files.

The result file is the sole input.  In particular, this module must never load
official test data: ``loss_matrix`` is already the held-out development
evaluation recorded by the runner.
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("global_0", "global_0.5", "global_1", "group_rule", "local_rule", "coordinated_rule")
REQUIRED_TASK_FIELDS = (
    "min_monitor_loss", "min_train_loss", "min_eval_loss", "initial_train_loss",
    "initial_eval_loss", "ever_hit", "first_hit_step", "first_hit_eval_loss", "target_loss",
)


def _number(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def _interval(values, seed=9141, samples=4000):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(v) for v in np.quantile(draws, [.025, .975])]}


def _matrix(row, field, k):
    matrix = row.get(field)
    if not isinstance(matrix, list) or len(matrix) != k or any(not isinstance(line, list) or len(line) != k for line in matrix):
        raise ValueError(f"root={row.get('root')} has invalid {field}")
    for learned in range(k):
        for tested in range(learned + 1):
            _number(matrix[learned][tested], f"root={row.get('root')} {field}[{learned}][{tested}]")
    return matrix


def _terminal_monitor(task):
    """The runner records the final monitor value under this explicit name.

    ``terminal_loss`` is accepted for compatibility with the v8-style row
    envelope, but is never confused with the held-out matrix diagonal.
    """
    value = task.get("terminal_monitor_loss", task.get("terminal_loss"))
    return _number(value, "task terminal_monitor_loss")


def row_metrics(row):
    """Validate a rollout and return fixed-denominator endpoint metrics."""
    for key in ("root", "method", "mode", "scope", "target_loss", "prefix_target"):
        if key not in row:
            raise ValueError(f"result needs {key}")
    if row["method"] not in METHODS:
        raise ValueError(f"unknown method {row['method']}")
    if row["mode"] not in ("early_stop", "fixed_budget") or row["scope"] not in ("full", "frozen_stem", "heads_only"):
        raise ValueError("invalid mode or scope")
    _number(row["target_loss"], "target_loss"); _number(row["prefix_target"], "prefix_target")
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 5:
        raise ValueError("v9 requires exactly five tasks")
    loss = _matrix(row, "loss_matrix", len(tasks))
    accuracy = _matrix(row, "accuracy_matrix", len(tasks))
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} is not an object")
        for field in REQUIRED_TASK_FIELDS:
            if field not in task:
                raise ValueError(f"task {index} lacks {field}")
        for field in REQUIRED_TASK_FIELDS:
            if field not in ("ever_hit", "first_hit_step", "first_hit_eval_loss"):
                _number(task[field], f"task {index} {field}")
        if not isinstance(task["ever_hit"], bool):
            raise ValueError(f"task {index} ever_hit must be bool")
        hit_step = task["first_hit_step"]
        if hit_step is not None and (not isinstance(hit_step, int) or isinstance(hit_step, bool) or hit_step < 0):
            raise ValueError(f"task {index} first_hit_step must be nonnegative int or null")
        if task["ever_hit"] != (hit_step is not None):
            raise ValueError(f"task {index} ever_hit and first_hit_step disagree")
        first_eval = task["first_hit_eval_loss"]
        if task["ever_hit"]:
            _number(first_eval, f"task {index} first_hit_eval_loss")
        elif first_eval is not None:
            raise ValueError(f"task {index} first_hit_eval_loss must be null when never hit")
        _terminal_monitor(task)
    diagonal = [float(loss[i][i]) for i in range(5)]
    targets = [float(task["target_loss"]) for task in tasks]
    # Task zero is a shared prefix and has a separate target by protocol.
    if not math.isclose(targets[0], float(row["prefix_target"]), abs_tol=1e-9):
        raise ValueError("task 0 target_loss must equal prefix_target")
    if any(not math.isclose(x, float(row["target_loss"]), abs_tol=1e-9) for x in targets[1:]):
        raise ValueError("suffix task target_loss must equal row target_loss")
    suffix = tasks[1:]
    suffix_diag = diagonal[1:]
    suffix_first_hit_heldout = [bool(t["ever_hit"] and float(t["first_hit_eval_loss"]) <= float(t["target_loss"]))
                                for t in suffix]
    monitor_hit_eval_miss = [bool(t["ever_hit"] and float(t["first_hit_eval_loss"]) > float(t["target_loss"]))
                             for t in suffix]
    terminal_gaps = []
    for task in suffix:
        if "terminal_train_loss" in task and "terminal_eval_loss" in task:
            terminal_gaps.append(_number(task["terminal_eval_loss"], "terminal_eval_loss") -
                                 _number(task["terminal_train_loss"], "terminal_train_loss"))
    final_old_changes = [float(loss[4][j]) - diagonal[j] for j in range(4)]
    raw_cost = row.get("cost", {})
    if not isinstance(raw_cost, dict):
        raise ValueError("cost must be an object")
    # Costs are raw pass/call/time counters.  Keep every declared field and
    # sum it later; silently dropping a counter would make a cost comparison
    # look cheaper than the runner reported.
    cost = {str(k): _number(v, f"cost.{k}") for k, v in raw_cost.items()}
    return {
        "root": row["root"], "method": row["method"], "mode": row["mode"], "scope": row["scope"],
        "target_loss": float(row["target_loss"]), "prefix_target": float(row["prefix_target"]),
        "prefix_monitor_ever_hit": tasks[0]["ever_hit"],
        "prefix_heldout_eval_hit": diagonal[0] <= targets[0],
        "suffix_monitor_ever_all": bool(all(t["ever_hit"] for t in suffix)),
        "suffix_monitor_ever_rate": float(np.mean([t["ever_hit"] for t in suffix])),
        "suffix_monitor_terminal_all": bool(all(_terminal_monitor(t) <= float(t["target_loss"]) for t in suffix)),
        # At the instant a monitor crossing chose to stop, use the separately
        # recorded held-out evaluation.  Four suffix tasks are always the
        # denominator: never-hit tasks count as false rather than disappearing.
        "suffix_first_hit_heldout_all": bool(all(suffix_first_hit_heldout)),
        "suffix_first_hit_heldout_rate": float(np.mean(suffix_first_hit_heldout)),
        "monitor_hit_eval_miss_rate": float(np.mean(monitor_hit_eval_miss)),
        "suffix_heldout_eval_diagonal_all": bool(all(value <= target for value, target in zip(suffix_diag, targets[1:]))),
        "suffix_heldout_eval_diagonal_rate": float(np.mean([value <= target for value, target in zip(suffix_diag, targets[1:])])),
        "final_mean_accuracy": _mean([float(accuracy[4][i]) for i in range(5)]),
        # This is intentionally diagonal accuracy, measured just after each
        # task was learned, not final accuracy after the whole sequence.
        "suffix_mean_accuracy": _mean([float(accuracy[i][i]) for i in range(1, 5)]),
        "suffix_mean_eval_loss": _mean(suffix_diag),
        "suffix_terminal_train_eval_gap": _mean(terminal_gaps),
        "final_positive_forgetting": _mean([max(0., x) for x in final_old_changes]),
        "final_signed_backward_change": _mean(final_old_changes),
        "loss_matrix": [[None if v is None else float(v) for v in line] for line in loss],
        "cost": cost,
    }


def _validate_rollout_path(row, protocol):
    """Check the pre-registered stopping semantics from runner records."""
    config = protocol.get("config", {})
    if not isinstance(config, dict) or "max_steps" not in config or "interval" not in config:
        return
    max_steps = config["max_steps"]; interval = config["interval"]
    if not isinstance(max_steps, int) or not isinstance(interval, int) or max_steps <= 0 or interval <= 0:
        raise ValueError("protocol.config max_steps and interval must be positive integers")
    for index, task in enumerate(row.get("tasks", [])[1:], start=1):
        if "steps" not in task or "success" not in task:
            raise ValueError(f"task {index} lacks steps or success")
        steps = task["steps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError(f"task {index} steps must be a nonnegative int")
        if row.get("mode") == "fixed_budget":
            if steps != max_steps:
                raise ValueError(f"fixed_budget task {index} must use steps={max_steps}")
            continue
        if row.get("mode") != "early_stop":
            continue
        ever = task.get("ever_hit"); first = task.get("first_hit_step")
        if ever:
            if steps != first or steps % interval != 0:
                raise ValueError(f"early_stop hit task {index} must stop at first_hit_step on interval")
            if not bool(task.get("success")) or _terminal_monitor(task) > float(task.get("target_loss")):
                raise ValueError(f"early_stop hit task {index} must terminate monitor-at-target with success")
        elif steps != max_steps or bool(task.get("success")):
            raise ValueError(f"early_stop miss task {index} must exhaust budget with success=false")


def _validate_prefix_fingerprints(rows):
    by_root = {}
    for row in rows:
        if "prefix_fingerprint" not in row or row["prefix_fingerprint"] in (None, ""):
            raise ValueError(f"root={row.get('root')} lacks prefix_fingerprint")
        root, fingerprint = row.get("root"), row["prefix_fingerprint"]
        if root in by_root and by_root[root] != fingerprint:
            raise ValueError(f"root={root} has inconsistent prefix_fingerprint")
        by_root[root] = fingerprint


def validate_protocol_coverage(rows, protocol):
    """Require exactly one independent rollout per expected cell/root pair."""
    cells, roots = protocol.get("expected_cells"), protocol.get("roots")
    if cells is None or roots is None:
        return
    if not isinstance(cells, list) or not isinstance(roots, list):
        raise ValueError("protocol.expected_cells and protocol.roots must be lists")
    expected = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {"mode", "scope", "method", "target_loss"}:
            raise ValueError("each expected_cells entry needs exactly mode, scope, method, target_loss")
        key = (cell["mode"], cell["scope"], cell["method"], float(cell["target_loss"]))
        if key in expected: raise ValueError(f"duplicate expected cell: {key}")
        expected.add(key)
    if len(roots) != len(set(map(repr, roots))):
        raise ValueError("protocol has duplicate roots")
    actual = defaultdict(int)
    for row in rows:
        key = (row.get("mode"), row.get("scope"), row.get("method"), float(row.get("target_loss", float("nan"))), row.get("root"))
        actual[key] += 1
    wanted = {(mode, scope, method, target, root) for mode, scope, method, target in expected for root in roots}
    missing = sorted(wanted - set(actual), key=repr)
    duplicate = sorted((key, count) for key, count in actual.items() if count != 1)
    unexpected = sorted(set(actual) - wanted, key=repr)
    if missing or duplicate or unexpected:
        raise ValueError(f"incomplete result grid: missing={missing}; duplicate={duplicate}; unexpected={unexpected}")
    _validate_prefix_fingerprints(rows)
    for row in rows:
        _validate_rollout_path(row, protocol)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        metric = row_metrics(row)
        key = (metric["mode"], metric["scope"], metric["method"], metric["target_loss"])
        groups[key].append(metric)
    result = {}
    metric_fields = ("prefix_monitor_ever_hit", "prefix_heldout_eval_hit", "suffix_monitor_ever_all",
                     "suffix_monitor_ever_rate", "suffix_monitor_terminal_all", "suffix_first_hit_heldout_all",
                     "suffix_first_hit_heldout_rate", "monitor_hit_eval_miss_rate", "suffix_heldout_eval_diagonal_all",
                     "suffix_heldout_eval_diagonal_rate", "suffix_mean_accuracy", "suffix_mean_eval_loss",
                     "final_mean_accuracy", "suffix_terminal_train_eval_gap", "final_positive_forgetting",
                     "final_signed_backward_change")
    for key, values in groups.items():
        if len({repr(x["root"]) for x in values}) != len(values):
            raise ValueError(f"duplicate root within cell {key}")
        totals = defaultdict(float)
        for value in values:
            for name, amount in value["cost"].items(): totals[name] += amount
        result[key] = {"roots": len(values), "root_ids": sorted([x["root"] for x in values]),
                       "metrics": {name: _interval([None if x[name] is None else float(x[name]) for x in values]) for name in metric_fields},
                       "cost_sum": dict(sorted(totals.items()))}
    return result, groups


def paired_differences(groups):
    """Descriptive same-root gaps, grouped only within identical conditions."""
    out = {}
    by_context = defaultdict(list)
    for mode, scope, method, target in groups:
        by_context[(mode, scope, target)].append(method)
    for context, methods in by_context.items():
        for left, right in itertools.combinations(sorted(methods), 2):
            a = {x["root"]: x for x in groups[(context[0], context[1], left, context[2])]}
            b = {x["root"]: x for x in groups[(context[0], context[1], right, context[2])]}
            common = sorted(set(a) & set(b))
            out[f"{context[0]}:{context[1]}:{context[2]}:{left}_minus_{right}"] = {
                "paired_roots": len(common),
                "final_positive_forgetting": _interval([a[r]["final_positive_forgetting"] - b[r]["final_positive_forgetting"] for r in common]),
                "suffix_mean_accuracy": _interval([a[r]["suffix_mean_accuracy"] - b[r]["suffix_mean_accuracy"] for r in common]),
                "suffix_heldout_eval_diagonal_all": _interval([float(a[r]["suffix_heldout_eval_diagonal_all"]) - float(b[r]["suffix_heldout_eval_diagonal_all"]) for r in common]),
            }
    return out


def _serialisable_groups(summary):
    return {"|".join(map(str, key)): value for key, value in sorted(summary.items(), key=lambda x: repr(x[0]))}


def write_report(summary, protocol, out):
    root_count = len(protocol.get("roots", [])) if isinstance(protocol.get("roots"), list) else "若干"
    lines = ["# v9 学习压力实验报告", "", f"本报告只读取 runner 已记录的开发集结果；不加载 official test。{root_count} 个开发 root 使用同一组 25 类，root 是初始化/数据流重复，不是独立类别总体。", "",
             "`monitor ever` 表示训练期间曾在当前任务在线监测集上越过目标；`monitor terminal` 表示结束时监测值仍越线；`held-out eval` 则是独立开发评测的对角线。三者不能互相替代。task 0 是共同 prefix，单独报告；suffix 是 tasks 1–4。", "",
             "| 条件 | root | prefix monitor-ever | suffix monitor-ever率 | hit时 held-out全达标 | monitor-hit/eval-miss率 | suffix 末端 held-out全达标 | 对角平均准确率 | 最终平均准确率 | F |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, group in sorted(summary.items(), key=lambda x: repr(x[0])):
        mode, scope, method, target = key; metrics = group["metrics"]
        get = lambda name: metrics[name]["mean"]
        pct = lambda name: "NA" if get(name) is None else f"{get(name):.1%}"
        num = lambda name: "NA" if get(name) is None else f"{get(name):.4f}"
        lines.append(f"| {mode}/{scope}/{method}/t={target:g} | {group['roots']} | {pct('prefix_monitor_ever_hit')} | {pct('suffix_monitor_ever_rate')} | {pct('suffix_first_hit_heldout_all')} | {pct('monitor_hit_eval_miss_rate')} | {pct('suffix_heldout_eval_diagonal_all')} | {pct('suffix_mean_accuracy')} | {pct('final_mean_accuracy')} | {num('final_positive_forgetting')} |")
    lines += ["", "F 是在 suffix 最后一个端点，对旧 tasks 0–3 相对各自首次学习时的 held-out CE 对角线的正增量均值；带符号后向变化使用同一增量但不截断。即使 suffix 没有达标，仍计入所有汇总。", "",
              "若训练/监测损失低而 held-out eval 较高，结果与过拟合相容，但本实验的观测不能单独证明因果。fixed_budget 与 early_stop 不是同一训练终点；fixed_budget 的 1.35 条件来自独立 rollout，不能从别的策略轨迹截取。配对 root 差仅为描述性结果，不作正式胜利或非劣（NI）声明。"]
    (out / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [(key, value) for key, value in summary.items() if key[0] == "early_stop" and key[1] == "full"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    targets = sorted({key[3] for key, _ in primary}, reverse=True)
    methods = [m for m in METHODS if any(key[2] == m for key, _ in primary)]
    for axis, field, title, ylabel in zip(axes, ("suffix_heldout_eval_diagonal_all", "suffix_mean_accuracy", "final_positive_forgetting"),
                                          ("Suffix held-out all-target rate", "Suffix mean diagonal accuracy (at task learned)", "Final positive forgetting F"),
                                          ("Rate", "Accuracy", "CE increase")):
        for method in methods:
            points = {key[3]: value["metrics"][field]["mean"] for key, value in primary if key[2] == method}
            axis.plot(targets, [points.get(t, np.nan) for t in targets], marker="o", label=method)
        axis.set_title(title); axis.set_xlabel("Suffix target loss"); axis.set_ylabel(ylabel); axis.invert_xaxis()
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1); axes[2].legend(fontsize=7, ncol=2)
    fig.savefig(out / "pressure.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty analysis directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list): raise ValueError("results.json must contain a list")
    protocol_path = args.run / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    validate_protocol_coverage(rows, protocol)
    summary, groups = aggregate(rows)
    result = {"protocol": {key: protocol.get(key) for key in ("status", "roots", "expected_cells", "class_partition", "config")},
              "groups": _serialisable_groups(summary), "paired_root_descriptive_differences": paired_differences(groups),
              "statistics": "Root-level exploratory bootstrap descriptions only; no formal win or non-inferiority conclusion."}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, protocol, args.out); plot(summary, args.out)


if __name__ == "__main__": main()

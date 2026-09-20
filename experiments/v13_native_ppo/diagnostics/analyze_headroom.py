"""Read-only, CPU-only developmental candidate diagnostics; no learner execution.

Native v13 JSON is converted conservatively. No missing counterfactual is inferred.
Use --help and README.md for the input contract and epistemic limits.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


ALLOWED = {"calibration", "development"}
KINDS = {"fixed", "rule", "clone", "ppo", "intervention", "unspecified"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate(bundle):
    if bundle.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for field in ("protocol_id", "metric_id"):
        if not isinstance(bundle.get(field), str) or not bundle[field]:
            raise ValueError("missing metadata: " + field)
    rows = bundle.get("records")
    if not isinstance(rows, list):
        raise ValueError("records must be a list")
    keys, roots, contexts, specs, streams = set(), {}, {}, {}, {}
    for row in rows:
        for field in ("root_id", "comparison_id", "checkpoint_id", "context_id",
                      "stream_id", "candidate_id", "role"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("missing/non-string record field: " + field)
        if row["role"] not in ALLOWED:
            raise ValueError("only calibration/development records permitted")
        if row.get("candidate_kind", "unspecified") not in KINDS:
            raise ValueError("unknown candidate_kind")
        if not isinstance(row.get("candidate_spec"), dict):
            raise ValueError("candidate_spec must explicitly identify the policy/config")
        for field in ("completed", "success", "independent_future"):
            if type(row.get(field)) is not bool:
                raise ValueError(field + " must be an explicit boolean")
        failures = row.get("n_fail")
        if type(failures) is not int or failures < 0:
            raise ValueError("n_fail must be a nonnegative integer")
        if row["success"] and (not row["completed"] or failures != 0):
            raise ValueError("successful history must be completed with zero failures")
        if row["completed"] and not row["success"] and failures == 0:
            raise ValueError("completed unsuccessful history must record failures")
        for field in ("monitor_forgetting", "eval_forgetting", "path_updates", "actual_updates", "wall_seconds"):
            value = row.get(field)
            if value is not None and not number(value):
                raise ValueError("non-finite/non-numeric field: " + field)
            if value is not None and field.endswith("updates") and (value < 0 or int(value) != value):
                raise ValueError("update counts must be nonnegative integers")
            if value is not None and field == "wall_seconds" and value < 0:
                raise ValueError("wall_seconds must be nonnegative")
        previous = roots.setdefault(row["root_id"], row["role"])
        if previous != row["role"]:
            raise ValueError("one root crosses development/calibration roles")
        state = (row["root_id"], row["comparison_id"])
        context = (row["checkpoint_id"], row["context_id"])
        if contexts.setdefault(state, context) != context:
            raise ValueError("incompatible checkpoint/context within comparison")
        spec = canonical(row["candidate_spec"])
        if specs.setdefault(row["candidate_id"], spec) != spec:
            raise ValueError("candidate_id denotes different policy specifications")
        stream_key = state + (row["stream_id"],)
        if streams.setdefault(stream_key, row["independent_future"]) != row["independent_future"]:
            raise ValueError("inconsistent independent_future declaration")
        key = stream_key + (row["candidate_id"],)
        if key in keys:
            raise ValueError("duplicate comparison/stream/candidate; use distinct IDs for replicates")
        keys.add(key)
    if len(set(roots.values())) > 1:
        raise ValueError("analyze calibration and development pools separately")
    return rows


def root_summary(pairs):
    by_root = collections.defaultdict(list)
    for root, value in pairs:
        by_root[root].append(value)
    means = {root: statistics.mean(values) for root, values in sorted(by_root.items())}
    return {"n_observations": len(pairs), "n_roots": len(means),
            "mean_of_root_means": statistics.mean(means.values()) if means else None,
            "per_root_means": means}


def paired(left, right):
    """left/right are equally keyed, matched start/stream outcomes; retain failures."""
    keys = sorted(set(left) & set(right))
    common = [k for k in keys if left[k]["success"] and right[k]["success"]]
    def difference(field):
        return root_summary([(left[k]["root_id"], left[k][field] - right[k][field])
                             for k in common if left[k].get(field) is not None and right[k].get(field) is not None])
    return {"n_matched": len(keys), "n_roots": len({left[k]["root_id"] for k in keys}),
            "left_only": len(set(left) - set(right)), "right_only": len(set(right) - set(left)),
            "left_successes": sum(left[k]["success"] for k in keys),
            "right_successes": sum(right[k]["success"] for k in keys),
            "left_unfinished": sum(not left[k]["completed"] for k in keys),
            "right_unfinished": sum(not right[k]["completed"] for k in keys),
            "common_successes": len(common),
            "common_success_eval_delta": difference("eval_forgetting"),
            "common_success_path_updates_delta": difference("path_updates"),
            "sign": "left minus reference; lower forgetting is better, conditional on common success"}


def pick(rows, field, tolerance):
    feasible = [r for r in rows if r["success"] and r.get(field) is not None]
    if not feasible:
        return None, set()
    value = min(r[field] for r in feasible)
    tied = [r for r in feasible if r[field] <= value + tolerance]
    choice = min(tied, key=lambda r: (r.get("path_updates") if r.get("path_updates") is not None else math.inf,
                                     r["candidate_id"]))
    return choice, {r["candidate_id"] for r in tied}


def analyze(bundle, reference, select_stream=None, evaluate_stream=None, tolerance=1e-12):
    if not number(tolerance) or tolerance < 0:
        raise ValueError("tie tolerance must be finite and nonnegative")
    if (select_stream is None) != (evaluate_stream is None):
        raise ValueError("supply both stream IDs or neither")
    if select_stream is not None and select_stream == evaluate_stream:
        raise ValueError("selection and evaluation stream must differ")
    rows = validate(bundle)
    states, candidates = collections.defaultdict(dict), collections.defaultdict(dict)
    for row in rows:
        state = (row["root_id"], row["comparison_id"], row["stream_id"])
        states[state][row["candidate_id"]] = row
        candidates[row["candidate_id"]][state] = row
    if rows and reference not in candidates:
        raise ValueError("explicit reference candidate is absent; available: " + ", ".join(sorted(candidates)))
    coverage = {name: {"records": len(items), "roots": len({r["root_id"] for r in items.values()}),
                       "successes": sum(r["success"] for r in items.values()),
                       "unfinished": sum(not r["completed"] for r in items.values()),
                       "missing_eval": sum(r.get("eval_forgetting") is None for r in items.values()),
                       "missing_monitor": sum(r.get("monitor_forgetting") is None for r in items.values()),
                       "actual_updates": sum(r.get("actual_updates") or 0 for r in items.values()),
                       "missing_actual_updates": sum(r.get("actual_updates") is None for r in items.values())}
                for name, items in sorted(candidates.items())}
    spread, gains, concordance, choices, minimizers = [], [], [], collections.Counter(), []
    eligible = []
    for state, panel in sorted(states.items()):
        root = state[0]
        valid = [r for r in panel.values() if r["success"] and r.get("eval_forgetting") is not None]
        if len(valid) >= 2:
            spread.append((root, max(r["eval_forgetting"] for r in valid) - min(r["eval_forgetting"] for r in valid)))
        best, tied = pick(list(panel.values()), "eval_forgetting", tolerance)
        if best:
            choices[best["candidate_id"]] += 1
            minimizers.append(tied)
            eligible.append(set(panel))
            ref = panel.get(reference)
            if ref and ref["success"] and ref.get("eval_forgetting") is not None:
                gains.append((root, best["eval_forgetting"] - ref["eval_forgetting"]))
        agree, disagree, ties = 0, 0, 0
        for a, b in itertools.combinations(valid, 2):
            if a.get("monitor_forgetting") is None or b.get("monitor_forgetting") is None:
                continue
            dm, de = a["monitor_forgetting"] - b["monitor_forgetting"], a["eval_forgetting"] - b["eval_forgetting"]
            if abs(dm) <= tolerance or abs(de) <= tolerance:
                ties += 1
            elif dm * de > 0:
                agree += 1
            else:
                disagree += 1
        concordance.append({"state": list(state), "agree": agree, "disagree": disagree, "ties": ties})
    same_menu = bool(eligible) and all(menu == eligible[0] for menu in eligible)
    common_minimizers = sorted(set.intersection(*minimizers)) if same_menu else None
    future = {"status": "unknown_no_requested_independent_future", "reason": "No independent rollout is generated by this tool."}
    if select_stream is not None:
        selected, privileged, references = {}, {}, {}
        skipped, used = collections.Counter(), []
        groups = sorted({key[:2] for key in states})
        for group in groups:
            first, second = states.get(group + (select_stream,)), states.get(group + (evaluate_stream,))
            if not first or not second:
                skipped["missing_stream"] += 1
                continue
            if set(first) != set(second) or reference not in first:
                skipped["incomplete_candidate_panel_or_reference"] += 1
                continue
            if not all(r["independent_future"] for r in second.values()):
                skipped["future_independence_not_declared"] += 1
                continue
            monitor, _ = pick(list(first.values()), "monitor_forgetting", tolerance)
            hindsight, _ = pick(list(first.values()), "eval_forgetting", tolerance)
            references[group] = second[reference]
            if monitor:
                selected[group] = second[monitor["candidate_id"]]
            else:
                skipped["no_feasible_monitor_selection"] += 1
            if hindsight:
                privileged[group] = second[hindsight["candidate_id"]]
            else:
                skipped["no_feasible_privileged_selection"] += 1
            used.append({"root_id": group[0], "comparison_id": group[1],
                         "monitor_choice": monitor["candidate_id"] if monitor else None,
                         "privileged_choice": hindsight["candidate_id"] if hindsight else None})
        future = {"status": "available" if references else "unknown_no_eligible_independent_future",
                  "selection_stream": select_stream, "evaluation_stream": evaluate_stream,
                  "n_requested_comparisons": len(groups), "skipped": dict(skipped),
                  "monitor_selection": paired(selected, references),
                  "privileged_selection_diagnostic_only": paired(privileged, references), "choices": used,
                  "limit": "Independent future batches, not independent tasks or evaluation examples; no population-optimality claim."}
    return {"schema_version": 1, "protocol_id": bundle["protocol_id"], "metric_id": bundle["metric_id"],
            "reference": reference, "tie_tolerance": tolerance, "n_records": len(rows),
            "coverage": coverage,
            "candidate_vs_reference": {name: paired(items, candidates[reference]) for name, items in sorted(candidates.items()) if name != reference},
            "within_stream_descriptive": {"n_states": len(states), "n_states_with_feasible_eval": len(minimizers),
                "feasible_eval_range": root_summary(spread),
                "hindsight_eval_delta_vs_reference_optimistic": root_summary(gains),
                "hindsight_choice_counts_with_cost_tiebreak": dict(choices),
                "common_minimum_forgetting_candidates": common_minimizers,
                "common_minimizers_status": "descriptive_same_menu" if same_menu else "unknown_missing_or_different_menus",
                "rank_pairs_by_state": concordance,
                "rank_agreement": root_summary([(x["state"][0], x["agree"] / (x["agree"] + x["disagree"]))
                                                for x in concordance if x["agree"] + x["disagree"]])},
            "independent_future": future,
            "limits": ["Exploratory calibration/development evidence only; not final confirmation.",
                       "Only actually executed candidates; no theoretical bound or global optimum.",
                       "Repeated checkpoints/streams are grouped by root; no significance test is produced.",
                       "Failures are retained; forgetting differences condition on common success with observed metrics.",
                       "Candidate differences do not isolate expert granularity, feedback or PPO mechanisms.",
                       "Numeric tie tolerance is not a scientific equivalence/non-inferiority margin."]}


def native_bundle(paths):
    """Conservative adapter for the executor's explicitly documented endpoint schema."""
    records, audit, protocols = [], [], set()
    for path in paths:
        obj = json.loads(Path(path).read_text())
        if not isinstance(obj, dict) or "protocol_revision" not in obj or "root" not in obj:
            audit.append({"path": str(path), "status": "skipped_not_endpoint"})
            continue
        root = obj["root"]
        if obj.get("reference_calibration") is True:
            audit.append({"path": str(path), "status": "excluded_target_generation_reference", "source_phase": obj.get("phase")})
            continue
        role = root.get("split", obj.get("phase"))
        if role not in ALLOWED:
            audit.append({"path": str(path), "status": "excluded_role", "role": role})
            continue
        required = ("history_completed", "history_success", "n_fail")
        missing = [key for key in required if key not in obj]
        checkpoint, stream, candidate = obj.get("checkpoint", {}), obj.get("batch_stream", {}), obj.get("candidate", {})
        if not checkpoint.get("prefix_sha256") or not checkpoint.get("base_revision"):
            missing.append("complete-prefix/base identity")
        if "id" not in stream or type(stream.get("independent_future")) is not bool:
            missing.append("explicit batch stream identity/independence")
        if not candidate.get("name") or root.get("id") is None:
            missing.append("root/candidate identity")
        tasks = obj.get("tasks", [])
        targets = obj.get("target_spec")
        if targets is None:
            if [x.get("task") for x in tasks] == root.get("tasks") and all("target" in x for x in tasks):
                targets = [{"task": x["task"], "target": x["target"]} for x in tasks]
            else:
                missing.append("complete target_spec (including unattempted tasks)")
        if missing:
            audit.append({"path": str(path), "status": "unusable_missing_fields", "missing": missing})
            continue
        protocols.add(canonical(obj["protocol_revision"]))
        actual, cost_path = obj.get("cost_actual", {}), obj.get("cost_path", {})
        monitor, evaluation = obj.get("monitor", {}), obj.get("evaluation", {})
        spec = {"candidate": candidate, "controller_sha256": checkpoint.get("controller_sha256")}
        # Policy checkpoint distinguishes learned rounds with the same display name.
        identity = str(candidate["name"])
        if checkpoint.get("controller_sha256"):
            identity += "@" + str(checkpoint["controller_sha256"])[:16]
        records.append({"root_id": str(root["id"]), "comparison_id": obj.get("comparison_id", "whole_history"),
                        "checkpoint_id": checkpoint["prefix_sha256"],
                        "context_id": digest({"base": checkpoint["base_revision"], "tasks": root["tasks"],
                                              "targets": targets, "protocol": obj["protocol_revision"]}),
                        "stream_id": str(stream["id"]), "independent_future": stream["independent_future"],
                        "role": role, "candidate_id": identity, "candidate_kind": candidate.get("kind", "unspecified"),
                        "candidate_spec": spec, "completed": obj["history_completed"], "success": obj["history_success"],
                        "n_fail": obj["n_fail"], "monitor_forgetting": monitor.get("forgetting") if monitor.get("full_memory") is True else None,
                        "eval_forgetting": evaluation.get("forgetting"), "path_updates": cost_path.get("updates"),
                        "actual_updates": actual.get("updates"), "wall_seconds": actual.get("wall_seconds"),
                        "source": str(path), "source_phase": obj.get("phase")})
        audit.append({"path": str(path), "status": "accepted", "source_phase": obj.get("phase")})
    if len(protocols) > 1:
        raise ValueError("mixed protocol revisions; analyze separately")
    return {"schema_version": 1, "protocol_id": next(iter(protocols), "no_usable_protocol"),
            "metric_id": "native_v13.evaluation.forgetting", "records": records, "adapter_audit": audit}


def render(result):
    lines = ["# v13 收益空间诊断（开发数据）", "", "这是已有候选的描述性分析，不是原生 MoE 的理论上界或最终确认。", "",
             "输入记录：%d；参考候选：`%s`。" % (result["n_records"], result["reference"]), "",
             "| 候选 | 记录 / 根 | 达标 | 未完成 | 缺评估 | 缺监测 |", "|---|---:|---:|---:|---:|---:|"]
    if result.get("source_phases") == ["pilot"]:
        lines[2] = "输入全部来自 pilot 工程/开发短测；未执行正式校准、克隆、PPO 或确认。指定参考只是 pilot 的均匀更新，不是开发期选出的强静态参考。"
    for name, row in result["coverage"].items():
        lines.append("| %s | %s / %s | %s | %s | %s | %s |" % (name.replace("|", "\\|"), row["records"], row["roots"], row["successes"], row["unfinished"], row["missing_eval"], row["missing_monitor"]))
    lines += ["", "## 与指定参考配对（差值为候选减参考）", "",
              "| 候选 | 配对数 | 候选 / 参考达标 | 共同达标 | 遗忘差（根均值） | 更新差（根均值） |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["candidate_vs_reference"].items():
        lines.append("| %s | %d | %d / %d | %d | %s | %s |" % (name.replace("|", "\\|"), row["n_matched"], row["left_successes"], row["right_successes"], row["common_successes"], row["common_success_eval_delta"]["mean_of_root_means"], row["common_success_path_updates_delta"]["mean_of_root_means"]))
    within = result["within_stream_descriptive"]
    lines += ["", "同流可行候选遗忘范围（根均值）：`%s`。" % within["feasible_eval_range"]["mean_of_root_means"],
              "跨已测状态共同的最低遗忘候选：`%s`；状态：`%s`。这忽略成本取舍，仅是数值容差内的描述。" % (within["common_minimum_forgetting_candidates"], within["common_minimizers_status"]),
              "", "## 独立未来复测", "", "状态：`%s`。" % result["independent_future"]["status"],
              "", "```json", json.dumps(result["independent_future"], ensure_ascii=False, indent=2), "```", "",
              "## 解释边界", ""]
    lines += ["- " + x for x in result["limits"]]
    lines += ["", "缺失输入、根内明细、监测排序、乐观样本内选择和原生字段适配审计见 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="canonical JSON bundle")
    source.add_argument("--native-dir", type=Path, help="read existing native v13 endpoint JSON recursively")
    parser.add_argument("--reference", required=True, help="explicit preselected candidate ID; never auto-picked")
    parser.add_argument("--select-stream")
    parser.add_argument("--evaluate-stream")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.native_dir and (args.output_dir.resolve() == args.native_dir.resolve() or
                            args.native_dir.resolve() in args.output_dir.resolve().parents):
        raise ValueError("output directory must be outside the recursively scanned input directory")
    if args.input:
        bundle = json.loads(args.input.read_text())
        provenance = {"input_path": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    else:
        files = sorted(args.native_dir.rglob("*.json"))
        bundle = native_bundle(files)
        provenance = {"native_dir": str(args.native_dir.resolve()), "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, "adapter_audit": bundle["adapter_audit"]}
    result = analyze(bundle, args.reference, args.select_stream, args.evaluate_stream, args.tie_tolerance)
    result["source_phases"] = sorted({row["source_phase"] for row in bundle["records"] if row.get("source_phase")})
    result["provenance"] = provenance
    if not result["n_records"]:
        result["status"] = "no_usable_records_no_scientific_result"
    else:
        result["status"] = "descriptive_pilot_engineering_analysis" if result["source_phases"] == ["pilot"] else "descriptive_development_analysis"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("headroom.json", "收益空间诊断.md"):
        if (args.output_dir / name).exists():
            raise ValueError("refusing to overwrite existing analysis: " + name)
    (args.output_dir / "headroom.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "收益空间诊断.md").write_text(render(result))
    print(json.dumps({"status": result["status"], "records": result["n_records"], "independent_future": result["independent_future"]["status"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

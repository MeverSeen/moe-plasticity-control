"""Root-aware, stream-cross-fitted analysis of v12 paired action-value probes.

This is an exploratory *decision-necessity* diagnostic. In particular, the
per-state oracle uses privileged stream-0 counterfactual outcomes; it is NOT
an implementable state-feedback policy or evidence that RL is useful.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def metrics(row):
    """Held-out CE changes in one equal-length action intervention."""
    before, after = row["pre_old_ce"], row["post_old_ce"]
    if not isinstance(before, list) or len(before) != len(after) or not before:
        raise ValueError("old task CE vectors missing or mismatched")
    gain = float(row["pre_new_ce"] - row["post_new_ce"])
    harm = sum(max(0.0, float(b - a)) for a, b in zip(before, after)) / len(before)
    if not math.isfinite(gain) or not math.isfinite(harm) or abs(gain) > 100 or not (0 <= harm <= 100):
        raise ValueError("implausible or non-finite CE effect")
    return {"gain": gain, "old_harm": harm, "utility": gain - harm}


def validate_rows(rows, protocol):
    candidates = tuple(protocol["candidate_actions"])
    streams = tuple(protocol["batch_streams"])
    references = protocol["reference_trajectories"]
    interval = protocol["config"]["interval"]
    by_state = defaultdict(dict)
    for row in rows:
        root, reference, stage = int(row["root"]), row["reference"], int(row["stage"])
        stream, arm = int(row["stream"]), row["action"]
        if root not in protocol["roots"] or reference not in references or stage not in references[reference]:
            raise ValueError(f"unplanned root/reference/stage: {(root, reference, stage)}")
        if stream not in streams or arm not in candidates:
            raise ValueError("unplanned stream or action")
        if int(row["learner_updates"]) != interval:
            raise ValueError("branches do not have the same number of learner updates")
        key = (root, reference, stage)
        entry_key = (stream, arm)
        if entry_key in by_state[key]:
            raise ValueError(f"duplicate state/stream/action {key} {entry_key}")
        if len(row["pre_old_ce"]) != stage:
            raise ValueError("old-task CE history length differs from stage")
        effect = metrics(row)
        by_state[key][entry_key] = (row, effect)

    expected = {(s, a) for s in streams for a in candidates}
    for state, items in by_state.items():
        if set(items) != expected:
            raise ValueError(f"incomplete action/stream grid at {state}: {set(items) ^ expected}")
        pre = items[next(iter(expected))][0]
        for stream in streams:
            hashes = {items[(stream, arm)][0]["batch_stream_hash"] for arm in candidates}
            if len(hashes) != 1:
                raise ValueError(f"unpaired train minibatches at {state}, stream={stream}")
        for row, _ in items.values():
            if abs(float(row["pre_new_ce"]) - float(pre["pre_new_ce"])) > 1e-7:
                raise ValueError(f"branch starts from different new-task CE: {state}")
            if any(abs(float(a) - float(b)) > 1e-7 for a, b in zip(row["pre_old_ce"], pre["pre_old_ce"])):
                raise ValueError(f"branch starts from different old-task CE: {state}")
    return by_state


def mean(values):
    return statistics.fmean(values) if values else 0.0


def root_balanced_score(states, arm, stage=None):
    grouped = defaultdict(list)
    for (root, _, task), items in states.items():
        if stage is None or task == stage:
            grouped[root].append(items[(0, arm)][1]["utility"])
    if not grouped:
        raise ValueError("no training states for stage")
    return mean([mean(values) for values in grouped.values()])


def select_action(states, candidates, stage=None):
    """Use *only stream 0*. Lexical tie-break is deterministic."""
    return min(sorted(candidates), key=lambda action: (-root_balanced_score(states, action, stage), action))


def per_root(rows, arm):
    return {field: mean(row[arm][field] for row in rows) for field in ("gain", "old_harm", "utility")}


def bootstrap_root_ci(values, seed=1712, draws=2000):
    """Descriptive root bootstrap; not a confirmation test for six roots."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    replicates = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
    return [replicates[int(.025 * (draws - 1))], replicates[int(.975 * (draws - 1))]]


def analyze(rows, protocol, inactive=None):
    states = validate_rows(rows, protocol)
    roots = sorted({state[0] for state in states})
    allowed = tuple(protocol["candidate_actions"])
    fixed_candidates = tuple(protocol["state_blind_actions"])
    if not set(fixed_candidates).issubset(set(allowed)):
        raise ValueError("fixed action not in candidates")
    if tuple(protocol["batch_streams"]) != (0, 1):
        raise ValueError("cross-fitting requires exact streams 0 and 1")

    groups = defaultdict(list)
    for key, items in states.items():
        groups[key[0]].append((key, items))
    root_rows = []
    chosen = Counter()
    per_state = []
    for root in roots:
        others = {key: value for key, value in states.items() if key[0] != root}
        # Pilot fallback is diagnostic-only, with no valid LOO selection.
        fixed = select_action(others, fixed_candidates) if others else "global_0"
        stage_choices = {}
        for stage in sorted({key[2] for key, _ in groups[root]}):
            stage_training = {key: value for key, value in others.items() if key[2] == stage}
            stage_choices[stage] = select_action(stage_training, fixed_candidates, stage) if stage_training else fixed
        records = []
        for key, items in sorted(groups[root]):
            stage = key[2]
            # Oracle selection sees privileged stream-0 outcomes on this SAME
            # state; evaluation is instead on independently reseeded stream 1.
            oracle = min(sorted(allowed), key=lambda a: (-items[(0, a)][1]["utility"], a))
            chosen[oracle] += 1
            policies = {"fixed": fixed, "stage_only": stage_choices[stage],
                        "route_new_top2": "route_new_top2", "hindsight_oracle": oracle}
            validated = {label: items[(1, action)][1] for label, action in policies.items()}
            records.append(validated)
            per_state.append({"root": root, "reference": key[1], "stage": stage,
                              "chosen_by_scout": oracle, "heldout_actions": policies,
                              "stream1": validated})
        vals = {arm: per_root(records, arm) for arm in ("fixed", "stage_only", "route_new_top2", "hindsight_oracle")}
        root_rows.append({"root": root, "active_states": len(records),
                          "loo_fixed": fixed, "loo_stage_actions": stage_choices,
                          "policies": vals})

    policies = ("fixed", "stage_only", "route_new_top2", "hindsight_oracle")
    root_mean = {arm: {metric: mean([r["policies"][arm][metric] for r in root_rows])
                       for metric in ("gain", "old_harm", "utility")} for arm in policies}
    paired = {}
    for a, b in (("hindsight_oracle", "stage_only"), ("hindsight_oracle", "fixed"),
                 ("route_new_top2", "stage_only")):
        deltas = {metric: [r["policies"][a][metric] - r["policies"][b][metric]
                           for r in root_rows] for metric in ("gain", "old_harm", "utility")}
        paired[a + "_minus_" + b] = {
            "mean": {k: mean(v) for k, v in deltas.items()},
            "root_differences": [{"root": r["root"], **{k: deltas[k][i] for k in deltas}}
                                 for i, r in enumerate(root_rows)],
            "positive_utility_roots": sum(v > 0 for v in deltas["utility"]),
            "utility_root_bootstrap_95pct_descriptive": bootstrap_root_ci(deltas["utility"]),
        }
    gate = protocol["screening_gate"]
    comparison = paired["hindsight_oracle_minus_stage_only"]
    delta = comparison["mean"]
    root_coverage = sum(r["active_states"] >= gate["min_active_states_per_root"] for r in root_rows)
    tests = {
        "sufficient_active_roots": root_coverage >= gate["min_complete_roots"],
        "stage_policy_shows_new_task_learning_pressure": root_mean["stage_only"]["gain"] >= gate["min_stage_only_new_gain_CE"],
        "cross_stream_oracle_utility_gain": delta["utility"] >= gate["min_oracle_minus_stage_utility_CE"],
        "no_material_new_task_sacrifice": delta["gain"] >= -gate["max_allowed_oracle_new_gain_deficit_CE"],
        "old_task_harm_reduction": delta["old_harm"] <= -gate["min_stage_minus_oracle_old_harm_CE"],
        "improvement_across_roots": comparison["positive_utility_roots"] >= gate["min_roots_with_positive_utility_difference"]
    }
    status = "exploratory_signal_only" if all(tests.values()) else "dynamic_necessity_not_established"
    if not roots:
        status = "no_active_states"
    return {
        "status": status, "development_only": True, "confirmatory": False,
        "active_states": len(states), "independent_roots": len(roots), "evaluation_rows": len(rows),
        "inactive_states": inactive or [], "oracle_is_privileged": True,
        "root_policy_means": root_mean, "per_root": root_rows,
        "paired_root_deltas": paired, "hindsight_scout_action_histogram": dict(chosen),
        "per_state": per_state, "screening_gate": {"checks": tests, "passed": all(tests.values())},
        "interpretation": "Even a passing exploratory gate would only motivate a new test: stream-0 hindsight action labels are not observable at deployment, this uses previously seen development classes, and the 5-step intervention does not show end-of-sequence continual-learning benefit."
    }


def report(summary):
    delta = summary["paired_root_deltas"].get("hindsight_oracle_minus_stage_only", {})
    lines = ["# v12 paired action-value diagnostic", "",
             f"**Status:** {summary['status']} (development only, non-confirmatory).", "",
             f"**Sample:** {summary['independent_roots']} roots, {summary['active_states']} distinct learner states, {summary['evaluation_rows']} five-update branches. Two training-minibatch streams per state; actions/stages/streams are not independent roots.", "",
             "## Results (root-balanced, stream-1 evaluation)", "",
             "| Policy | New CE reduction (higher better) | Old positive CE increase (lower better) | Utility (gain−harm) |", "|---|---:|---:|---:|"]
    for name, row in summary["root_policy_means"].items():
        lines.append(f"| {name} | {row['gain']:.6f} | {row['old_harm']:.6f} | {row['utility']:.6f} |")
    if delta:
        lines += ["", "Hindsight oracle minus leave-one-root-out stage-only rule: " +
                  ", ".join(f"{k}={v:.6f}" for k, v in delta['mean'].items()) +
                  f"; positive-utility roots {delta['positive_utility_roots']}/{summary['independent_roots']}."]
    lines += ["", "## Locked exploratory screening gate", ""]
    for key, val in summary["screening_gate"]["checks"].items():
        lines.append(f"- {'PASS' if val else 'NOT MET'}: {key}")
    lines += ["", "This is **not a significance or noninferiority test**. The hindsight oracle selects each state's action using counterfactual labels from stream 0, unavailable to any deployed controller. Stream 1 holds out training minibatch draws, **not the eval images, states, roots, or classes**. A positive oracle gap would show exploitable decision sensitivity in this narrow environment, not that a state-observable learned policy can achieve it.", "",
              "All old/new held-out losses are measured on task-specific disjoint evaluation batches from the official CIFAR-100 *training* archive. The stem and task heads continue to update at expert multiplier zero. Equal learner steps are not a FLOPs guarantee. No pretrained language MoE or final-sequence benefit is established."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = json.loads((run / "locked_protocol.json").read_text())
    rows = json.loads((run / "branches.json").read_text())
    inactive = json.loads((run / "inactive_states.json").read_text())
    result = analyze(rows, protocol, inactive)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(report(result))
    print(json.dumps({"status": result["status"], "independent_roots": result["independent_roots"],
                      "active_states": result["active_states"], "evaluation_rows": result["evaluation_rows"]}))


if __name__ == "__main__":
    main()

"""Validate a completed v11 diagnostic run without training or selection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {0.0, 0.5, 1.0}
COST_KEYS = ("learner_updates", "train_examples", "monitor_examples", "route_examples",
             "controller_calls", "audit_examples")


def fail(message):
    raise ValueError(message)


def expected_eval_keys(protocol):
    keys = set()
    for rule in protocol["rules"]:
        for root in protocol["development_roots"]:
            keys.add(("baseline", rule, None, None, "greedy", None, root))
    for arm in protocol["arms"]:
        for seed in protocol["policy_seeds"]:
            for epoch in protocol["checkpoints"]:
                for root in protocol["development_roots"]:
                    keys.add(("learned", arm, seed, epoch, "greedy", None, root))
                    for draw in protocol["sampling_draws"]:
                        keys.add(("learned", arm, seed, epoch, "sample", draw, root))
    return keys


def eval_key(row):
    return tuple(row.get(field) for field in ("rowkind", "arm", "policy_seed", "epoch", "mode", "draw", "root"))


def finite(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        fail(f"{label} must be finite")


def validate_matrix(matrix, tasks, label):
    if not isinstance(matrix, list) or len(matrix) != tasks:
        fail(f"invalid {label} matrix size")
    for learned, values in enumerate(matrix):
        if not isinstance(values, list) or len(values) != tasks:
            fail(f"invalid {label}[{learned}] row size")
        for tested, value in enumerate(values):
            if tested <= learned:
                finite(value, f"{label}[{learned}][{tested}]")
            elif value is not None:
                fail(f"{label}[{learned}][{tested}] must be null above diagonal")


def validate_row(row, protocol, root_reference):
    key = eval_key(row)
    if row.get("split") != "dev": fail(f"{key}: split must be dev")
    tasks = protocol["config"]["tasks"]
    records = row.get("tasks")
    if not isinstance(records, list) or len(records) != tasks:
        fail(f"{key}: invalid tasks")
    for index, task in enumerate(records):
        if not isinstance(task, dict) or task.get("task") != index:
            fail(f"{key}: task index {index}")
        steps = task.get("steps")
        if not isinstance(steps, int) or not 0 <= steps <= protocol["config"]["max_steps"]:
            fail(f"{key}: illegal task steps")
    if sum(task["steps"] for task in records) != row.get("cost", {}).get("learner_updates"):
        fail(f"{key}: task updates do not equal cost learner_updates")
    validate_matrix(row.get("loss_matrix"), tasks, f"{key}:loss")
    validate_matrix(row.get("accuracy_matrix"), tasks, f"{key}:accuracy")
    cost = row.get("cost")
    if not isinstance(cost, dict): fail(f"{key}: missing cost")
    for name, value in cost.items(): finite(value, f"{key}:cost.{name}")
    if row.get("final_memory_counts", {}).get("replay", -1) > protocol["config"]["replay_capacity"]:
        fail(f"{key}: replay memory bound")
    if row.get("final_memory_counts", {}).get("monitor", -1) > protocol["config"]["monitor_capacity"]:
        fail(f"{key}: monitor memory bound")
    classes = row.get("classes")
    reference = root_reference.setdefault(row["root"], (classes, [task["task"] for task in records]))
    if (classes, [task["task"] for task in records]) != reference:
        fail(f"{key}: root classes/tasks differ across rollouts")
    events = row.get("events")
    if not isinstance(events, list): fail(f"{key}: missing events")
    by_task = {index: [] for index in range(tasks)}
    for event in events:
        task = event.get("task")
        if task not in by_task: fail(f"{key}: illegal event task")
        values = event.get("multipliers")
        if not isinstance(values, list) or len(values) != 2 or any(not isinstance(layer, list) or len(layer) != protocol["config"]["experts"] for layer in values):
            fail(f"{key}: illegal action shape")
        if any(float(value) not in LEVELS for layer in values for value in layer): fail(f"{key}: illegal action value")
        if task == 0 and any(float(value) != 1.0 for layer in values for value in layer): fail(f"{key}: task0 must use all-one prefix")
        by_task[task].append(event["steps"])
    interval = protocol["config"]["interval"]
    for index, task in enumerate(records):
        if by_task[index] != list(range(interval, task["steps"] + 1, interval)):
            fail(f"{key}: event steps inconsistent with task updates")


def validate_grid(rows, protocol):
    expected = expected_eval_keys(protocol)
    actual = [eval_key(row) for row in rows]
    counts = Counter(actual)
    if len(actual) != len(expected): fail(f"evaluation row count {len(actual)} != expected {len(expected)}")
    duplicate = [key for key, count in counts.items() if count != 1]
    if duplicate: fail(f"duplicate evaluation key {duplicate[0]}")
    missing = expected - set(actual)
    unexpected = set(actual) - expected
    if missing: fail(f"missing evaluation key {sorted(missing, key=repr)[0]}")
    if unexpected: fail(f"unexpected evaluation key {sorted(unexpected, key=repr)[0]}")
    references = {}
    for row in rows: validate_row(row, protocol, references)
    return len(expected)


def totals(records):
    return {key: sum(record.get(key, 0) for record in records) for key in COST_KEYS}


def validate_cost(cost, training, rows, probes):
    train_costs = [record["cost"] for record in training]
    groups = {"controller_training": totals(train_costs),
              "development_evaluation": totals([row["cost"] for row in rows]),
              "fixed_probe_collection": totals(probes)}
    for name, expected in groups.items():
        if cost.get(name) != expected: fail(f"cost mismatch for {name}")
    expected_total = {key: sum(group[key] for group in groups.values()) for key in COST_KEYS}
    if cost.get("total") != expected_total: fail("cost total mismatch")
    for record in training:
        value = record["cost"]
        if value.get("branches") != 4 * len(record["log"]):
            fail("training branches do not match groups x 4")
        if value.get("prefix_updates", -1) > value.get("learner_updates", -1):
            fail("training prefix updates exceed physical updates")
        decisions = sum(sum(entry.get("decision_counts", ())) for entry in record["log"])
        expected_updates = value["prefix_updates"] + decisions * 5
        if value["learner_updates"] != expected_updates:
            fail("physical training updates do not equal shared prefixes plus paired branch intervals")
    return groups, expected_total


def validate_sources(run, protocol):
    for relative, expected in protocol.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            fail(f"source hash mismatch {relative}")
    dataset = protocol.get("dataset", {})
    if dataset.get("loaded_splits") != ["train"] or dataset.get("official_test_loaded") is not False:
        fail("run must claim official CIFAR train only")
    completion = json.loads((run / "completion.json").read_text())
    if completion.get("complete") is not True or completion.get("source_hashes_unchanged") is not True:
        fail("completion/source stability check failed")


def validate_probes(run, protocol):
    payload = json.loads((run / "probes.json").read_text())
    provenance = payload.get("provenance")
    if not isinstance(provenance, list): fail("probe provenance missing")
    observations = torch.load(run / "probe_observations.pt", map_location="cpu", weights_only=True)
    if not isinstance(observations, list) or len(observations) != len(provenance) or not observations:
        fail("probe tensor/provenance count mismatch")
    seen = set()
    for tensor, item in zip(observations, provenance):
        if item.get("root") not in protocol["training_roots"]: fail("probe uses non-training root")
        if item.get("task") not in range(1, protocol["config"]["tasks"]): fail("probe is not a controlled task")
        key = (item["root"], item["task"], item.get("step"), item.get("sha256"))
        if key in seen: fail("duplicate probe provenance")
        seen.add(key)
        digest = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
        if digest != item.get("sha256"): fail("probe tensor hash mismatch")
    return len(provenance), payload.get("cost", [])


def validate_checkpoints(run, protocol):
    expected = {f"{arm}-{seed}-epoch{epoch}.pt" for arm in protocol["arms"]
                for seed in protocol["policy_seeds"] for epoch in protocol["checkpoints"]}
    found = {path.name for path in (run / "checkpoints").glob("*.pt")}
    if found != expected: fail("checkpoint file grid mismatch")
    if any((run / "checkpoints" / name).stat().st_size == 0 for name in found): fail("empty checkpoint")
    return len(found)


def validate_replay(run):
    values = json.loads((run / "original_replay_check.json").read_text())
    for arm in ("local", "coordinated"):
        choices = [item for item in values if item.get("arm") == arm and item.get("seed") == 881]
        if len(choices) != 1 or choices[0].get("applicable") is not True or choices[0].get("passed") is not True:
            fail(f"original replay check failed for {arm}/881")
        if choices[0].get("compared_rounds", 0) < 6: fail(f"insufficient original replay rounds for {arm}/881")


def audit(run):
    run = Path(run)
    protocol = json.loads((run / "locked_protocol.json").read_text())
    validate_sources(run, protocol)
    rows = json.loads((run / "evaluations.json").read_text())
    expected_rows = validate_grid(rows, protocol)
    probe_count, probes = validate_probes(run, protocol)
    checkpoint_count = validate_checkpoints(run, protocol)
    validate_replay(run)
    training_paths = sorted((run / "training").glob("*.json"))
    expected_training = {f"{arm}-{seed}.json" for arm in protocol["arms"] for seed in protocol["policy_seeds"]}
    if {path.name for path in training_paths} != expected_training: fail("training record file grid mismatch")
    training = [json.loads(path.read_text()) for path in training_paths]
    cost = json.loads((run / "total_cost.json").read_text())
    groups, total = validate_cost(cost, training, rows, probes)
    return {"status": "integrity audit; development-only diagnostic, no selection claim", "evaluation_rows": expected_rows,
            "checkpoint_files": checkpoint_count, "probe_observations": probe_count,
            "validated_sources": len(protocol.get("source_hashes", {})), "physical_cost": total,
            "cost_groups": groups, "development_only": protocol.get("development_only") is True,
            "warning": "Rows share four development roots; policy draws and checkpoints are not independent histories."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(); result = audit(args.run)
    out = args.run / "analysis"; out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"rows": result["evaluation_rows"], "checkpoints": result["checkpoint_files"]}))


if __name__ == "__main__": main()

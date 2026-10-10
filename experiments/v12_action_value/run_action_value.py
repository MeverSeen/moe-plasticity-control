"""v12: paired five-step intervention test before training another controller.

Development-only. Does NOT load CIFAR-100 official test data or pretrained MoEs.
Run from anywhere: python experiments/v12_action_value/run_action_value.py
    --download --out runs/v12-first-probe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V8 = HERE.parent / "v8_multilayer_retention"
sys.path.insert(0, str(V8))
from control import rule_chooser  # noqa: E402
from data import CIFAR100_MD5, class_partition, load_dataset  # noqa: E402
from environment import RetentionEnv, measure  # noqa: E402
from run_retention import history_for  # noqa: E402

from action_grid import ACTIONS, STATE_BLIND, action_for, batch_stream_fingerprint, stream_seed  # noqa: E402
from analyze_action_value import analyze, report  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def lock_hashes():
    files = [HERE / name for name in ("protocol.json", "action_grid.py", "run_action_value.py",
                                      "analyze_action_value.py")]
    files.extend(sorted(V8.glob("*.py")))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def eval_ce(model, batch):
    losses, _ = measure(model, batch)
    return float(losses.mean())


def heldout_ce(model, history, stage):
    """Use disjoint per-task eval from the official train split; no monitor data."""
    old = [eval_ce(model, task["eval"]) for task in history[:stage]]
    new = eval_ce(model, history[stage]["eval"])
    return new, old, sum(len(task["eval"]) for task in history[:stage + 1])


def evaluate_state(env, history, root, reference, stage, protocol):
    """Return every action/stream from ONE immutable learner state."""
    if env.current_loss <= env.c["target_loss"] or env.local_steps != 0:
        raise ValueError("state must be at the first active decision of a task")
    observation = env.observe().detach().clone()
    baseline_new, baseline_old, eval_size = heldout_ce(env.model, history, stage)
    interval, batch = protocol["config"]["interval"], protocol["config"]["batch"]
    old_batch = env.memory.replay()
    old_size = 0 if old_batch is None else len(old_batch)
    rows, accounting = [], {"branch_updates": 0, "branch_eval_examples": eval_size,
                             "branch_train_examples": 0, "branch_monitor_examples": 0,
                             "branch_route_examples": 0, "branch_controller_calls": 0}
    for stream in protocol["batch_streams"]:
        seed = stream_seed(root, reference, stage, stream)
        paired_hash = batch_stream_fingerprint(root, reference, stage, stream,
                                               len(env.current["train"]), old_size,
                                               interval, batch)
        for name in protocol["candidate_actions"]:
            branch = env.clone()  # learner, optimizer moments, buffers and RNGs are copied
            branch.rng = torch.Generator().manual_seed(seed)
            action = action_for(observation, name)
            if action.shape != (2, protocol["config"]["experts"]):
                raise ValueError("invalid action shape")
            if not torch.isin(action, torch.tensor([0., .5, 1.])).all():
                raise ValueError("illegal action multiplier")
            before_cost = dict(branch.cost)
            branch.advance(action)  # exactly one fixed 5-step window; no early stopping
            new_ce, old_ce, examples = heldout_ce(branch.model, history, stage)
            delta = {key: branch.cost[key] - before_cost[key]
                     for key in ("learner_updates", "train_examples", "monitor_examples",
                                 "route_examples", "controller_calls")}
            if delta["learner_updates"] != interval:
                raise RuntimeError(f"unexpected branch length: {delta}")
            row = {"root": root, "reference": reference, "stage": stage,
                   "stream": stream, "action": name, "batch_stream_hash": paired_hash,
                   "initial_monitor_ce": env.current_loss,
                   "post_monitor_ce": branch.current_loss,
                   "post_monitor_hit": branch.current_loss <= branch.c["target_loss"],
                   "pre_new_ce": baseline_new, "post_new_ce": new_ce,
                   "pre_old_ce": baseline_old, "post_old_ce": old_ce,
                   "learner_updates": delta["learner_updates"],
                   "train_examples": delta["train_examples"],
                   "monitor_examples": delta["monitor_examples"],
                   "route_examples": delta["route_examples"],
                   "controller_calls": delta["controller_calls"],
                   "eval_examples": examples,
                   "action_matrix": action.tolist()}
            rows.append(row)
            accounting["branch_updates"] += delta["learner_updates"]
            accounting["branch_train_examples"] += delta["train_examples"]
            accounting["branch_monitor_examples"] += delta["monitor_examples"]
            accounting["branch_route_examples"] += delta["route_examples"]
            accounting["branch_controller_calls"] += delta["controller_calls"]
            accounting["branch_eval_examples"] += examples
    return rows, accounting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/datasets")
    parser.add_argument("--download", action="store_true", help="Download checksum-verified official CIFAR-100")
    parser.add_argument("--pilot", action="store_true", help="Single-root engineering smoke; cannot pass screening gate")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("output exists; specify a fresh directory")
    if (ROOT / "experiments").resolve() in out.parents:
        parser.error("refuse to write into original experiment archive")
    plan = json.loads((HERE / "protocol.json").read_text())
    if tuple(plan["candidate_actions"]) != ACTIONS or tuple(plan["state_blind_actions"]) != STATE_BLIND:
        raise RuntimeError("frozen protocol/action code mismatch")
    if plan["class_pool"] != "dev" or plan["class_partition_seed"] != 8800:
        raise RuntimeError("unreviewed class-pool change")
    if tuple(plan["batch_streams"]) != (0, 1):
        raise RuntimeError("this run requires two held-out future batch streams")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    source_hashes = lock_hashes()
    images, labels = load_dataset(args.cache, "train", download=args.download)
    partition = class_partition(seed=plan["class_partition_seed"])
    roots = plan["roots"][:1] if args.pilot else plan["roots"]
    locked = dict(plan, roots=roots, pilot=args.pilot, source_hashes=source_hashes,
                  class_partition=partition,
                  loaded_dataset={"name": "CIFAR-100", "archive_md5": CIFAR100_MD5,
                                  "splits": ["train"], "official_test_loaded": False})
    out.mkdir(parents=True)
    write_json(out / "locked_protocol.json", locked)
    write_json(out / "environment.json", {"python": sys.version, "torch": torch.__version__,
                                           "numpy": np.__version__, "platform": platform.platform(),
                                           "threads": 1, "device": "cpu"})
    config = locked["config"]
    rows, inactive, collections, costs, reference_costs = [], [], [], [], []
    for root in roots:
        history = history_for(images, labels, partition["dev"], root, config)
        for reference, stages in locked["reference_trajectories"].items():
            env = RetentionEnv(history, root, config)
            for stage in range(len(history)):
                if stage in stages:
                    if env.current_loss > env.c["target_loss"] and env.local_steps == 0:
                        branch_rows, cost = evaluate_state(env, history, root, reference, stage, locked)
                        rows.extend(branch_rows)
                        costs.append({"root": root, "reference": reference, "stage": stage, **cost})
                        collections.append({"root": root, "reference": reference, "stage": stage,
                                            "initial_monitor_ce": env.current_loss,
                                            "state_observation_sha256": hashlib.sha256(env.observe().numpy().tobytes()).hexdigest()})
                    else:
                        inactive.append({"root": root, "reference": reference, "stage": stage,
                                         "initial_monitor_ce": env.current_loss,
                                         "reason": "already_at_target_or_not_first_decision"})
                env.run_task(rule_chooser("global_1" if stage == 0 else reference))
            reference_costs.append({"root": root, "reference": reference, **{key: env.cost[key] for key in ("learner_updates", "train_examples", "monitor_examples", "route_examples", "controller_calls")}})
            print(f"root {root} reference {reference} done: states={len(collections)}, branches={len(rows)}", flush=True)
        write_json(out / "branches.json", rows)
        write_json(out / "inactive_states.json", inactive)
        write_json(out / "states.json", collections)
        write_json(out / "branch_costs.json", costs)
        write_json(out / "reference_costs.json", reference_costs)

    if not rows:
        raise RuntimeError("No active learner state in locked sample; report task coverage as negative diagnostic")
    if source_hashes != lock_hashes():
        raise RuntimeError("experiment source files changed during the locked run")
    summary = analyze(rows, locked, inactive)
    (out / "analysis").mkdir()
    write_json(out / "analysis/summary.json", summary)
    (out / "analysis/report.md").write_text(report(summary))
    write_json(out / "completion.json", {"complete": True, "source_hashes_unchanged": True,
                                         "branch_rows": len(rows), "root_count": len(roots),
                                         "active_states": len(collections),
                                         "reference_updates": sum(c["learner_updates"] for c in reference_costs),
                                         "reference_train_examples": sum(c["train_examples"] for c in reference_costs),
                                         "reference_monitor_examples": sum(c["monitor_examples"] for c in reference_costs),
                                         "branch_updates": sum(c["branch_updates"] for c in costs),
                                         "branch_train_examples": sum(c["branch_train_examples"] for c in costs),
                                         "branch_eval_examples": sum(c["branch_eval_examples"] for c in costs),
                                         "wall_seconds": time.perf_counter() - started,
                                         "pilot": args.pilot,
                                         "result_status": summary["status"]})
    print(f"v12 status={summary['status']} roots={summary['independent_roots']} states={summary['active_states']} rows={summary['evaluation_rows']}")


if __name__ == "__main__":
    main()

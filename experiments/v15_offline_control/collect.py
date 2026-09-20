"""Bounded multi-behavior data collection and paired restore verification."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import save_json, file_sha
from environment import Config, BudgetStop, hash_tree
from logged_environment import LoggedEnvironment, LoggedLearner
from structured_policy import Rule


def checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp); tmp.replace(path)
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


class Collector:
    def __init__(self, root, allocation, protocol):
        self.root, self.allocation = Path(root), Path(allocation)
        self.plan = json.loads(Path(protocol).read_text())
        self.protocol_sha256 = file_sha(protocol)
        self.cfg = Config(**self.plan["environment"])
        self.started = time.time()
        self.deadline = min(self.started + self.plan["collection_cap_seconds"],
                            self.plan["hard_deadline_epoch"] - self.plan["evaluation_archive_reserve_seconds"])
        os.environ["V13_DEADLINE_EPOCH"] = str(self.plan["hard_deadline_epoch"])
        os.environ["V13_STAGE_DEADLINE_EPOCH"] = str(self.deadline)
        self.data = torch.load(self.allocation / "dataset.pt", weights_only=False)
        self.learner = LoggedLearner(str(self.allocation / "model"))
        self.rows = []; self.durations = []
        self.emergency = self.root / "emergency.pt"

    def progress(self, stage, **extra):
        value = {"stage": stage, "epoch": time.time(), "completed": len(self.rows),
                 "wall_seconds": time.time() - self.started,
                 "learner_cost": self.learner.cost, **extra}
        save_json(self.root / "progress.json", value)
        print(json.dumps(value), flush=True)

    def root_spec(self, raw):
        spec = copy.deepcopy(raw)
        if spec.get("target_source"):
            path = self.allocation / spec.pop("target_source")
            old = json.loads(path.read_text())["root"]
            assert old["seed"] == spec["seed"] and old["tasks"] == spec["tasks"]
            spec["targets"] = old["targets"]
            save_json(self.root / "targets" / (spec["id"] + ".json"),
                      {"root": spec, "source": str(path), "sha256": file_sha(path)})
        else:
            e = LoggedEnvironment(self.learner, self.data, spec, self.cfg, reference=True)
            reference = e.run(Rule([4] * 16, mode="fixed"), checkpoint_path=self.emergency)
            save_json(self.root / "references" / (spec["id"] + ".json"), reference)
            spec["targets"] = [float(np.median([x["loss"] for x in curve
                                               if x["step"] in (20, 25, 30)]))
                               for curve in reference["loss_curves"]]
            save_json(self.root / "targets" / (spec["id"] + ".json"), {"root": spec})
        return spec

    def prefix(self, spec):
        e = LoggedEnvironment(self.learner, self.data, spec, self.cfg)
        state, info = e.prepare_prefix(self.emergency)
        saved = checkpoint(self.root / "prefixes" / (spec["id"] + ".pt"), state)
        save_json(self.root / "prefixes" / (spec["id"] + ".json"), {**info, "file": saved})
        return state

    def restore_probe(self, state, root_id):
        original_hash = hash_tree(state)
        outputs, states = [], []
        for repeat in range(2):
            e = LoggedEnvironment.from_state(self.learner, self.data, self.cfg,
                                             state["environment"], state["learner"])
            losses = []
            for _ in range(self.cfg.hold):
                e._guard(self.emergency, {})
                e._update(torch.ones(16, 64)); losses.append(e.current_loss)
            after = self.learner.snapshot()
            states.append(hash_tree({"weights": after["weights"], "optimizer": after["optimizer"]}))
            outputs.append(losses)
            if hash_tree(state) != original_hash:
                raise RuntimeError("Reusable prefix was mutated")
        outcome = {"root": root_id, "window": self.cfg.hold, "losses": outputs,
                   "state_hashes": states, "checkpoint_immutable": True,
                   "equal": outputs[0] == outputs[1] and states[0] == states[1]}
        save_json(self.root / "restore_checks" / (root_id + ".json"), outcome)
        if not outcome["equal"]:
            raise RuntimeError("Paired same-action restore check failed; collection stopped")

    def one(self, state, spec, behavior, index, directory="train"):
        policy = Rule([behavior["base"]] * 16, epsilon=behavior["epsilon"], mode=behavior["mode"])
        e = LoggedEnvironment.from_state(self.learner, self.data, self.cfg,
                                         state["environment"], state["learner"])
        # Defensive check is performed on fresh restored learner state for every path.
        restored = self.learner.snapshot()
        expected = hash_tree({"weights": state["learner"]["weights"], "optimizer": state["learner"]["optimizer"]})
        observed = hash_tree({"weights": restored["weights"], "optimizer": restored["optimizer"]})
        if observed != expected:
            raise RuntimeError("Full restored weights/Adam state mismatch")
        name = f'{spec["id"]}__{index:03d}_{behavior["name"]}'
        start = time.time(); self.progress("history_start", root=spec["id"], behavior=behavior["name"])
        try:
            result = e.run_logged(policy, state["learner"]["weights"],
                                  self.root / "results" / directory / (name + ".json"),
                                  stochastic=behavior["stochastic"], action_seed=15150000 + index,
                                  behavior_name=behavior["name"], checkpoint_path=self.emergency)
        except BudgetStop:
            save_json(self.root / "incomplete" / (name + ".json"),
                      {"root": spec, "history_completed": False, "trace": e.trace,
                       "reason": "budget, not a learner task failure"})
            raise
        adapter = checkpoint(self.root / "adapters" / (name + ".pt"),
                             {"weights": [p.detach().cpu().clone() for p in self.learner.parameters],
                              "base_revision": self.plan["base_revision"],
                              "root": spec["id"], "behavior": behavior["name"]})
        adapter["factor_content_sha256"] = hash_tree([p.detach().cpu() for p in self.learner.parameters])
        result["endpoint_adapter"] = adapter
        result["source_protocol_sha256"] = self.protocol_sha256
        result["total_wall_seconds_including_logging"] = time.time() - start
        save_json(self.root / "results" / directory / (name + ".json"), result)
        self.durations.append(time.time() - start)
        row = {"name": name, "root": spec["id"], "behavior": behavior["name"],
               "success": result["success"], "decisions": len(result["trace"]),
               "monitor_F": result["monitor"]["forgetting"],
               "eval_F_diagnostic": result["evaluation"]["forgetting"],
               "old_ce": float(np.mean([x["mean_ce"] for x in result["raw_terminal"]["old_tasks"]])),
               "effective_distance": result["parameter_distance"]["effective_sq"],
               "wall_seconds": time.time() - start, "directory": directory}
        self.rows.append(row); self.progress("history_complete", row=row)
        return result

    def main(self):
        state = None
        try:
            prefixes, specs = {}, []
            for raw in self.plan["roots"]["train"]:
                spec = self.root_spec(raw); specs.append(spec)
                self.progress("prefix_start", root=spec["id"])
                prefixes[spec["id"]] = self.prefix(spec)
                self.restore_probe(prefixes[spec["id"]], spec["id"])
            # Balanced round robin; stop by cost, never by whether a behavior wins.
            for b, behavior in enumerate(self.plan["behaviors"]):
                for r, spec in enumerate(specs):
                    estimate = max(360., max(self.durations[-4:] or [360.]) * 1.25)
                    if time.time() + estimate >= self.deadline:
                        raise BudgetStop("Admission reserve for a complete trajectory")
                    state = prefixes[spec["id"]]
                    result = self.one(state, spec, behavior, b * len(specs) + r)
                    if b == 0 and r == 0:
                        repeated = self.one(state, spec, behavior, 999, "restore_diagnostic")
                        same = (result["loss_curves"] == repeated["loss_curves"] and
                                result["monitor"] == repeated["monitor"] and
                                result["endpoint_adapter"]["factor_content_sha256"] ==
                                repeated["endpoint_adapter"]["factor_content_sha256"])
                        save_json(self.root / "restore_checks/full_history.json", {"equal": same,
                                  "scope": "same prefix, complete all-one B/C histories"})
                        if not same:
                            raise RuntimeError("Complete-history replay diverged")
            status = "completed"
        except BudgetStop as error:
            status = "bounded_incomplete"
            self.progress("budget_stop", reason=str(error))
        self.learner.check_frozen()
        save_json(self.root / "collection_summary.json", {"status": status, "rows": self.rows,
                  "native_cost": self.learner.cost, "wall_seconds": time.time() - self.started,
                  "scope": "training collection; no held-out policy efficacy conclusion"})
        self.progress("collection_finished", status=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    Collector(args.root, args.allocation, args.protocol).main()

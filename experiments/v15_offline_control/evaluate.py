"""Development selection and frozen closed-loop evaluation; isolated from fitting."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import save_json, file_sha
from collect import Collector, checkpoint
from environment import BudgetStop, hash_tree
from logged_environment import LoggedEnvironment
from structured_policy import ActorCritic, Rule
from evaluation import GeneralEvaluator, compare_general
from lora import frozen_hash


def choose_iql(results, required_roots):
    names = sorted({r["method"] for r in results if r["method"].startswith("iql_")})
    scores = {}
    for name in names:
        rows = [r for r in results if r["method"] == name]
        if {r["root"] for r in rows} != set(required_roots):
            continue
        scores[name] = [sum(r["failures"] for r in rows),
                        float(np.mean([r["F"] for r in rows])),
                        float(np.mean([r["final_ce"] for r in rows])),
                        sum(r["updates"] for r in rows)]
    if not scores:
        raise ValueError("No complete IQL development candidate; confirmation cannot select from partial results")
    return min(scores, key=lambda x: (scores[x], x)), scores


class Evaluator(Collector):
    def __init__(self, root, allocation, protocol):
        super().__init__(root, allocation, protocol)
        self.deadline = self.plan["hard_deadline_epoch"] - 900
        os.environ["V13_STAGE_DEADLINE_EPOCH"] = str(self.deadline)
        self.answers = torch.load(self.allocation / "v14/answer_lookup.pt", weights_only=False)
        self.general = GeneralEvaluator(json.loads((self.allocation / "v14/general_examples.json").read_text()),
                                        json.loads((self.allocation / "v14/general_manifest.json").read_text()))
        self.initial_general = json.loads((self.allocation / "v14/results/short_comparison/initial_general.json").read_text())
        self.eval_rows = []

    def policy(self, name):
        if name == "fixed1": return Rule([4] * 16, epsilon=0., mode="fixed")
        if name == "fixed050": return Rule([2] * 16, epsilon=0., mode="fixed")
        if name == "rule": return Rule([4] * 16, epsilon=0.)
        policy = ActorCritic()
        saved = torch.load(self.root / "offline" / name / "controller.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(saved["policy"]); policy.eval()
        return policy

    def evaluate_one(self, spec, prefix, name, phase):
        if time.time() + 420 >= self.deadline:
            raise BudgetStop("Evaluation admission reserve")
        e = LoggedEnvironment.from_state(self.learner, self.data, self.cfg,
                                         prefix["environment"], prefix["learner"],
                                         answer_lookup=self.answers if phase == "confirmation" else None)
        current = self.learner.snapshot()
        assert hash_tree({"weights": current["weights"], "optimizer": current["optimizer"]}) == hash_tree(
            {"weights": prefix["learner"]["weights"], "optimizer": prefix["learner"]["optimizer"]})
        policy = self.policy(name)
        self.progress("evaluation_start", root=spec["id"], method=name, phase=phase)
        path = self.root / "results" / phase / (spec["id"] + "__" + name + ".json")
        started = time.time()
        try:
            result = e.run_logged(policy, prefix["learner"]["weights"], path,
                                  behavior_name=name, checkpoint_path=self.emergency)
        except BudgetStop:
            save_json(path.with_name(path.stem + "__incomplete.json"),
                      {"root": spec, "method": name, "history_completed": False,
                       "trace": e.trace, "reason": "time budget, not task failure"})
            raise
        result["method"] = name; result["phase"] = phase
        if name not in ("fixed1", "fixed050", "rule"):
            result["controller_sha256"] = file_sha(self.root / "offline" / name / "controller.pt")
        if phase == "confirmation":
            result["endpoint_adapter"] = checkpoint(self.root / "evaluation_adapters" / (path.stem + ".pt"),
                {"weights": [p.detach().cpu().clone() for p in self.learner.parameters],
                 "base_revision": self.plan["base_revision"], "root": spec["id"], "method": name})
            e._guard(self.emergency, {})
            general = self.general.run(self.learner)
            save_json(path.with_name(path.stem + "__general.json"), general)
            result["general_ability"] = compare_general(self.initial_general, general)
            result["general_cost_actual"] = general["cost"]
        result["total_wall_seconds_including_logging"] = time.time() - started
        save_json(path, result)
        row = {"phase": phase, "root": spec["id"], "method": name,
               "failures": result["failure_count"], "success": result["success"],
               "F": result["evaluation"]["forgetting"], "final_ce": result["evaluation"]["final_loss"],
               "updates": result["cost_path"]["updates"], "decisions": len(result["trace"]),
               "wall_seconds": time.time() - started}
        self.eval_rows.append(row)
        save_json(self.root / "evaluation_rows.json", self.eval_rows)
        self.progress("evaluation_complete", row=row)

    def main(self):
        base_start = frozen_hash(self.learner.model)
        self.progress("evaluation_initialized")
        fits = json.loads((self.root / "offline/summary.json").read_text())["fits"]
        iql = ["iql_" + x["reward"] for x in fits if x["method"] == "iql"]
        status = "completed"
        try:
            for raw in self.plan["roots"]["development"]:
                spec = self.root_spec(raw); prefix = self.prefix(spec)
                for name in ["fixed1", "fixed050", "rule", "bc"] + iql:
                    self.evaluate_one(spec, prefix, name, "development")
            selected, scores = choose_iql(self.eval_rows,
                                         [x["id"] for x in self.plan["roots"]["development"]])
            save_json(self.root / "selection_lock.json", {"selected": selected, "scores": scores,
                       "locked_epoch": time.time(), "controller_sha256": file_sha(self.root / "offline" / selected / "controller.pt"),
                       "rule": "failures, mean positive eval forgetting, mean final CE, updates; two dev roots only"})
            for raw in self.plan["roots"]["confirmation"]:
                spec = self.root_spec(raw); prefix = self.prefix(spec)
                for name in ["fixed1", "fixed050", "rule", "bc", selected]:
                    self.evaluate_one(spec, prefix, name, "confirmation")
        except BudgetStop as error:
            status = "bounded_incomplete"; self.progress("evaluation_budget_stop", reason=str(error))
        self.learner.check_frozen()
        base_end = frozen_hash(self.learner.model)
        assert base_start == base_end
        save_json(self.root / "evaluation_summary.json", {"status": status, "rows": self.eval_rows,
                   "native_cost": self.learner.cost, "wall_seconds": time.time() - self.started,
                   "base_start": base_start, "base_end": base_end,
                   "scope": "repeated held-out task pool, two development/two evaluation roots, single controller seed"})
        self.progress("evaluation_finished", status=status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--allocation", required=True); p.add_argument("--protocol", required=True)
    args = p.parse_args(); Evaluator(args.root, args.allocation, args.protocol).main()

"""Focused corruption tests for scripts.audit_v11_run validation helpers."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_v11_run as audit


def protocol():
    return {"rules": ["r"], "development_roots": [1], "arms": ["local"], "policy_seeds": [7],
            "checkpoints": [0, 1], "sampling_draws": [0, 1, 2], "training_roots": [9],
            "config": {"tasks": 2, "experts": 2, "max_steps": 10, "interval": 5,
                       "replay_capacity": 8, "monitor_capacity": 8}}


def row(key):
    kind, arm, seed, epoch, mode, draw, root = key
    tasks = [{"task": 0, "steps": 5, "success": True}, {"task": 1, "steps": 5, "success": True}]
    events = [{"task": task, "steps": 5, "multipliers": [[1., 1.], [1., 1.]]} for task in range(2)]
    return {"rowkind": kind, "arm": arm, "policy_seed": seed, "epoch": epoch, "mode": mode, "draw": draw,
            "root": root, "split": "dev", "tasks": tasks, "loss_matrix": [[1., None], [1.1, 1.]],
            "accuracy_matrix": [[.5, None], [.4, .5]], "cost": {key: 10 for key in audit.COST_KEYS},
            "final_memory_counts": {"replay": 2, "monitor": 2}, "classes": [[0], [1]], "events": events}


class AuditTests(unittest.TestCase):
    def test_expected_grid_is_344_for_real_protocol_shape(self):
        real = {"rules": list(range(6)), "development_roots": list(range(4)), "arms": ["l", "c"],
                "policy_seeds": [881, 882], "checkpoints": [0, 3, 6, 24, 48], "sampling_draws": [0, 1, 2]}
        self.assertEqual(344, len(audit.expected_eval_keys(real)))

    def test_grid_rejects_missing_and_duplicate_rows(self):
        p = protocol(); rows = [row(key) for key in audit.expected_eval_keys(p)]
        self.assertEqual(len(rows), audit.validate_grid(rows, p))
        with self.assertRaisesRegex(ValueError, "row count"):
            audit.validate_grid(rows[:-1], p)
        broken = rows[:-1] + [copy.deepcopy(rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.validate_grid(broken, p)

    def test_cost_recomputation_rejects_double_charge(self):
        p = protocol(); rows = [row(key) for key in audit.expected_eval_keys(p)]
        training = [{"log": [{"decision_counts": [0, 0, 0, 0]}, {"decision_counts": [0, 0, 0, 0]}],
                     "cost": {**{key: 2 for key in audit.COST_KEYS}, "branches": 8, "prefix_updates": 2}}]
        probes = [{key: 3 for key in audit.COST_KEYS}]
        groups = {"controller_training": audit.totals([training[0]["cost"]]),
                  "development_evaluation": audit.totals([item["cost"] for item in rows]),
                  "fixed_probe_collection": audit.totals(probes)}
        total = {key: sum(group[key] for group in groups.values()) for key in audit.COST_KEYS}
        cost = {**groups, "total": total}
        audit.validate_cost(cost, training, rows, probes)
        cost["total"]["learner_updates"] += 1
        with self.assertRaisesRegex(ValueError, "total"):
            audit.validate_cost(cost, training, rows, probes)

    def test_row_rejects_illegal_action_and_matrix(self):
        p = protocol(); valid = row(next(iter(audit.expected_eval_keys(p))))
        audit.validate_row(valid, p, {})
        broken = copy.deepcopy(valid); broken["events"][0]["multipliers"][0][0] = .25
        with self.assertRaisesRegex(ValueError, "illegal action"):
            audit.validate_row(broken, p, {})
        broken = copy.deepcopy(valid); broken["loss_matrix"][0][1] = 1.
        with self.assertRaisesRegex(ValueError, "above diagonal"):
            audit.validate_row(broken, p, {})


if __name__ == "__main__": unittest.main(verbosity=2)

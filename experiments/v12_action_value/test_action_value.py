import copy
import json
from pathlib import Path
import unittest

import torch

from action_grid import ACTIONS, STATE_BLIND, action_for, stream_seed, batch_stream_fingerprint
from analyze_action_value import analyze, metrics, select_action, validate_rows


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "protocol.json").read_text())


def synthetic_rows(roots=(101, 102, 103, 104, 105, 106)):
    plan = dict(PLAN, roots=list(roots), reference_trajectories={"global_0": [1, 2]})
    rows = []
    for root in roots:
        for stage in (1, 2):
            for stream in (0, 1):
                for action in ACTIONS:
                    # Artificial action interactions: a dynamic oracle should
                    # use scout labels, never peek at the validation stream.
                    chosen = "global_1" if stage == 1 else "global_0"
                    gain = .04 if action == chosen else .02
                    harm = .0 if action == chosen else .01
                    rows.append({"root": root, "reference": "global_0", "stage": stage,
                                 "stream": stream, "action": action,
                                 "batch_stream_hash": f"{root}-{stage}-{stream}",
                                 "learner_updates": plan["config"]["interval"],
                                 "pre_new_ce": 2.0, "post_new_ce": 2.0 - gain,
                                 "pre_old_ce": [1.0] * stage,
                                 "post_old_ce": [1.0 + harm] * stage})
    return rows, plan


class ActionGridTests(unittest.TestCase):
    def setUp(self):
        self.obs = torch.zeros((2, 8, 13), dtype=torch.float32)
        self.obs[:, :, 3] = torch.tensor([.9, .8, .2, .1, .1, .1, .1, .1])
        self.obs[:, :, 1] = torch.tensor([.1, .2, .3, .7, .8, .1, .1, .1])

    def test_fixed_and_sparse_mask_shapes(self):
        for name in ACTIONS:
            action = action_for(self.obs, name)
            self.assertEqual((2, 8), tuple(action.shape))
            self.assertTrue(torch.isin(action, torch.tensor([0., .5, 1.])).all())
        for name in ("static_top2", "route_new_top2", "route_old_top2"):
            self.assertTrue((action_for(self.obs, name).sum(-1) == 2).all())
        self.assertTrue((action_for(self.obs, "global_0") == 0).all())
        self.assertTrue((action_for(self.obs, "global_0.5") == .5).all())
        self.assertTrue((action_for(self.obs, "global_1") == 1).all())
        self.assertTrue((action_for(self.obs, "static_top2")[:, :2] == 1).all())
        self.assertEqual(1., float(action_for(self.obs, "route_new_top2")[0, 0]))
        self.assertEqual(1., float(action_for(self.obs, "route_old_top2")[0, 4]))

    def test_invalid_action_or_shape(self):
        with self.assertRaises(ValueError):
            action_for(self.obs, "dynamic_winner")
        with self.assertRaises(ValueError):
            action_for(torch.zeros((8, 13)), "global_0")

    def test_pairing_stream_seed_does_not_depend_on_action(self):
        a = stream_seed(89120, "global_0", 2, 0)
        b = stream_seed(89120, "global_0", 2, 1)
        self.assertNotEqual(a, b)
        self.assertEqual(a, stream_seed(89120, "global_0", 2, 0))
        self.assertNotEqual(batch_stream_fingerprint(89120, "global_0", 2, 0, 240, 32, 5, 32),
                            batch_stream_fingerprint(89120, "global_0", 2, 1, 240, 32, 5, 32))


class AnalyzerTests(unittest.TestCase):
    def test_complete_grid_and_cross_fitted_schedules(self):
        rows, plan = synthetic_rows()
        result = analyze(rows, plan)
        self.assertEqual(6, result["independent_roots"])
        self.assertEqual(12, result["active_states"])
        self.assertEqual(144, result["evaluation_rows"])
        self.assertEqual("global_1", result["per_root"][0]["loo_stage_actions"][1])
        self.assertEqual("global_0", result["per_root"][0]["loo_stage_actions"][2])
        self.assertEqual("global_1", result["per_state"][0]["chosen_by_scout"])
        self.assertEqual("global_0", result["per_state"][1]["chosen_by_scout"])
        self.assertEqual(.04, round(result["root_policy_means"]["hindsight_oracle"]["gain"], 8))
        self.assertTrue(result["development_only"])
        self.assertFalse(result["confirmatory"])

    def test_oracle_selection_uses_stream_zero_not_one(self):
        rows, plan = synthetic_rows()
        for row in rows:
            if row["root"] == 101 and row["stage"] == 1 and row["stream"] == 1:
                # Reverse held-out action ranking for one state; oracle must
                # still choose the scout winner, not a validation winner.
                row["post_new_ce"] = 1.8 if row["action"] == "global_1" else 1.7
        result = analyze(rows, plan)
        row = next(r for r in result["per_state"] if r["root"] == 101 and r["stage"] == 1)
        self.assertEqual("global_1", row["chosen_by_scout"])

    def test_corrupted_pairing_and_missing_actions_rejected(self):
        rows, plan = synthetic_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(rows + [copy.deepcopy(rows[0])], plan)
        altered = copy.deepcopy(rows)
        altered[0]["batch_stream_hash"] = "unpaired"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            validate_rows(altered, plan)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_rows(rows[:-1], plan)
        altered = copy.deepcopy(rows)
        altered[0]["learner_updates"] += 1
        with self.assertRaisesRegex(ValueError, "same number"):
            validate_rows(altered, plan)
        altered = copy.deepcopy(rows)
        altered[0]["pre_new_ce"] += .01
        with self.assertRaisesRegex(ValueError, "different new-task"):
            validate_rows(altered, plan)

    def test_one_root_is_not_independent_confirmation(self):
        rows, plan = synthetic_rows(roots=(101,))
        result = analyze(rows, plan)
        self.assertFalse(result["screening_gate"]["passed"])
        self.assertEqual("dynamic_necessity_not_established", result["status"])

    def test_no_underlearning_can_be_hidden_by_old_harm(self):
        record = {"pre_new_ce": 1.7, "post_new_ce": 1.8,
                  "pre_old_ce": [1.0, 1.2], "post_old_ce": [1.1, 1.15]}
        result = metrics(record)
        self.assertAlmostEqual(-.1, result["gain"])
        self.assertAlmostEqual(.05, result["old_harm"])
        self.assertAlmostEqual(-.15, result["utility"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

import copy
import sys
import unittest
from pathlib import Path

import torch

HERE, V8 = Path(__file__).parent, Path(__file__).resolve().parents[1] / "v8_multilayer_retention"
sys.path[:0] = [str(HERE), str(V8)]
from environment import DEFAULT
from test_environment import tiny_history
from pressure_env import PressureEnv, run_prefix, run_suffix


def config(tasks=3, **extra):
    result = dict(DEFAULT, tasks=tasks, experts=2, width=4, expert_hidden=8, top_k=2,
                  replay_capacity=64, monitor_capacity=64, batch=4, max_steps=3,
                  interval=1, target_loss=99.0)
    result.update(extra)
    return result


def state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class PressureEnvironmentTests(unittest.TestCase):
    def test_fixed_budget_runs_when_initially_hit_and_early_stop_does_not(self):
        history = tiny_history()
        prefix = run_prefix(PressureEnv(history, 31, config(), capture=True))
        fixed = run_suffix(prefix.clone(), "global_1", 99., "fixed_budget", "full")
        early = run_suffix(prefix.clone(), "global_1", 99., "early_stop", "full")
        self.assertTrue(all(x["steps"] == 3 for x in fixed.completed[1:]))
        self.assertTrue(all(x["steps"] == 0 for x in early.completed[1:]))

    def test_frozen_modules_and_adam_moments_do_not_change(self):
        prefix = run_prefix(PressureEnv(tiny_history(), 32, config(), capture=True))
        env = prefix.clone()
        before_weights = state(env.model.stem)
        before_moments = copy.deepcopy(env.optimizer.state_dict())
        frozen_positions = [i for i, parameter in enumerate(env.model.parameters())
                            if parameter in set(env.model.stem.parameters())]
        frozen_ids = [before_moments["param_groups"][0]["params"][i] for i in frozen_positions]
        run_suffix(env, "global_1", .0001, "fixed_budget", "frozen_stem")
        self.assertTrue(all(torch.equal(v, env.model.stem.state_dict()[k]) for k, v in before_weights.items()))
        after_moments = env.optimizer.state_dict()
        for identifier in frozen_ids:
            before = before_moments["state"].get(identifier, {})
            after = after_moments["state"].get(identifier, {})
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]) if isinstance(before[key], torch.Tensor) else before[key] == after[key])

    def test_clone_and_heldout_labels_do_not_change_trajectory(self):
        history = tiny_history()
        prefix = run_prefix(PressureEnv(history, 33, config(), capture=True))
        left = run_suffix(prefix.clone(), "global_1", .0001, "fixed_budget", "full")
        altered_history = copy.deepcopy(history)
        for task in altered_history:
            task["eval"] = task["eval"].__class__(task["eval"].x, (task["eval"].y + 1) % 5,
                task["eval"].task, task["eval"].ids)
        # Preserve the complete prefix state; only the unused heldout labels differ.
        right = prefix.clone()
        right.history = altered_history
        right = run_suffix(right, "global_1", .0001, "fixed_budget", "full")
        for a, b in zip(left.model.state_dict().values(), right.model.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual([x["steps"] for x in left.completed], [x["steps"] for x in right.completed])

    def test_target_changes_actual_early_stop_not_fixed_budget_truncation(self):
        prefix = run_prefix(PressureEnv(tiny_history(), 34, config(), capture=True))
        easy = run_suffix(prefix.clone(), "global_1", 99., "early_stop", "full")
        hard = run_suffix(prefix.clone(), "global_1", .0001, "early_stop", "full")
        self.assertEqual([0, 0], [x["steps"] for x in easy.completed[1:]])
        self.assertEqual([3, 3], [x["steps"] for x in hard.completed[1:]])

    def test_prefix_retargets_and_suffix_rejects_invalid_conditions(self):
        # Construction begins at 99, but prefix metrics must use its 1.55 goal.
        prefix = run_prefix(PressureEnv(tiny_history(), 35, config(target_loss=99.), capture=True))
        self.assertEqual(1.55, prefix.completed[0]["target_loss"])
        self.assertEqual(prefix.completed[0]["initial_loss"] <= 1.55,
                         prefix.completed[0]["first_hit_step"] == 0)
        with self.assertRaises(ValueError):
            run_suffix(prefix.clone(), "global_1", 1.35, "fixed_budget", "heads_only")
        with self.assertRaises(ValueError):
            run_suffix(prefix.clone(), "global_0", float("nan"), "fixed_budget", "heads_only")

    def test_fixed_budget_cannot_be_finished_early(self):
        env = run_prefix(PressureEnv(tiny_history(), 36, config(), capture=True)).clone()
        env.mode = "fixed_budget"
        with self.assertRaises(RuntimeError):
            env.finish_task()


if __name__ == "__main__":
    unittest.main(verbosity=2)

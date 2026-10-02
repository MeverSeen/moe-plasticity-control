import copy
import unittest

import torch

from data import Batch
from environment import DEFAULT, RetentionEnv, evaluate_endpoints, run_remaining


def tiny_history(tasks=3):
    """Small deterministic 5-way tasks with disjoint IDs and valid local labels."""
    generator = torch.Generator().manual_seed(7)
    history = []
    for task in range(tasks):
        def make(n, offset):
            return Batch(
                torch.rand(n, 3, 32, 32, generator=generator).mul_(2).sub_(1),
                torch.arange(n, dtype=torch.long).remainder(5),
                torch.full((n,), task, dtype=torch.long),
                torch.arange(n, dtype=torch.long) + task * 10_000 + offset,
            )
        history.append({"classes": list(range(task * 5, task * 5 + 5)),
                        "train": make(10, 0), "monitor": make(5, 100),
                        "eval": make(5, 200)})
    return history


def tiny_config(tasks=3):
    config = dict(DEFAULT)
    config.update(tasks=tasks, experts=2, width=4, expert_hidden=8, top_k=2,
                  replay_capacity=64, monitor_capacity=64, batch=4,
                  max_steps=2, interval=1, target_loss=-1.0)
    return config


def all_one_action(env):
    return torch.ones(2, env.c["experts"])


def constant_chooser(env):
    return lambda obs: (all_one_action(env), None)


def assert_optimizer_state_equal(test, first, second):
    first, second = first.state_dict(), second.state_dict()
    test.assertEqual(first["param_groups"], second["param_groups"])
    test.assertEqual(first["state"].keys(), second["state"].keys())
    for key in first["state"]:
        for field, value in first["state"][key].items():
            other = second["state"][key][field]
            if isinstance(value, torch.Tensor):
                test.assertTrue(torch.equal(value, other))
            else:
                test.assertEqual(value, other)


class RetentionEnvironmentTests(unittest.TestCase):
    def test_completed_history_is_retained_and_audited(self):
        history, config = tiny_history(), tiny_config()
        env = RetentionEnv(history, root=14, config=config, capture=True)
        run_remaining(env, constant_chooser(env))
        memory, refs = env.memory.monitor()
        self.assertTrue(env.done)
        self.assertEqual({0, 1, 2}, set(memory.task.tolist()))
        self.assertEqual(len(memory), len(refs))
        report = evaluate_endpoints(env, [task["eval"] for task in history], "dev", "test")
        self.assertEqual(3, len(report["loss_matrix"]))
        for endpoint in range(3):
            self.assertTrue(all(report["loss_matrix"][endpoint][old] is not None
                                for old in range(endpoint + 1)))

    def test_complete_checkpoint_clone_has_identical_future(self):
        history, config = tiny_history(), tiny_config()
        prefix = RetentionEnv(history, root=15, config=config, capture=True)
        prefix.run_task(constant_chooser(prefix))
        left, right = prefix.clone(), prefix.clone()
        run_remaining(left, constant_chooser(left))
        run_remaining(right, constant_chooser(right))
        for a, b in zip(left.model.state_dict().values(), right.model.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(left.completed, right.completed)
        self.assertEqual(left.memory.retained_replay_ids.tolist(), right.memory.retained_replay_ids.tolist())
        left_refs, right_refs = left.memory.monitor()[1], right.memory.monitor()[1]
        self.assertTrue(torch.equal(left_refs, right_refs))
        assert_optimizer_state_equal(self, left.optimizer, right.optimizer)

    def test_offline_audit_does_not_change_completed_rollout(self):
        history, config = tiny_history(), tiny_config()
        env = RetentionEnv(history, root=16, config=config, capture=True)
        run_remaining(env, constant_chooser(env))
        state_before = copy.deepcopy(env.model.state_dict())
        optimizer_before = copy.deepcopy(env.optimizer.state_dict())
        completed_before, cost_before = copy.deepcopy(env.completed), copy.deepcopy(env.cost)
        evaluate_endpoints(env, [task["eval"] for task in history], "dev", "test")
        for key, before in state_before.items():
            self.assertTrue(torch.equal(before, env.model.state_dict()[key]))
        # Compare saved state without tensor truth-value ambiguity.
        clone = RetentionEnv(history, root=16, config=config)
        clone.optimizer.load_state_dict(optimizer_before)
        assert_optimizer_state_equal(self, clone.optimizer, env.optimizer)
        self.assertEqual(completed_before, env.completed)
        self.assertEqual(cost_before, env.cost)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

import copy

import torch
from torch.distributions import Categorical

from control import SharedController, train_controller
from data import Batch
from environment import DEFAULT, FEATURES


def tiny_history(tasks=2):
    generator = torch.Generator().manual_seed(41)
    history = []
    for task in range(tasks):
        def batch(count, offset):
            return Batch(
                torch.rand(count, 3, 32, 32, generator=generator).mul_(2).sub_(1),
                torch.arange(count, dtype=torch.long).remainder(5),
                torch.full((count,), task, dtype=torch.long),
                torch.arange(count, dtype=torch.long) + task * 1_000 + offset,
            )
        history.append({"classes": list(range(task * 5, task * 5 + 5)),
                        "train": batch(10, 0), "monitor": batch(5, 100),
                        "eval": batch(5, 200)})
    return history


def tiny_config():
    config = dict(DEFAULT)
    config.update(tasks=2, experts=2, width=4, expert_hidden=8, top_k=2,
                  replay_capacity=32, monitor_capacity=32, batch=4,
                  max_steps=2, interval=1, target_loss=-1.0)
    return config


class ControllerTests(unittest.TestCase):
    def test_expert_permutation_equivariance(self):
        obs = torch.randn(2, 4, len(FEATURES))
        permutation = torch.tensor([2, 0, 3, 1])
        for coordinated in (False, True):
            policy = SharedController(seed=29, coordinated=coordinated)
            self.assertTrue(torch.allclose(policy(obs[:, permutation]), policy(obs)[:, permutation]))

    def test_chooser_records_joint_factorized_log_probability(self):
        policy = SharedController(seed=30, coordinated=True)
        obs = torch.randn(2, 4, len(FEATURES))
        action, record = policy.chooser(sample=False)(obs)
        expected = Categorical(logits=policy(obs)).log_prob(record["action"]).sum()
        self.assertTrue(torch.equal(record["old_logp"], expected))
        self.assertTrue(torch.equal(action, torch.tensor([0., .5, 1.])[record["action"]]))

    def test_train_controller_runs_paired_branches_and_updates_policy(self):
        initial = SharedController(seed=52, coordinated=True)
        initial_state = copy.deepcopy(initial.state_dict())
        policy, candidates, log, cost = train_controller(
            [(71, tiny_history())], tiny_config(), seed=52,
            rounds=1, group_size=2, coordinated=True,
        )
        self.assertEqual(2, cost["branches"])
        self.assertEqual(2, cost["prefix_updates"])
        # One two-step fixed prefix plus two equal two-step suffix branches.
        self.assertEqual(6, cost["learner_updates"])
        self.assertEqual(1, len(log))
        self.assertEqual(2, log[0]["nonempty_trajectories"])
        self.assertEqual(2, len(log[0]["rewards"]))
        self.assertEqual(2, len(candidates))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in policy.parameters()))
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(initial_state.values(), policy.state_dict().values())))


if __name__ == "__main__":
    unittest.main(verbosity=2)

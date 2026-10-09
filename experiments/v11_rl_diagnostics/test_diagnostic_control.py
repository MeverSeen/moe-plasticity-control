import copy
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V8 = HERE.parent / 'v8_multilayer_retention'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V8))

from control import SharedController, train_controller
from data import Batch
from environment import DEFAULT, FEATURES
from diagnostic_control import _objective, _probe_summary, train_diagnostic


def tiny_history(tasks=2):
    generator = torch.Generator().manual_seed(41)
    history = []
    for task in range(tasks):
        def batch(count, offset):
            return Batch(torch.rand(count, 3, 32, 32, generator=generator).mul_(2).sub_(1),
                         torch.arange(count, dtype=torch.long).remainder(5),
                         torch.full((count,), task, dtype=torch.long),
                         torch.arange(count, dtype=torch.long) + task * 1000 + offset)
        history.append({'classes': list(range(task*5, task*5+5)), 'train': batch(10, 0),
                        'monitor': batch(5, 100), 'eval': batch(5, 200)})
    return history


def tiny_config(tasks=2):
    config = dict(DEFAULT)
    config.update(tasks=tasks, experts=2, width=4, expert_hidden=8, top_k=2,
                  replay_capacity=32, monitor_capacity=32, batch=4,
                  max_steps=2, interval=1, target_loss=-1.0)
    return config


class DiagnosticControllerTests(unittest.TestCase):
    def test_instrumented_checkpoints_match_v8_update(self):
        histories = [(71, tiny_history())]
        baseline, base_candidates, base_logs, base_cost = train_controller(
            histories, tiny_config(), seed=52, rounds=2, group_size=4, coordinated=True)
        policy, candidates, logs, cost = train_diagnostic(
            histories, tiny_config(), seed=52, coordinated=True, rounds=2,
            checkpoints=[0, 1, 2], probe_observations=[])
        self.assertEqual([epoch for epoch, _ in base_candidates], [epoch for epoch, _ in candidates])
        for (_, expected), (_, actual) in zip(base_candidates, candidates):
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]), name)
        for expected, actual in zip(base_logs, logs):
            self.assertEqual(expected['rewards'], actual['rewards'])
            self.assertEqual(expected['decision_counts'], actual['decision_counts'])
            self.assertEqual(expected['loss'], actual['loss'])
        for key in ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples', 'controller_calls', 'branches', 'prefix_updates']:
            self.assertEqual(base_cost[key], cost[key])
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, policy.state_dict()[name]))

    def test_policy_gradient_need_not_follow_near_zero_standardized_loss(self):
        policy = SharedController(seed=33, coordinated=True)
        observations = torch.randn(2, 2, 2, len(FEATURES))
        with torch.no_grad():
            action = policy(observations).argmax(-1)
            old_logp = torch.distributions.Categorical(logits=policy(observations)).log_prob(action).sum((-2, -1))
        trajectories = [[{'obs': observations[0], 'action': action[0], 'old_logp': old_logp[0]}],
                        [{'obs': observations[1], 'action': action[1], 'old_logp': old_logp[1]}]]
        policy_loss, entropy_loss, _, _, _ = _objective(policy, trajectories, torch.tensor([1., -1.]))
        gradients = torch.autograd.grad(policy_loss, tuple(policy.parameters()), allow_unused=True)
        norm = torch.sqrt(sum((grad.float().square().sum() for grad in gradients if grad is not None), torch.tensor(0.)))
        self.assertAlmostEqual(float(policy_loss.detach()), 0.0, places=6)
        self.assertGreater(float(norm), 0.0)
        self.assertTrue(torch.isfinite(entropy_loss))

    def test_reward_components_reconstruct_reward_and_clip_statistics(self):
        policy, candidates, logs, cost = train_diagnostic(
            [(71, tiny_history())], tiny_config(), seed=54, coordinated=False, rounds=1,
            checkpoints=[0, 1], probe_observations=[])
        log = logs[0]
        for reward, parts in zip(log['rewards'], log['reward_parts']):
            self.assertAlmostEqual(reward, sum(parts.values()), places=7)
        self.assertLessEqual(log['total_grad_norm_post_clip'], 1.00001)
        self.assertAlmostEqual(log['clip_reported_pre_norm'], log['total_grad_norm_pre_clip'], places=6)
        self.assertGreaterEqual(log['importance_ratio']['clip_fraction'], 0.)
        self.assertLessEqual(log['importance_ratio']['clip_fraction'], 1.)
        self.assertEqual(3, len(log['action_policy']['sampled_action_frequencies']))

    def test_probe_probability_can_move_without_argmax_change(self):
        policy = SharedController(seed=55, coordinated=False)
        probe = torch.zeros(2, 2, len(FEATURES))
        initial_payload = _probe_summary(policy, [probe])
        initial = initial_payload[1]
        with torch.no_grad():
            policy.output.bias.add_(torch.tensor([.01, 0., -.01]))
        result, _ = _probe_summary(policy, [probe], initial)
        self.assertGreater(result['kl_from_initial'], 0.)
        self.assertEqual(0., result['argmax_changed_fraction'])
        self.assertEqual(1, len(result['per_probe']))
        self.assertEqual(2, len(result['per_probe'][0]['argmax']))

    def test_empty_trace_is_recorded_without_nan(self):
        policy, candidates, logs, cost = train_diagnostic(
            [(71, tiny_history(tasks=1))], tiny_config(tasks=1), seed=56, coordinated=True,
            rounds=1, checkpoints=[0, 1], probe_observations=[])
        self.assertEqual(0, logs[0]['nonempty_trajectories'])
        self.assertIsNone(logs[0]['loss'])
        self.assertTrue(all(reward == 0. for reward in logs[0]['rewards']))


if __name__ == '__main__':
    unittest.main(verbosity=2)

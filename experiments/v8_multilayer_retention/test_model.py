import copy
import unittest

import torch
from torch.nn import functional as F

from model import MultiLayerMoE, ScaledAdam, optimizer_step


class MultiLayerMoETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, top_k=2, seed=9)
        self.x = torch.randn(6, 3, 32, 32)

    def test_routes_are_fixed_after_training_changes(self):
        before = self.model.route_stats(self.x)
        opt = ScaledAdam(self.model, lr=1e-2)
        loss = self.model(self.x, torch.zeros(6, dtype=torch.long)).square().mean()
        loss.backward()
        opt.step(torch.ones(2, 4))
        after = self.model.route_stats(self.x)
        for old, new in zip(before, after):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(tuple(self.model(self.x, torch.zeros(6, dtype=torch.long)).shape), (6, 5))

    def test_seed_initializes_every_component_without_touching_global_rng(self):
        torch.manual_seed(101)
        expected_next = torch.rand(4)
        torch.manual_seed(101)
        first = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, seed=33)
        actual_next = torch.rand(4)
        second = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, seed=33)
        third = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, seed=34)
        self.assertTrue(torch.equal(expected_next, actual_next))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters())))

    def test_route_stat_normalization(self):
        usage, mass = self.model.route_stats(self.x)
        self.assertTrue(torch.allclose(usage.sum(1), torch.full((2,), 2.0)))
        self.assertTrue(torch.allclose(mass.sum(1), torch.ones(2)))

    def test_task_heads_are_selected_per_example(self):
        with torch.no_grad():
            for task, head in enumerate(self.model.heads):
                head.weight.zero_()
                head.bias.fill_(float(task))
        task_ids = torch.tensor([0, 2, 1, 2, 0, 1])
        logits = self.model(self.x, task_ids)
        self.assertTrue(torch.equal(logits[:, 0], task_ids.float()))
        self.assertTrue(torch.equal(logits, task_ids.float().unsqueeze(1).expand(-1, 5)))

    def test_zero_multiplier_blocks_weights_but_updates_adam_moments(self):
        optimizer = ScaledAdam(self.model, lr=1e-2)
        selected, _ = self.model._routes(self.x)
        expert_id = selected[0, 0, 0].item()
        expert = self.model.expert_parameters()[0][expert_id]
        old = [p.detach().clone() for p in expert]
        loss = F.cross_entropy(self.model(self.x, torch.zeros(6, dtype=torch.long)), torch.arange(6) % 5)
        loss.backward()
        multipliers = torch.ones(2, 4)
        multipliers[0, expert_id] = 0.
        optimizer_step(self.model, optimizer, multipliers)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, expert)))
        self.assertTrue(any(p in optimizer.state and optimizer.state[p]["exp_avg"].abs().sum() > 0 for p in expert))

    def test_one_multiplier_matches_plain_adam_and_states_can_be_cloned(self):
        first = self.model
        second = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, top_k=2, seed=9)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        scaled, plain = ScaledAdam(first, lr=1e-3), torch.optim.Adam(second.parameters(), lr=1e-3)
        task_ids = torch.tensor([0, 1, 2, 0, 1, 2])
        labels = torch.tensor([1, 2, 3, 4, 0, 1])
        F.cross_entropy(first(self.x, task_ids), labels).backward()
        F.cross_entropy(second(self.x, task_ids), labels).backward()
        scaled.step(torch.ones(2, 4))
        plain.step()
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(a, b))
        cloned_model = MultiLayerMoE(tasks=3, experts=4, width=8, expert_hidden=12, top_k=2, seed=9)
        cloned_model.load_state_dict(copy.deepcopy(first.state_dict()))
        cloned_opt = ScaledAdam(cloned_model, lr=1e-3)
        cloned_opt.load_state_dict(copy.deepcopy(scaled.state_dict()))
        original_state, copied_state = scaled.state_dict(), cloned_opt.state_dict()
        self.assertEqual(original_state["param_groups"], copied_state["param_groups"])
        self.assertEqual(original_state["state"].keys(), copied_state["state"].keys())
        for key in original_state["state"]:
            for field, value in original_state["state"][key].items():
                other = copied_state["state"][key][field]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, other))
                else:
                    self.assertEqual(value, other)

    def test_multiplier_must_be_finite_and_nonnegative(self):
        optimizer = ScaledAdam(self.model, lr=1e-3)
        for bad in (float("nan"), float("inf"), -0.1):
            multipliers = torch.ones(2, 4)
            multipliers[0, 0] = bad
            with self.assertRaises(ValueError):
                optimizer.step(multipliers)


if __name__ == "__main__":
    unittest.main()

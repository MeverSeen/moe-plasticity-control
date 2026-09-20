import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from common import V14
from metrics import effective_distance
from dataset import convert, endpoint_reward
from offline import expectile_loss, bellman_target, ValueNetwork, train
from structured_policy import Rule
from logged_environment import LoggedAdamW
from logged_environment import LoggedEnvironment
from optimizer import GatedAdamW
from evaluate import choose_iql


def example():
    obs = [torch.zeros(16, 64, 10).tolist(), torch.zeros(10).tolist()]
    action, latent, lp, _, _ = Rule([2] * 16, mode="fixed").act(tuple(torch.tensor(x) for x in obs))
    return {"root": {"id": "train_x", "split": "train"}, "history_completed": True,
            "failure_count": 0, "monitor": {"forgetting": 0.},
            "raw_terminal": {"old_tasks": [{"mean_ce": .5}, {"mean_ce": 1.}]},
            "parameter_distance": {"effective_sq": 2.},
            "trace": [{"duration": 5, "observation": obs, "experts": action.tolist(),
                       "latent": [int(x) for x in latent], "policy_logp": float(lp)},
                      {"duration": 2, "observation": obs, "experts": action.tolist(),
                       "latent": [int(x) for x in latent], "policy_logp": float(lp)}]}


class TestOffline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_distance_matches_materialized_and_rescaling(self):
        torch.manual_seed(1)
        old, new = [], []
        for i in range(6):
            dims = (2, 5) if i % 2 == 0 else (3, 2)
            old.append(torch.randn(*dims, dtype=torch.double))
            new.append(torch.randn(*dims, dtype=torch.double))
        expected = sum(float((new[i + 1] @ new[i] - old[i + 1] @ old[i]).square().sum()) for i in (0, 2, 4))
        self.assertAlmostEqual(effective_distance(new, old)["effective_sq"], expected, places=10)
        equivalent = [x * 4 if i % 2 == 0 else x / 4 for i, x in enumerate(old)]
        self.assertLess(effective_distance(equivalent, old)["effective_sq"], 1e-10)
        self.assertGreater(effective_distance(equivalent, old)["factor_sq"], 0)

    def test_reward_relabelling_keeps_raw_information(self):
        x = example()
        self.assertEqual(endpoint_reward(x, "positive_forgetting"), 0.)
        self.assertEqual(endpoint_reward(x, "old_ce"), -7.5)
        self.assertEqual(endpoint_reward(x, "effective_distance", distance_scale=2), -1.)
        del x["raw_terminal"]
        with self.assertRaises(ValueError): endpoint_reward(x, "old_ce")

    def test_only_true_history_end_is_terminal(self):
        data = convert(example(), "old_ce")
        self.assertFalse(data[0]["terminated"])
        self.assertEqual(data[0]["reward"], 0.)
        self.assertTrue(data[1]["terminated"])
        self.assertEqual(data[1]["duration"], 2)
        self.assertEqual(data[1]["reward"], -7.5)
        target = bellman_target(torch.tensor([0., -7.5]), torch.tensor([0., 1.]), torch.tensor([2., 999.]))
        self.assertTrue(torch.equal(target, torch.tensor([2., -7.5])))

    def test_holdout_and_incomplete_rejected(self):
        x = example(); x["root"]["split"] = "confirmation"
        with self.assertRaises(ValueError): convert(x, "old_ce")

    def test_selection_requires_all_development_roots(self):
        rows = [{"method": "iql_old_ce", "root": "dev_a", "failures": 0,
                 "F": 0., "final_ce": 1., "updates": 1}]
        with self.assertRaises(ValueError): choose_iql(rows, ["dev_a", "dev_b"])
        rows += [{"method": "iql_positive_forgetting", "root": r, "failures": 0,
                  "F": .1, "final_ce": 1., "updates": 10} for r in ["dev_a", "dev_b"]]
        selected, scores = choose_iql(rows, ["dev_a", "dev_b"])
        self.assertEqual(selected, "iql_positive_forgetting")
        self.assertNotIn("iql_old_ce", scores)
        x = example(); x["history_completed"] = False
        with self.assertRaises(ValueError): convert(x, "old_ce")
        x = example(); x["trace"][-1]["duration"] = 0
        with self.assertRaises(ValueError): convert(x, "old_ce")

    def test_expectile_weights_positive_residual_more(self):
        self.assertAlmostEqual(float(expectile_loss(torch.tensor([1.]), .7)), .7, places=6)
        self.assertAlmostEqual(float(expectile_loss(torch.tensor([-1.]), .7)), .3, places=6)

    def test_logged_optimizer_preserves_updates_and_moments(self):
        torch.manual_seed(2)
        aa = [[torch.nn.Parameter(torch.randn(2, 3)) for _ in range(6)] for _ in range(2)]
        bb = [[torch.nn.Parameter(p.detach().clone()) for p in g] for g in aa]
        original, logged = GatedAdamW(aa), LoggedAdamW(bb)
        for multiplier in ([0., .5], [1., .25]):
            for a, b in zip(original.parameters, logged.parameters):
                a.grad = torch.randn_like(a); b.grad = a.grad.clone()
            before = [p.detach().clone() for p in logged.parameters]
            original.step(multiplier); logged.step(multiplier)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(original.parameters, logged.parameters)))
            expected = sum(float((a.detach() - b).square().sum()) for a, b in zip(logged.parameters, before))
            self.assertAlmostEqual(logged.last_displacement["factor_step_sq"], expected, places=10)
            for a, b in zip(original.optimizer.state.values(), logged.optimizer.state.values()):
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_offline_smoke_writes_finite_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory / "data.pt"
            torch.save({"reward": "old_ce", "transitions": convert(example(), "old_ce")}, data)
            summary = train(data, directory / "fit", steps=3, batch_size=2, actor_every=1, warmup=0)
            saved = torch.load(directory / "fit/controller.pt", weights_only=False)
            self.assertEqual(summary["transitions"], 2)
            self.assertTrue(all(torch.isfinite(x).all() for x in saved["policy"].values()))

    def test_logged_environment_raw_ce_and_existing_observations(self):
        from test_environment import FakeNative, data, spec, config
        from environment import Environment

        class FakeLogged(FakeNative):
            def __init__(self):
                super().__init__(); self.parameters = []
                self.optim.zero_grad = lambda: None
                self.last_train_loss = 0.

            def step(self, *args):
                super().step(*args)
                self.last_train_loss = 1.
                self.optim.last_displacement = {"factor_step_sq": .1}

            def snapshot(self):
                result = super().snapshot(); result["optimizer"] = {}; return result

        normal = Environment(FakeNative(), data(), spec(), config()).run(Rule([2] * 16, mode='fixed'))
        e = LoggedEnvironment(FakeLogged(), data(), spec(), config())
        with tempfile.TemporaryDirectory() as folder, patch('logged_environment.effective_distance', return_value={'effective_sq': 1.}):
            actual = e.run_logged(Rule([2] * 16, mode='fixed'), [], Path(folder) / 'result.json', checkpoint_path=Path(folder) / 'emergency.pt')
        self.assertEqual(normal['monitor'], actual['monitor'])
        self.assertEqual(normal['evaluation'], actual['evaluation'])
        self.assertEqual([x['observation'] for x in normal['trace']], [x['observation'] for x in actual['trace']])
        self.assertEqual(len(actual['raw_terminal']['old_tasks']), 2)
        self.assertEqual(len(actual['raw_boundaries']), len(actual['trace']))
        self.assertTrue(all(x['behavior_logp'] == 0 for x in actual['trace']))


if __name__ == "__main__": unittest.main()

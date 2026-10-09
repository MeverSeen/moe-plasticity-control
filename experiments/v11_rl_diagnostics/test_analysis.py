import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze as a


def row(root=1, arm="local", seed=881, epoch=0, mode="sample", draw=0,
        successes=(True, True, True), final_old=.7):
    losses = [[.5, None, None], [.6, .4, None], [final_old, .5, .3]]
    accuracy = [[.5, None, None], [.5, .5, None], [.5, .6, .7]]
    return {"rowkind": "learned", "arm": arm, "policy_seed": seed, "epoch": epoch,
            "mode": mode, "draw": draw if mode == "sample" else None, "root": root,
            "tasks": [{"task": i, "success": success} for i, success in enumerate(successes)],
            "loss_matrix": losses, "accuracy_matrix": accuracy,
            "target_loss": .6,
            "cost": {"learner_updates": 10 + (draw or 0), "monitor_examples": 20}}


class AnalysisTests(unittest.TestCase):
    def test_forgetting_uses_diagonal_and_is_not_final_absolute_loss(self):
        metric = a.row_metrics(row(final_old=.9))
        self.assertAlmostEqual(metric["final_positive_forgetting"], (.4 + .1) / 2)
        self.assertAlmostEqual(metric["final_mean_loss"], (.9 + .5 + .3) / 3)
        # The task reached target when learned, then task 0 missed at the end.
        self.assertTrue(metric["diagonal_all_reached"])
        self.assertFalse(metric["final_all_reached"])

    def test_failed_tasks_remain_in_root_summary(self):
        good, failed = row(root=1, draw=0), row(root=2, draw=0, successes=(True, False, True))
        groups = a.aggregate_roots([good, failed])
        summary = a.summarize(groups)[("learned", "local", 881, 0, "sample")]
        self.assertEqual(summary["roots"], 2)
        self.assertEqual(summary["metrics"]["sequence_success"]["mean"], .5)
        self.assertEqual(summary["metrics"]["failed_tasks"]["mean"], .5)

    def test_sample_draws_average_inside_root_before_cross_root_summary(self):
        first, second = row(root=1, draw=0, final_old=.7), row(root=1, draw=1, final_old=.9)
        other = row(root=2, draw=0, final_old=.7)
        groups = a.aggregate_roots([first, second, other])
        roots = groups[("learned", "local", 881, 0, "sample")]
        self.assertEqual(len(roots), 2)
        self.assertEqual(roots[0]["draws"], 2)
        # Root 1 mean F=(.15+.25)/2=.20; root 2 F=.15; root mean=.175.
        summary = a.summarize(groups)[("learned", "local", 881, 0, "sample")]
        self.assertAlmostEqual(summary["metrics"]["final_positive_forgetting"]["mean"], .175)

    def test_policy_seeds_pool_inside_root_after_sample_draws(self):
        rows = [row(root=1, seed=881, draw=0, final_old=.7), row(root=1, seed=881, draw=1, final_old=.9),
                row(root=1, seed=882, draw=0, final_old=.9), row(root=1, seed=882, draw=1, final_old=.7),
                row(root=2, seed=881, draw=0, final_old=.7), row(root=2, seed=881, draw=1, final_old=.7),
                row(root=2, seed=882, draw=0, final_old=.7), row(root=2, seed=882, draw=1, final_old=.7)]
        pooled = a.aggregate_policy_seeds(a.aggregate_roots(rows))
        roots = pooled[("learned", "local", 0, "sample")]
        self.assertEqual(len(roots), 2)  # never eight draw/seed pseudo-replicates
        self.assertEqual(roots[0]["policy_seeds"], 2)
        summary = a.summarize(pooled)[("learned", "local", 0, "sample")]
        self.assertAlmostEqual(summary["metrics"]["final_positive_forgetting"]["mean"], .175)

    def test_cli_development_only_artifacts(self):
        greedy = row(mode="greedy", draw=None)
        sampled = row(mode="sample", draw=0)
        with tempfile.TemporaryDirectory() as folder:
            run, out = Path(folder) / "run", Path(folder) / "out"
            (run / "training").mkdir(parents=True)
            (run / "locked_protocol.json").write_text(json.dumps({"development_only": True}))
            (run / "evaluations.json").write_text(json.dumps([greedy, sampled]))
            (run / "training" / "local-881.json").write_text(json.dumps({"log": [
                {"round": 1, "reward_std": 1.0, "policy_gradient_grad_norm": .2, "entropy_grad_norm": .1,
                 "probe": {"mean_probabilities": [.2, .3, .5], "argmax_changed_fraction": 0.}},
                {"round": 2, "reward_std": .5, "policy_gradient_grad_norm": .3, "entropy_grad_norm": .1,
                 "probe": {"mean_probabilities": [.2, .4, .4], "argmax_changed_fraction": .1}}], "cost": {"seconds": 1}}))
            with patch("sys.argv", ["analyze.py", "--run", str(run), "--out", str(out)]): a.main()
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "研究诊断报告.md").exists())
            self.assertTrue((out / "training_curves.png").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertIn("local", summary["development_greedy_selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

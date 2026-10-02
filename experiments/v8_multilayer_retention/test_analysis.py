import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze as a


def row(root, method, seed, losses, successes=(True, True, True), target=1.0):
    k = len(losses)
    accuracy = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1): accuracy[i][j] = .5
    return {"split": "audit", "root": root, "method": method, "policy_seed": seed,
            "tasks": [{"task": i, "initial_loss": 1.2, "terminal_loss": losses[i][i], "success": successes[i], "steps": 1} for i in range(k)],
            "loss_matrix": losses, "accuracy_matrix": accuracy, "target_loss": target,
            "cost": {"learner_updates": 1, "train_examples": 2, "monitor_examples": 3, "audit_examples": 4, "controller_calls": 5, "seconds": 6}}


class AnalysisTests(unittest.TestCase):
    def test_forgetting_uses_original_diagonal_not_later_checkpoint(self):
        losses = [[.5, None, None], [.7, .4, None], [.9, .8, .3]]
        m = a.row_metrics(row(1, "m", 1, losses))
        self.assertIsNone(m["trajectory"][0]["positive_forgetting"])
        self.assertIsNone(m["trajectory"][0]["signed_backward_change"])
        self.assertAlmostEqual(m["trajectory"][1]["positive_forgetting"], .2)
        self.assertAlmostEqual(m["trajectory"][2]["positive_forgetting"], .4)  # mean(.9-.5, .8-.4)
        self.assertAlmostEqual(m["final_positive_forgetting"], .4)

    def test_failures_stay_in_root_summary(self):
        good = row(1, "m", 1, [[.5, None, None], [.6, .5, None], [.6, .6, .5]])
        bad = row(2, "m", 1, [[.5, None, None], [.5, .5, None], [.5, .5, .5]], successes=(True, False, True))
        groups = a.aggregate_roots([good, bad]); summary = a.summarize_group(groups[("audit", "m")])
        self.assertEqual(summary["roots"], 2)
        self.assertEqual(summary["sequence_success"]["mean"], .5)
        self.assertEqual(summary["failed_tasks"]["mean"], .5)

    def test_policy_replicas_average_before_root_pairing(self):
        a1 = row(1, "a", 1, [[.5, None, None], [.6, .5, None], [.6, .6, .5]])
        a2 = row(1, "a", 2, [[.5, None, None], [.8, .5, None], [.8, .8, .5]])
        b1 = row(1, "b", 1, [[.5, None, None], [.7, .5, None], [.7, .7, .5]])
        grouped = a.aggregate_roots([a1, a2, b1])
        self.assertEqual(grouped[("audit", "a")][0]["replicas"], 2)
        comparison = a.compare_groups(grouped[("audit", "a")], grouped[("audit", "b")])
        # F(a)=(.1 + .3)/2; F(b)=.2, so one root gives zero paired gap.
        self.assertAlmostEqual(comparison["final_positive_forgetting_all_roots"]["mean"], 0.)

    def test_cli_writes_required_artifacts_and_reads_pilot_protocol(self):
        record = row(1, "m", 1, [[.5, None, None], [.6, .5, None], [.6, .6, .5]])
        with tempfile.TemporaryDirectory() as folder:
            run, out = Path(folder) / "run", Path(folder) / "analysis"
            run.mkdir()
            (run / "results.json").write_text(json.dumps([record]))
            (run / "protocol.json").write_text(json.dumps({"status": "pilot", "config": {"tasks": 3}, "selection": {"source": "dev"}}))
            with patch("sys.argv", ["analyze.py", "--run", str(run), "--out", str(out)]):
                a.main()
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "实验报告.md").exists())
            self.assertTrue((out / "results.png").exists())
            self.assertEqual(json.loads((out / "summary.json").read_text())["protocol"]["status"], "pilot")

    def test_protocol_grid_rejects_missing_or_duplicate_replicas(self):
        records = [row(1, "global_0", None, [[.5, None, None], [.6, .5, None], [.6, .6, .5]])]
        protocol = {"roots": {"audit": [1]}, "expected_policy_seeds": {"global_0": [None], "learned": [881]}}
        with self.assertRaisesRegex(ValueError, "incomplete result grid"):
            a.validate_protocol_coverage(records, protocol)
        duplicate = records + [dict(records[0])]
        with self.assertRaisesRegex(ValueError, "incomplete result grid"):
            a.validate_protocol_coverage(duplicate, {"roots": {"audit": [1]}, "expected_policy_seeds": {"global_0": [None]}})

    def test_missing_or_nonfinite_past_matrix_cells_are_rejected(self):
        valid = row(1, "m", 1, [[.5, None, None], [.6, .5, None], [.6, .6, .5]])
        missing = json.loads(json.dumps(valid)); missing["loss_matrix"][2][0] = None
        with self.assertRaisesRegex(ValueError, "loss_matrix\\[2\\]\\[0\\]"):
            a.row_metrics(missing)
        nonfinite = json.loads(json.dumps(valid)); nonfinite["accuracy_matrix"][2][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "accuracy_matrix\\[2\\]\\[1\\]"):
            a.row_metrics(nonfinite)


if __name__ == "__main__": unittest.main(verbosity=2)

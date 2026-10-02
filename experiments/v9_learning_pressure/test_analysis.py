import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_pressure as a


def record(root=88410, method="global_0", target=1.35, mode="early_stop", scope="full",
           suffix_hits=(True, True, True, True), diagonal=(1.50, 1.30, 1.30, 1.30, 1.30)):
    losses = [[None] * 5 for _ in range(5)]
    accuracy = [[None] * 5 for _ in range(5)]
    for learned in range(5):
        for tested in range(learned + 1):
            losses[learned][tested] = diagonal[tested] + (.10 if learned == 4 and tested < 4 else 0.)
            accuracy[learned][tested] = .6
    tasks = []
    for i in range(5):
        hit = True if i == 0 else suffix_hits[i - 1]
        task_target = 1.55 if i == 0 else target
        tasks.append({"task": i, "min_monitor_loss": 1.2, "terminal_monitor_loss": 1.2 if hit else 1.6,
                      "min_train_loss": 1.1, "min_eval_loss": diagonal[i], "initial_train_loss": 1.7,
                      "initial_eval_loss": 1.7, "ever_hit": hit, "first_hit_step": 5 if hit else None,
                      "first_hit_eval_loss": 1.3 if hit else None, "target_loss": task_target,
                      "steps": 5 if mode == "early_stop" and hit else 80, "success": hit})
    return {"root": root, "method": method, "mode": mode, "scope": scope, "target_loss": target,
            "prefix_target": 1.55, "prefix_fingerprint": f"prefix-{root}", "tasks": tasks, "loss_matrix": losses, "accuracy_matrix": accuracy,
            "cost": {"learner_updates": 10, "monitor_examples": 20}}


class AnalysisTests(unittest.TestCase):
    def test_fixed_diagonal_forgetting_uses_all_old_tasks(self):
        row = record()
        row["loss_matrix"][4][:4] = [1.7, 1.1, 1.5, 1.3]
        metrics = a.row_metrics(row)
        # Relative to [1.5, 1.3, 1.3, 1.3], not a later checkpoint.
        self.assertAlmostEqual(metrics["final_positive_forgetting"], (0.2 + 0 + .2 + 0) / 4)
        self.assertAlmostEqual(metrics["final_signed_backward_change"], (.2 - .2 + .2 + 0) / 4)

    def test_suffix_failure_is_retained_and_monitor_is_not_eval(self):
        row = record(suffix_hits=(True, False, True, True), diagonal=(1.5, 1.2, 1.6, 1.2, 1.2))
        metric = a.row_metrics(row)
        self.assertFalse(metric["suffix_monitor_ever_all"])
        self.assertFalse(metric["suffix_monitor_terminal_all"])
        self.assertFalse(metric["suffix_heldout_eval_diagonal_all"])
        group, _ = a.aggregate([row])
        self.assertEqual(group[("early_stop", "full", "global_0", 1.35)]["roots"], 1)
        self.assertEqual(group[("early_stop", "full", "global_0", 1.35)]["metrics"]["suffix_monitor_ever_all"]["mean"], 0.)

    def test_missing_matrix_cells_are_rejected(self):
        row = record(); row["loss_matrix"][4][2] = None
        with self.assertRaisesRegex(ValueError, r"loss_matrix\[4\]\[2\]"):
            a.row_metrics(row)

    def test_protocol_rejects_missing_duplicate_and_unexpected_cells(self):
        protocol = {"config": {"max_steps": 80, "interval": 5}, "roots": [88410, 88411], "expected_cells": [
            {"mode": "early_stop", "scope": "full", "method": "global_0", "target_loss": 1.35}]}
        with self.assertRaisesRegex(ValueError, "missing"):
            a.validate_protocol_coverage([record(88410)], protocol)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            a.validate_protocol_coverage([record(88410), record(88410), record(88411)], protocol)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            a.validate_protocol_coverage([record(88410), record(88411), record(88412)], protocol)

    def test_new_metrics_distinguish_prefix_monitor_and_heldout_eval(self):
        row = record(diagonal=(1.60, 1.2, 1.2, 1.2, 1.2))
        metric = a.row_metrics(row)
        self.assertTrue(metric["prefix_monitor_ever_hit"])
        self.assertFalse(metric["prefix_heldout_eval_hit"])
        self.assertTrue(metric["suffix_heldout_eval_diagonal_all"])

    def test_cli_writes_artifacts_without_overwrite(self):
        row = record()
        protocol = {"status": "pilot", "config": {"max_steps": 80, "interval": 5}, "roots": [88410], "expected_cells": [
            {"mode": "early_stop", "scope": "full", "method": "global_0", "target_loss": 1.35}]}
        with tempfile.TemporaryDirectory() as folder:
            run, out = Path(folder) / "run", Path(folder) / "analysis"
            run.mkdir(); (run / "results.json").write_text(json.dumps([row])); (run / "protocol.json").write_text(json.dumps(protocol))
            with patch("sys.argv", ["analyze_pressure.py", "--run", str(run), "--out", str(out)]): a.main()
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "实验报告.md").exists())
            self.assertTrue((out / "pressure.png").exists())
            with self.assertRaises(FileExistsError):
                with patch("sys.argv", ["analyze_pressure.py", "--run", str(run), "--out", str(out)]): a.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)

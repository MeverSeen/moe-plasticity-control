import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyze_headroom import analyze, native_bundle, root_summary, validate


def record(candidate="fixed", stream="0", root="r0", ev=.2, mon=None, success=True, updates=10):
    return {"root_id": root, "comparison_id": "whole", "checkpoint_id": "prefix-" + root,
            "context_id": "ctx", "stream_id": stream, "independent_future": stream != "0",
            "candidate_id": candidate, "candidate_kind": "fixed", "candidate_spec": {"name": candidate},
            "role": "calibration", "completed": True, "success": success, "n_fail": int(not success),
            "eval_forgetting": ev, "monitor_forgetting": ev if mon is None else mon,
            "path_updates": updates, "actual_updates": updates, "wall_seconds": 1.0}


def bundle(rows):
    return {"schema_version": 1, "protocol_id": "v13-test-only", "metric_id": "synthetic-test-metric", "records": rows}


class HeadroomTests(unittest.TestCase):
    def test_independent_future_can_reverse_in_sample_gain(self):
        rows = [record(ev=.2), record("other", ev=.1), record(stream="1", ev=.2), record("other", "1", ev=.5)]
        out = analyze(bundle(rows), "fixed", "0", "1")
        future = out["independent_future"]
        self.assertEqual(future["status"], "available")
        self.assertAlmostEqual(future["monitor_selection"]["common_success_eval_delta"]["mean_of_root_means"], .3)
        self.assertEqual(future["choices"][0]["monitor_choice"], "other")

    def test_monitor_and_privileged_selection_differ(self):
        rows = [record(ev=.1, mon=.5), record("other", ev=.4, mon=.1),
                record(stream="1", ev=.1), record("other", "1", ev=.4)]
        future = analyze(bundle(rows), "fixed", "0", "1")["independent_future"]
        self.assertEqual(future["choices"][0]["monitor_choice"], "other")
        self.assertEqual(future["choices"][0]["privileged_choice"], "fixed")

    def test_future_failure_not_dropped(self):
        rows = [record(ev=.2), record("other", ev=.1), record(stream="1"), record("other", "1", ev=.01, success=False)]
        result = analyze(bundle(rows), "fixed", "0", "1")["independent_future"]["monitor_selection"]
        self.assertEqual(result["n_matched"], 1)
        self.assertEqual(result["left_successes"], 0)
        self.assertEqual(result["right_successes"], 1)
        self.assertEqual(result["common_successes"], 0)
        self.assertIsNone(result["common_success_eval_delta"]["mean_of_root_means"])

    def test_missing_panel_or_no_independence_is_unknown(self):
        rows = [record(), record("other"), record(stream="1")]
        output = analyze(bundle(rows), "fixed", "0", "1")["independent_future"]
        self.assertIn("unknown", output["status"])
        self.assertEqual(output["skipped"]["incomplete_candidate_panel_or_reference"], 1)
        rows.append(record("other", "1"))
        for row in rows:
            row["independent_future"] = False
        output = analyze(bundle(rows), "fixed", "0", "1")["independent_future"]
        self.assertEqual(output["skipped"]["future_independence_not_declared"], 1)

    def test_same_stream_not_independent(self):
        with self.assertRaises(ValueError):
            analyze(bundle([record()]), "fixed", "0", "0")
        self.assertIn("unknown", analyze(bundle([record()]), "fixed")["independent_future"]["status"])

    def test_different_labels_can_have_common_minimizer(self):
        rows = [record(updates=1), record("other", updates=2),
                record(root="r1", updates=2), record("other", root="r1", updates=1)]
        result = analyze(bundle(rows), "fixed")["within_stream_descriptive"]
        self.assertEqual(result["hindsight_choice_counts_with_cost_tiebreak"], {"fixed": 1, "other": 1})
        self.assertEqual(result["common_minimum_forgetting_candidates"], ["fixed", "other"])

    def test_root_weights_do_not_count_checkpoints_as_independent(self):
        value = root_summary([("r0", 1)] * 10 + [("r1", 3)])
        self.assertEqual(value["n_observations"], 11)
        self.assertEqual(value["n_roots"], 2)
        self.assertEqual(value["mean_of_root_means"], 2)

    def test_missing_metrics_not_zero(self):
        row = record("other")
        row["eval_forgetting"] = row["monitor_forgetting"] = None
        result = analyze(bundle([record(), row]), "fixed")
        self.assertEqual(result["coverage"]["other"]["missing_eval"], 1)
        self.assertIsNone(result["candidate_vs_reference"]["other"]["common_success_eval_delta"]["mean_of_root_means"])

    def test_invalid_and_incompatible_inputs_rejected(self):
        for patch in [{"role": "confirmation"}, {"eval_forgetting": math.nan},
                      {"eval_forgetting": True}, {"path_updates": -1}, {"path_updates": .5},
                      {"success": True, "completed": False}, {"n_fail": -1}, {"independent_future": "false"}]:
            row = record()
            row.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate(bundle([row]))
        for patch in [{"checkpoint_id": "different"}, {"context_id": "different"}]:
            row = record("other")
            row.update(patch)
            with self.assertRaises(ValueError):
                validate(bundle([record(), row]))
        with self.assertRaises(ValueError):
            validate(bundle([record(), record()]))
        row = record(stream="1")
        row["candidate_spec"] = {"different": True}
        with self.assertRaises(ValueError):
            validate(bundle([record(), row]))

    def test_roles_analyzed_separately(self):
        row = record(root="r1")
        row["role"] = "development"
        with self.assertRaises(ValueError):
            validate(bundle([record(), row]))

    def test_all_failed_selection_explicitly_unavailable(self):
        rows = [record(success=False), record(stream="1")]
        result = analyze(bundle(rows), "fixed", "0", "1")["independent_future"]
        self.assertEqual(result["skipped"]["no_feasible_monitor_selection"], 1)
        self.assertEqual(result["monitor_selection"]["right_only"], 1)

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "input.json"
            source.write_text(json.dumps(bundle([record(), record("other")])))
            command = [sys.executable, str(Path(__file__).with_name("analyze_headroom.py")), "--input", str(source),
                       "--reference", "fixed", "--output-dir", str(temp / "out")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((temp / "out/headroom.json").read_text())["n_records"], 2)
            self.assertTrue((temp / "out/收益空间诊断.md").exists())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_native_adapter_conservative_semantics(self):
        endpoint = {"protocol_revision": "v13-test", "phase": "calibration", "root": {"id": "r0", "tasks": ["A", "B", "C"]},
                    "checkpoint": {"base_revision": "model", "prefix_sha256": "full-prefix"},
                    "candidate": {"name": "fixed", "base_indices": [2, 2], "execution": "greedy"},
                    "batch_stream": {"id": 0, "independent_future": False},
                    "tasks": [{"task": x, "target": .1} for x in ["A", "B", "C"]],
                    "history_completed": True, "history_success": True, "n_fail": 0,
                    "monitor": {"forgetting": .2, "full_memory": False}, "evaluation": {"forgetting": .3},
                    "cost_actual": {"updates": 2}, "cost_path": {"updates": 5}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "endpoint.json"
            path.write_text(json.dumps(endpoint))
            adapted = native_bundle([path])
            validate(adapted)
            self.assertIsNone(adapted["records"][0]["monitor_forgetting"])
            self.assertEqual(adapted["records"][0]["actual_updates"], 2)
            self.assertEqual(adapted["records"][0]["path_updates"], 5)
            endpoint["phase"] = "confirmation"
            path.write_text(json.dumps(endpoint))
            self.assertEqual(native_bundle([path])["records"], [])
            endpoint["phase"] = "calibration"
            del endpoint["history_completed"]
            path.write_text(json.dumps(endpoint))
            adapted = native_bundle([path])
            self.assertEqual(adapted["records"], [])
            self.assertEqual(adapted["adapter_audit"][0]["status"], "unusable_missing_fields")

    def test_partial_native_history_does_not_invent_failure_target(self):
        obj = {"protocol_revision": "test", "phase": "calibration", "root": {"id": "r0", "tasks": ["A", "B"]},
               "checkpoint": {"base_revision": "model", "prefix_sha256": "prefix"},
               "candidate": {"name": "fixed"}, "batch_stream": {"id": 0, "independent_future": False},
               "history_completed": True, "history_success": False, "n_fail": 1,
               "tasks": [{"task": "A", "target": .1}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "p.json"
            path.write_text(json.dumps(obj))
            result = native_bundle([path])
            self.assertEqual(len(result["records"]), 0)
            self.assertIn("complete target_spec (including unattempted tasks)", result["adapter_audit"][0]["missing"])

    def test_empty_input_is_not_research_result(self):
        result = analyze(bundle([]), "fixed")
        self.assertEqual(result["n_records"], 0)
        self.assertIsNone(result["within_stream_descriptive"]["common_minimum_forgetting_candidates"])
        self.assertIn("unknown", result["independent_future"]["status"])

    def test_cli_cannot_write_inside_native_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, str(Path(__file__).with_name("analyze_headroom.py")), "--native-dir", temp,
                       "--reference", "fixed", "--output-dir", str(Path(temp) / "nested")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside", result.stderr)


if __name__ == "__main__":
    unittest.main()

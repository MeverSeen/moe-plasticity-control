"""Regression guards for v11 pairing and original-v8 replay checks."""
import json
import unittest

from run_diagnostics import V8, action_seed, compare_original, total_cost


class RunnerTests(unittest.TestCase):
    def test_action_randomness_is_externally_keyed(self):
        # Candidate/architecture are intentionally absent; task-local streams
        # cannot be shifted by a preceding task's different stopping time.
        first = action_seed(88520, 1, 2, 10)
        action_seed(88520, 1, 1, 75)
        self.assertEqual(first, action_seed(88520, 1, 2, 10))
        self.assertEqual(len({action_seed(88520, d, t, s)
            for d in range(3) for t in range(1,5) for s in range(0,80,5)}), 192)

    def test_replay_check_detects_reward_or_branch_length_changes(self):
        path = V8/'results/pilot/models/learned_local_881_training.json'
        logs = json.loads(path.read_text())['log']
        self.assertTrue(compare_original(logs, 'local', 881)['passed'])
        logs[1]['rewards'][0] += .01
        self.assertFalse(compare_original(logs, 'local', 881)['passed'])
        logs = json.loads(path.read_text())['log']
        logs[1]['decision_counts'][0] += 1
        self.assertFalse(compare_original(logs, 'local', 881)['passed'])

    def test_physical_cost_includes_probe_and_each_real_rollout_once(self):
        costs = total_cost([{'learner_updates': 100}],
            [{'cost': {'learner_updates': 40}}, {'cost': {'learner_updates': 40}}],
            [{'learner_updates': 10}])
        self.assertEqual(costs['total']['learner_updates'], 190)
        self.assertEqual(costs['development_evaluation']['learner_updates'], 80)


if __name__ == '__main__':
    unittest.main()

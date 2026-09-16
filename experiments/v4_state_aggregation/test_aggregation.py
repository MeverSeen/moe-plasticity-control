"""Mechanism checks for learner-state querying and continuation comparisons."""
import copy
import unittest

import numpy as np
import torch

import dagger as d


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(d.b.base.CONFIG, max_steps=22, target=1e-8, levels=[1e-8])
        cls.item = d.b.make_bank([459901], cls.config)[0]

    def env(self):
        return d.b.new_env(self.item, self.config)

    def test_sampling_uses_actual_learner_trajectory(self):
        states, _ = d.trajectory_snapshots(self.item, self.config, lambda env: 1, limit=3)
        replay = self.env()
        for state in states:
            while replay.steps < state.steps:
                replay.advance(1)
            self.assertEqual(d.fingerprint(state), d.fingerprint(replay))
        self.assertLessEqual(len(states), 3)

    def test_union_deduplicates_same_complete_state(self):
        policies = {'progress': lambda env: 2, 'feedback': lambda env: 2}
        states, _ = d.learned_snapshots(self.item, self.config, policies, 3)
        self.assertEqual(len(states), 3)
        self.assertTrue(all(sources == ['progress', 'feedback'] for _, sources in states))

    def test_fingerprint_distinguishes_momentum_and_observation_history(self):
        a = self.env()
        clone = d.b.clone_env(a)
        self.assertEqual(d.fingerprint(a), d.fingerprint(clone))
        clone.adam_m.add_(.01)
        self.assertNotEqual(d.fingerprint(a), d.fingerprint(clone))
        clone = d.b.clone_env(a)
        clone.previous_loss *= 2
        self.assertNotEqual(d.fingerprint(a), d.fingerprint(clone))

    def test_branch_query_preserves_the_original_state(self):
        a = self.env()
        a.advance(2)
        before = d.fingerprint(a)
        result, cost = d.query(a, 3, 2, 'unit', ['test'])
        self.assertEqual(before, d.fingerprint(a))
        self.assertEqual(len(result['branches']), 5)
        self.assertTrue(all(len(rows) == 2 for rows in result['branches']))
        self.assertGreater(cost['branch_updates'], 0)

    def test_constant_callback_matches_fixed_continuation(self):
        a = self.env()
        fixed, _ = d.query(a, 2, 2, 'unit', [], 'fixed')
        adaptive, _ = d.query(a, lambda env: 2, 2, 'unit', [], 'adaptive')
        self.assertEqual(fixed['branches'], adaptive['branches'])

    def test_constant_action_means_entire_remaining_rollout(self):
        a = self.env()
        result, _ = d.query(a, 4, 1, 'unit', [], 'constant')
        for action in range(5):
            branch = d.b.resample_future(a, 0)
            d.b.finish(branch, action)
            expected = d.b.endpoint(branch, d.b.evaluator_reference(a))
            for key in expected:
                self.assertEqual(result['branches'][action][0][key], expected[key])

    def test_matched_collection_has_exact_query_count(self):
        entries, _ = d.behavior_snapshots(self.item, self.config, 7, 1)
        self.assertEqual(len(entries), 7)
        self.assertTrue(all(sources[0].startswith('nonlearned_') for _, sources in entries))

    def test_eval_labels_cannot_change_collection_or_policy_inputs(self):
        changed = copy.deepcopy(self.item)
        for task in changed[0]['tasks']:
            task['test'][1].fill_(1e9)
        a, _ = d.trajectory_snapshots(self.item, self.config, lambda env: 2, 3)
        c, _ = d.trajectory_snapshots(changed, self.config, lambda env: 2, 3)
        self.assertEqual([d.fingerprint(v) for v in a], [d.fingerprint(v) for v in c])
        model = d.b.Scorer()
        mean, std = np.zeros(14), np.ones(14)
        x = a[-1].observe(True).numpy()
        y = x.copy(); y[12:] += 100
        torch.testing.assert_close(d.b.features(x, mean, std, False),
                                   d.b.features(y, mean, std, False), rtol=0, atol=0)

    def test_splits_are_new_and_disjoint_including_pilot(self):
        seen = set(range(310000, 350400)) | set(range(230000, 250080))
        for split in [d.SPLITS, d.PILOT_SPLITS]:
            for roots in split.values():
                self.assertFalse(seen.intersection(roots))
                seen.update(roots)


if __name__ == '__main__':
    unittest.main(verbosity=2)

import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import branching as b
import diagnose as d
import learning as l


class LearningTests(unittest.TestCase):
    def test_legacy_objective_matches_frozen_recipe(self):
        states = l.positive_states(690000, 3)
        data = l.prepare(states, 'legacy')
        torch.manual_seed(2); model = b.Scorer()
        tx = b.features(data['x'], data['mean'], data['std'], True)
        ph, pf = model(tx); target, valid = data['target'], data['valid']
        denom = valid.sum(1).clamp_min(1)
        hit = F.binary_cross_entropy_with_logits(ph, data['hit'], reduction='none').mean(1)
        absolute = (F.smooth_l1_loss(pf, target, reduction='none') * valid).sum(1) / denom
        pc = pf - (pf * valid).sum(1, keepdim=True) / denom[:, None]
        tc = target - (target * valid).sum(1, keepdim=True) / denom[:, None]
        relative = (F.smooth_l1_loss(pc, tc, reduction='none') * valid).sum(1) / denom
        expected = ((hit + absolute + relative) * data['weights']).mean()
        torch.testing.assert_close(l.objective(model, tx, data, 'legacy'), expected)

    def test_raw_targets_are_conditional_arithmetic_mean(self):
        states = l.positive_states(690000, 1)
        states[0]['branches'][0] = [dict(success=True, forgetting=.01), dict(success=True, forgetting=.09),
                                    dict(success=False, forgetting=100.)]
        for state in states:
            for action in range(5):
                if state is not states[0] or action != 0:
                    state['branches'][action] *= 3
        raw = l.prepare(states, 'raw_mse')
        self.assertAlmostEqual(float(raw['target'][0, 0]), 5., places=5)
        self.assertAlmostEqual(float(raw['hit'][0, 0]), 2/3, places=6)

    def test_failure_only_targets_have_finite_gradient(self):
        states = l.positive_states(690000, 2)
        for state in states:
            for branches in state['branches']:
                for row in branches: row['success'] = False
        for kind in l.LOSSES:
            data = l.prepare(states, kind); model = b.Scorer()
            tx = b.features(data['x'], data['mean'], data['std'], True)
            loss = l.objective(model, tx, data, kind); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_feedback_does_not_enter_progress_features(self):
        states = l.positive_states(690000, 3)
        data = l.prepare(states, 'legacy')
        original = b.features(data['x'], data['mean'], data['std'], False)
        x = data['x'].copy(); x[:, 12:] = 1e8
        changed = b.features(x, data['mean'], data['std'], False)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_positive_control_is_balanced_conditionally_on_progress(self):
        states = l.positive_states(690000, 4)
        for i in range(0, len(states), 5):
            group = states[i:i+5]
            self.assertEqual([s['label'] for s in group], list(range(5)))
            for state in group:
                np.testing.assert_array_equal(state['x'][:12], group[0]['x'][:12])
                f = [s[0]['forgetting'] for s in state['branches']]
                self.assertEqual(int(np.argmin(f)), state['label'])

    def test_selection_distinguishes_raw_and_saturated_metrics(self):
        def metrics(fs):
            return l.endpoint_metrics([dict(success=True, forgetting=f, cost=b.objective(1, f), work=1) for f in fs])
        spiky, stable = metrics([0., .1]), metrics([.02, .02])
        self.assertLess(l.selection_key(spiky, 'branch_J'), l.selection_key(stable, 'branch_J'))
        self.assertGreater(l.selection_key(spiky, 'branch_F'), l.selection_key(stable, 'branch_F'))
        failed = dict(stable, reach=.99, raw_forgetting=0., cost=0.)
        self.assertLess(l.selection_key(stable, 'closed_F'), l.selection_key(failed, 'closed_F'))

    def test_tiny_subset_uses_distinct_nontrivial_roots(self):
        states = l.positive_states(690000, 5)
        selected = d.tiny_subset(states, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({s['root'] for s in selected}), 3)

    def test_new_splits_are_disjoint(self):
        seen = set()
        for splits in [d.SPLITS, d.PILOT]:
            for roots in splits.values():
                self.assertFalse(seen.intersection(roots)); seen.update(roots)
        self.assertGreater(min(seen), 549999)


if __name__ == '__main__': unittest.main(verbosity=2)

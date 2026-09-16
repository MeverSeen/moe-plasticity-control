"""Check weighting, failure retention and independent-future scoring."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze as a


class AnalysisTests(unittest.TestCase):
    def test_roots_not_checkpoints_are_statistical_units(self):
        roots = np.array([1, 1, 1, 2])
        values = np.array([0., 0., 0., 1.])
        self.assertEqual(a.clustered(values, roots).tolist(), [0., 1.])
        self.assertEqual(a.interval(a.clustered(values, roots))['mean'], .5)

    def test_failures_remain_in_reach_and_cost(self):
        first = [np.array([[[0.]], [[1.]]]), np.array([[[0.]], [[.03]]]),
                 np.array([[[2.]], [[.75]]])]
        second = [np.ones((2, 1, 1)), np.array([[[.01]], [[.01]]]), np.full((2, 1, 1), .5)]
        result = a.compare(first, second, np.array([1, 2]))
        self.assertEqual(result['reach_difference']['mean'], -.5)
        self.assertEqual(result['cost_difference']['mean'], .875)
        self.assertAlmostEqual(result['joint_forgetting_difference']['mean'], .02)
        self.assertEqual(result['joint_forgetting_difference']['histories'], 1)

    def test_continuation_actions_are_scored_on_independent_shared_future(self):
        def state(values):
            return dict(root=1, x=[0.] * 14, branches=[
                [dict(success=True, forgetting=f) for f in pair] for pair in values])
        constant = state([[.04, .04], [0., .04], [.04, .04], [.04, .04], [.04, .04]])
        adaptive = state([[0., .04], [.01, 0.], [.04, .04], [.04, .04], [.04, .04]])
        row = dict(root=1, families=dict(constant=constant, fixed=constant, adaptive=adaptive))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'test_continuations.jsonl').write_text(json.dumps(row) + '\n')
            result = a.continuation_analysis(path)
        # Qadaptive selects action 0 with stream 0, but action 1 wins on stream 1.
        # Both first actions must be scored under adaptive's stream-1 outcomes.
        self.assertAlmostEqual(result['paired']['adaptive_vs_constant']['cost_difference']['mean'], .8)
        self.assertEqual(result['disagreements']['constant']['disjoint_best_action_sets'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)

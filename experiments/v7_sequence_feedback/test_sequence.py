"""Mechanism checks for continuous-state and information-matched comparisons."""
import copy
import unittest

import numpy as np
import torch

import control as l
import sequence_env as e


class SequenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = dict(e.CONFIG, base_steps=4, train_n=24, monitor_n=12, test_n=24,
                     replay_memory=12, monitor_memory=12, batch=4, old_batch=4,
                     segment_steps=6, interval=2, target=1e-12)
        cls.item = e.make_item(710001, cls.c)

    def env(self):
        return e.SequenceEnv(self.item, self.c)

    def test_adam_matches_torch_across_boundaries(self):
        env = self.env()
        w = env.w.detach().clone().requires_grad_()
        opt = torch.optim.Adam([w], lr=self.c['lr'], betas=(.9, .99))
        while not env.done:
            for step in range(env.local_steps, min(env.local_steps+self.c['interval'], self.c['segment_steps'])):
                ni, oi = env.new_indices[step], env.old_indices[step]
                new, old = (env.new[0][ni], env.new[1][ni]), (env.old[0][oi], env.old[1][oi])
                loss = e.core.loss(w, new, env.router, self.c)+self.c['replay_weight']*e.core.loss(w, old, env.router, self.c)
                opt.zero_grad(); loss.backward(); opt.step()
            env.advance_window(4)
            torch.testing.assert_close(env.w, w, atol=2e-6, rtol=2e-5)
        self.assertEqual(env.steps, 18)

    def test_boundaries_preserve_references_and_moments(self):
        env = self.env()
        refs, initial = env.old_refs.copy(), env.initial_w.clone()
        env.act(4, 'block')
        self.assertEqual((env.stage, env.local_steps, env.steps), (1, 0, 6))
        self.assertGreater(float(env.adam_m.abs().sum()), 0.)
        self.assertGreater(float(env.adam_v.sum()), 0.)
        np.testing.assert_array_equal(refs, env.old_refs)
        torch.testing.assert_close(initial, env.initial_w)
        expected = np.maximum(env.old_current-refs, 0).mean()
        self.assertAlmostEqual(env.forgetting, expected)

    def test_no_refreshed_old_loss_in_progress(self):
        env = self.env()
        original_feature = env.observe()[5]
        env.act(4, 'block')
        self.assertEqual(env.observe()[5], original_feature)
        before = env.observe(False)
        env.forgetting += 1.; env.old_delta += 2.; env.old_current += 3.
        np.testing.assert_array_equal(before, env.observe(False))
        self.assertFalse(np.array_equal(before, env.observe(True)))

    def test_hold_and_repeated_action_identical_state_and_charge(self):
        block, within = self.env(), self.env()
        while not block.done: block.act(2, 'block')
        while not within.done: within.act(2, 'within')
        torch.testing.assert_close(block.w, within.w, rtol=0, atol=0)
        self.assertEqual(block.work, within.work)
        self.assertEqual(len(block.decisions), 3)
        self.assertEqual(len(within.decisions), 9)

    def test_actual_displacement_freezes_protected_group(self):
        env = self.env()
        initial = env.w.detach().clone()
        env.advance_window(0)
        torch.testing.assert_close(initial[env.mask], env.w[env.mask], rtol=0, atol=0)
        self.assertGreater(float(env.adam_m[env.mask].abs().sum()), 0.)

    def test_clone_isolates_state_and_future_streams(self):
        env = self.env(); branch = e.clone_env(env)
        original_indices = env.new_indices.clone()
        original_w = env.w.detach().clone()
        branch.set_future_rep(1); branch.act(4, 'within')
        torch.testing.assert_close(original_indices, env.new_indices)
        torch.testing.assert_close(original_w, env.w)
        self.assertEqual(env.steps, 0)
        self.assertIs(branch.world, env.world)

    def test_paired_future_resampling_and_stage_keying(self):
        env = self.env(); env.act(4, 'within')
        a, b = e.clone_env(env), e.clone_env(env)
        past = a.new_indices[:2].clone()
        a.set_future_rep(2); b.set_future_rep(2)
        torch.testing.assert_close(a.new_indices, b.new_indices)
        torch.testing.assert_close(past, a.new_indices[:2])
        a.act(0, 'block'); b.act(4, 'block')
        self.assertEqual(a.stage, b.stage)
        torch.testing.assert_close(a.new_indices, b.new_indices)
        self.assertEqual(a.future_rep, 2)

    def test_test_labels_do_not_affect_online_trajectory(self):
        world, count, w = self.item
        altered = copy.deepcopy(world)
        for task in altered['tasks']:
            x, y = task['test']; task['test'] = (x, y+1000)
        a, b = self.env(), e.SequenceEnv((altered, count, w), self.c)
        while not a.done:
            np.testing.assert_array_equal(a.observe(), b.observe())
            a.act(2, 'within'); b.act(2, 'within')
        torch.testing.assert_close(a.w, b.w)
        self.assertNotEqual(e.endpoint(a)['old_eval_loss'], e.endpoint(b)['old_eval_loss'])

    def test_failed_blocks_remain_in_sequence_success(self):
        env = self.env()
        while not env.done: env.act(0, 'block')
        row = e.endpoint(env)
        self.assertFalse(row['success'])
        self.assertEqual(row['segment_hits'], [False]*3)
        self.assertEqual(len(row['segments']), 3)

    def test_fixed_mask_memory_cap_and_budget(self):
        env = self.env(); mask = env.mask.clone()
        old_data, refs = env.old, env.old_refs.copy()
        while not env.done:
            env.act(4, 'block')
            torch.testing.assert_close(mask, env.mask)
            self.assertIs(old_data, env.old)
            np.testing.assert_array_equal(refs, env.old_refs)
        self.assertLessEqual(len(env.old[0]), self.c['replay_memory'])
        self.assertLessEqual(env.old_n, self.c['monitor_memory'])
        self.assertLessEqual(env.work, env.budget)

    def test_mask_after_normalization_and_feature_shape(self):
        x = self.env().observe()
        mean, std = np.ones(len(e.FEATURES)), np.ones(len(e.FEATURES))*2
        tx = l.features(x, mean, std, False)
        self.assertEqual(len(tx), 18)
        torch.testing.assert_close(tx[-2:], torch.zeros(2))
        model = l.Scorer()
        self.assertEqual(tuple(model(tx)[0].shape), (5,))

    def test_selection_prioritizes_reach(self):
        safe = dict(reach=1., forgetting=.1, work=100.)
        less_safe = dict(reach=.99, forgetting=.001, work=10.)
        self.assertLess(l.selection_key(safe), l.selection_key(less_safe))


if __name__ == '__main__':
    unittest.main()

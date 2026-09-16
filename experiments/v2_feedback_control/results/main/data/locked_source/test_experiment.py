"""Tests of the experiment's causal, measurement and budget boundaries."""
import copy
import unittest

import numpy as np
import torch

import experiment as e


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(e.CONFIG,max_steps=17,target=1e-8,levels=[1e-8])
        cls.item = e.make_bank([220017],cls.config)[0]

    def env(self,policy=False):
        return e.ScalarEnv(self.item,self.config,e.Policy.macs() if policy else 0)

    def test_test_labels_are_inaccessible_to_updates_rewards_and_state(self):
        world,task,w,refs = self.item
        other = e.core.make_world(world['seed'],self.config,include_test=True)
        for t in other['tasks']:
            t['test'][1].fill_(1e9)
        a = self.env()
        b = e.ScalarEnv((other,task,w,refs),self.config)
        for action in [2,0,4]:
            torch.testing.assert_close(a.observe(),b.observe(),rtol=0,atol=0)
            self.assertEqual(a.advance(action),b.advance(action))
            torch.testing.assert_close(a.w,b.w,rtol=0,atol=0)

    def test_zero_multiplier_freezes_even_with_momentum(self):
        a = self.env()
        self.assertTrue(bool(a.mask.any()))
        self.assertTrue(bool((~a.mask).any()))
        original = a.w.detach().clone()
        a.adam_m.fill_(0.1)
        a.adam_v.fill_(0.1)
        a.advance(0)
        torch.testing.assert_close(a.w[a.mask],original[a.mask],rtol=0,atol=0)
        self.assertFalse(torch.equal(a.w[~a.mask],original[~a.mask]))

    def test_grouping_and_routing_are_fixed_during_learning(self):
        a = self.env()
        mask = a.mask.clone()
        old_routes = e.core.routes(a.new[0],a.router,a.c)
        while not a.done:
            a.advance(3)
            torch.testing.assert_close(a.mask,mask,rtol=0,atol=0)
        for before,after in zip(old_routes,e.core.routes(a.new[0],a.router,a.c)):
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_progress_policy_cannot_observe_current_old_feedback(self):
        a = self.env()
        before = a.observe(False)
        a.forgetting = 5
        a.old_losses += 9
        torch.testing.assert_close(before,a.observe(False),rtol=0,atol=0)
        self.assertFalse(torch.equal(before,a.observe(True)))

    def test_work_budget_includes_policy_and_monitor_cost(self):
        a = self.env(policy=True)
        while not a.done:
            a.advance(4)
            self.assertLessEqual(a.work,a.budget+1e-7)
        expected = (a.initial_work+a.steps*a.step_charge+
                    len(a.events)*(a.old_n+a.policy_charge))
        self.assertAlmostEqual(a.work,expected,places=7)
        self.assertLessEqual(a.steps,self.config['max_steps'])

    def test_old_memory_is_bounded(self):
        a = self.env()
        self.assertLessEqual(len(a.old[0]),self.config['replay_memory'])
        self.assertLessEqual(a.old_n,self.config['monitor_memory'])

    def test_shaping_telescopes_to_terminal_objective(self):
        a = self.env()
        potential = a.potential()
        rewards = 0
        while not a.done:
            rewards += a.advance(2)
        self.assertAlmostEqual(rewards,a.terminal_reward()-potential,places=7)

    def test_expert_permutation_preserves_scalar_observation_and_mask(self):
        world,task,w,refs = self.item
        perm = torch.tensor([3,0,5,2,1,4])
        other = copy.deepcopy(world)
        other['router'] = world['router'][:,perm]
        a = self.env()
        b = e.ScalarEnv((other,task,w[perm],refs),self.config)
        torch.testing.assert_close(a.observe(),b.observe(),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(b.mask,a.mask[perm])
        a.advance(2)
        b.advance(2)
        torch.testing.assert_close(b.w,a.w[perm],atol=1e-5,rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

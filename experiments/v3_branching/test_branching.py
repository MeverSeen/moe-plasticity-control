"""Tests of cloning, intervention, leakage and endpoint semantics."""
import copy
import unittest

import numpy as np
import torch

import branching as b


class BranchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=dict(b.base.CONFIG,max_steps=37,target=1e-8,levels=[1e-8])
        cls.item=b.make_bank([350103],cls.config)[0]

    def env(self):return b.new_env(self.item,self.config)

    def test_full_clone_replays_identically_and_leaves_source_unchanged(self):
        source=self.env();source.advance(3)
        w=source.w.detach().clone();m=source.adam_m.clone()
        a=b.clone_env(source);c=b.clone_env(source)
        for action in [0,4,2]:
            a.advance(action);c.advance(action)
        for key in ['w','adam_m','adam_v','new_indices','old_indices']:
            torch.testing.assert_close(getattr(a,key),getattr(c,key),rtol=0,atol=0)
        self.assertEqual(a.events,c.events)
        self.assertEqual(a.work,c.work)
        torch.testing.assert_close(source.w,w,rtol=0,atol=0)
        torch.testing.assert_close(source.adam_m,m,rtol=0,atol=0)
        self.assertEqual(source.steps,5)

    def test_future_replicates_preserve_past_and_pair_actions(self):
        source=self.env();source.advance(4)
        a=b.resample_future(source,0);c=b.resample_future(source,0)
        d=b.resample_future(source,1)
        torch.testing.assert_close(a.new_indices,c.new_indices,rtol=0,atol=0)
        torch.testing.assert_close(a.old_indices[:5],source.old_indices[:5],rtol=0,atol=0)
        self.assertFalse(torch.equal(a.new_indices[5:],d.new_indices[5:]))
        self.assertEqual(a.events,source.events)

    def test_evaluation_labels_do_not_affect_state_or_training(self):
        other=copy.deepcopy(self.item)
        for task in other[0]['tasks']:task['test'][1].fill_(1e8)
        a=self.env();c=b.new_env(other,self.config)
        for action in [2,0,4]:
            torch.testing.assert_close(a.observe(),c.observe(),rtol=0,atol=0)
            a.advance(action);c.advance(action)
            torch.testing.assert_close(a.w,c.w,rtol=0,atol=0)

    def test_zero_displacement_freezes_protected_parameters_with_momentum(self):
        a=self.env();a.mask[:]=False;a.mask[0]=True
        a.adam_m.fill_(.3);a.adam_v.fill_(.2)
        old=a.w.detach().clone()
        a.advance(0)
        torch.testing.assert_close(a.w[0],old[0],rtol=0,atol=0)
        self.assertFalse(torch.equal(a.w[1:],old[1:]))

    def test_branch_is_continued_to_endpoint_not_just_H_steps(self):
        a=self.env();a.advance(1)
        self.assertFalse(a.done)
        self.assertEqual(a.steps,5)
        b.finish(a,4)
        self.assertTrue(a.done)
        self.assertGreater(a.steps,5)
        self.assertLessEqual(a.work,a.budget+1e-7)
        self.assertFalse(a.success)

    def test_first_crossing_stops_inside_control_interval(self):
        a=self.env()
        a.c=dict(a.c,target=1e9)
        a.advance(4)
        self.assertTrue(a.done)
        self.assertTrue(a.success)
        self.assertEqual(a.steps,1)

    def test_progress_preprocessing_excludes_feedback(self):
        x=np.ones((2,14),dtype=np.float32)
        x[1,12:]=[100,-50]
        mean=np.arange(14,dtype=np.float32)/10
        std=np.ones(14,dtype=np.float32)
        a=b.features(x,mean,std,False)
        torch.testing.assert_close(a[0],a[1],rtol=0,atol=0)
        self.assertFalse(torch.equal(*b.features(x,mean,std,True)))

    def test_no_protected_experts_is_an_exact_negative_control(self):
        a=self.env();a.mask[:]=False
        c=b.clone_env(a)
        a.advance(0);c.advance(4)
        torch.testing.assert_close(a.w,c.w,rtol=0,atol=0)
        self.assertEqual(a.current_loss,c.current_loss)

    def test_history_splits_are_disjoint(self):
        seen=set()
        for seeds in b.SPLITS.values():
            self.assertFalse(seen.intersection(seeds));seen.update(seeds)


if __name__=='__main__':unittest.main(verbosity=2)

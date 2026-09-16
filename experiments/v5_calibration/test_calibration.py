"""Check that calibration measures decision value rather than forcing conflict."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import calibrate as c


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=dict(c.b.base.CONFIG,max_steps=17,target=1e-8,levels=[1e-8])
        cls.item=c.b.make_bank([549901],cls.config)[0]

    def test_plan_family_includes_fixed_time_and_progress(self):
        plans=c.plans()
        self.assertEqual(len(plans),17)
        self.assertEqual(len({p['name'] for p in plans}),17)
        self.assertEqual({kind:sum(p['kind']==kind for p in plans) for kind in ['fixed','time','progress']},
                         dict(fixed=5,time=6,progress=6))

    def test_progress_switch_is_sticky_and_uses_loss_progress(self):
        choose=c.chooser(dict(kind='progress',early=0,late=4,threshold=.5))
        env=SimpleNamespace(c={'target':.01,'max_steps':100},initial_loss=1.,current_loss=1.,steps=0)
        self.assertEqual(choose(env),0)
        env.current_loss=.05
        self.assertEqual(choose(env),4)
        env.current_loss=.8
        self.assertEqual(choose(env),4)

    def test_time_switch_does_not_depend_on_old_feedback(self):
        choose=c.chooser(dict(kind='time',early=4,late=0,threshold=.3))
        env=SimpleNamespace(c={'max_steps':100},steps=29,forgetting=1000)
        self.assertEqual(choose(env),4)
        env.steps=30
        self.assertEqual(choose(env),0)

    def test_future_streams_are_paired_and_independent(self):
        a=c.b.new_env(self.item,self.config)
        x=c.paired_future(a,0);y=c.paired_future(a,0);z=c.paired_future(a,1)
        torch.testing.assert_close(x.new_indices,y.new_indices,rtol=0,atol=0)
        self.assertFalse(torch.equal(x.new_indices,z.new_indices))
        r1=c.run_plan(self.item,self.config,c.plans()[2],0)
        r2=c.run_plan(self.item,self.config,c.plans()[2],0)
        self.assertEqual(r1,r2)

    def test_budget_settings_share_new_and_replay_batch_prefixes(self):
        short=c.paired_future(c.b.new_env(self.item,self.config),0)
        long=c.paired_future(c.b.new_env(self.item,dict(self.config,max_steps=37)),0)
        torch.testing.assert_close(short.new_indices,long.new_indices[:17],rtol=0,atol=0)
        torch.testing.assert_close(short.old_indices,long.old_indices[:17],rtol=0,atol=0)

    def test_eval_labels_cannot_change_training_or_plan_actions(self):
        other=copy.deepcopy(self.item)
        for task in other[0]['tasks']:
            task['test'][1].fill_(1e8)
        a=c.run_plan(self.item,self.config,c.plans()[-1],0)
        d=c.run_plan(other,self.config,c.plans()[-1],0)
        self.assertEqual(a['action_trace'],d['action_trace'])
        self.assertEqual(a['new_monitor_loss'],d['new_monitor_loss'])
        self.assertEqual(a['work'],d['work'])

    def test_oracle_prefers_feasible_plan_over_zero_forgetting_failure(self):
        values=dict(success=np.array([[[0.,1.],[1.,1.]]]),
                    forgetting=np.array([[[0.,0.],[.2,.3]]]),work=np.ones((1,2,2)))
        self.assertEqual(c.best_index(values,[0,1]),1)
        # Selection uses only stream 0, never stream-1 outcomes.
        values['forgetting'][0,1,1]=1e8
        self.assertEqual(c.history_choices(values,[0,1]).tolist(),[1])

    def test_setting_selection_does_not_require_low_multiplier_failure(self):
        def summary(gain,pressure):
            return dict(selection_stream_oracle_reach=1.,selection_stream_active_fraction=1.,
                diagnostics={'low_fails_high_succeeds_selection_stream':pressure},
                cross_future={'plan_oracle_vs_nonfeedback':{'cost_difference':{'mean':-gain}}})
        chosen,_=c.select_setting({'a':summary(.1,0),'b':summary(.01,5)},[{'name':'a'},{'name':'b'}])
        self.assertEqual(chosen,'a')

    def test_new_roots_are_disjoint(self):
        seen=set()
        for splits in [c.SPLITS,c.PILOT_SPLITS]:
            for roots in splits.values():
                self.assertFalse(seen.intersection(roots));seen.update(roots)
        self.assertGreater(min(seen),459999)


if __name__=='__main__':
    unittest.main(verbosity=2)

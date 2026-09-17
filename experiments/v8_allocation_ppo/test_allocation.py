import copy
import unittest
import numpy as np
import torch
import sequence_env as e
import control as l
import ppo

class AllocationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=dict(e.CONFIG,base_steps=4,train_n=24,monitor_n=12,test_n=24,replay_memory=12,monitor_memory=12,batch=4,old_batch=4,segment_steps=6,interval=1,target=1e-12)
        cls.item=e.make_item(890001,cls.c)
    def test_strict_old_column(self):
        self.assertEqual(e.ACTIONS[:5],[(p,1.) for p in [0.,.25,.5,.75,1.]])
    def test_displacements_and_unscaled_moments(self):
        base=e.SequenceEnv(self.item,self.c); base.mask[:3]=True; base.mask[3:]=False
        initial=base.w.detach().clone(); full=e.clone_env(base); full.advance_window(4)
        for a,(p,u) in enumerate(e.ACTIONS):
            branch=e.clone_env(base); branch.advance_window(a)
            alpha=torch.where(base.mask,p,u)[:,None]
            torch.testing.assert_close(branch.w-initial,(full.w-initial)*alpha,atol=1e-7,rtol=2e-4)
            torch.testing.assert_close(branch.adam_m,full.adam_m,rtol=0,atol=0)
            torch.testing.assert_close(branch.adam_v,full.adam_v,rtol=0,atol=0)
    def test_reward_cost_and_actual_policy_update(self):
        torch.manual_seed(881); model=ppo.ActorCritic(15)
        ck=dict(mean=np.zeros(18),std=np.ones(18),feedback=True)
        rows,summary=ppo.episode(model,ck,self.item,self.c)
        self.assertAlmostEqual(sum(r['reward'] for r in rows),-summary['forgetting']/e.F_SCALE)
        self.assertEqual(rows[-1]['terminal_cost'],1.)
        self.assertTrue(all(r['cost_return']==1 for r in rows))
        opt=torch.optim.Adam(model.parameters(),lr=.0003)
        result=ppo.update(model,opt,rows,ck,1.)
        self.assertGreater(result['actor_parameter_delta'],0)
    def test_ppo_test_label_isolation(self):
        world,n,w=self.item; altered=copy.deepcopy(world)
        for task in altered['tasks']:
            x,y=task['test'];task['test']=(x,y+100)
        model=ppo.ActorCritic(5); ck=dict(mean=np.zeros(18),std=np.ones(18),feedback=False)
        torch.manual_seed(9); a,sa=ppo.episode(model,ck,self.item,self.c)
        torch.manual_seed(9); b,sb=ppo.episode(model,ck,(altered,n,w),self.c)
        self.assertEqual(a,b); self.assertEqual(sa,sb)
    def test_extended_observation_action_encoding(self):
        env=e.SequenceEnv(self.item,self.c);env.advance_window(14)
        self.assertEqual(env.observe()[11],3.5);self.assertEqual(len(env.observe()),18)

if __name__=='__main__':unittest.main()

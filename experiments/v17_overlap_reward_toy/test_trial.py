import copy
import importlib.util
from pathlib import Path
import unittest
import torch
from run import old, rollout

path = Path(__file__).resolve().parent.parent/'v16_step_reward_toy'/'run.py'
spec = importlib.util.spec_from_file_location('frozen_v16',path)
v16 = importlib.util.module_from_spec(spec);spec.loader.exec_module(v16)


class TestOverlapReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = dict(old.CONFIG, base_steps=3,max_steps=8,train_n=32,monitor_n=16,
                     test_n=32,replay_memory=8,monitor_memory=8,batch=8,old_batch=8,
                     target=1e-10,interval=1)
        # A fixture with both protected and free routed experts; the empty-mask
        # case is exercised separately below.
        cls.item = old.make_bank([1790101],cls.c,test=True)[0]

    def policy(self):
        torch.manual_seed(11);p=old.Policy()
        with torch.no_grad():p.actor.bias[2]=1.
        return p

    def test_original_two_rewards_preserve_exact_trajectory(self):
        for reward in ('forgetting','movement'):
            a,ta,wa=v16.rollout(self.item,self.c,1,reward,.1,self.policy())
            b,tb,wb=rollout(self.item,self.c,1,reward,.1,self.policy())
            self.assertTrue(torch.equal(wa,wb))
            for key in ('path_energy','new_monitor_loss','test_F','terminal_reward','actions'):
                self.assertEqual(a[key],b[key])
            for x,y in zip(ta,tb):
                self.assertTrue(torch.equal(x['x'],y['x']))
                self.assertEqual(x['reward'],y['reward'])

    def test_partition_ledger_and_reward(self):
        row,tr,w=rollout(self.item,self.c,1,'overlap',.1,self.policy(),keep_trace=True)
        self.assertAlmostEqual(row['path_energy'],row['overlap_path_energy']+row['other_path_energy'],places=12)
        self.assertAlmostEqual(row['net_displacement_sq'],row['overlap_net_displacement_sq']+row['other_net_displacement_sq'],places=12)
        for event in row['events']:
            protected=sum(x for x,m in zip(event['expert_update_sq'],row['protected_mask']) if m)
            self.assertAlmostEqual(protected,event['overlap_update_sq'],places=12)
        self.assertAlmostEqual(sum(x['reward'] for x in tr),row['terminal_reward'],places=10)
        self.assertAlmostEqual(row['terminal_reward'],-row['overlap_path_energy']/.1-10,places=10)

    def test_freeze_has_zero_protected_motion_but_free_experts_learn(self):
        row,_,w=rollout(self.item,self.c,1,'overlap',.1,action=0)
        self.assertGreater(row['protected_experts'],0)
        self.assertLess(row['protected_experts'],self.c['experts'])
        self.assertEqual(row['overlap_path_energy'],0.)
        self.assertEqual(row['overlap_net_displacement_sq'],0.)
        self.assertGreater(row['other_path_energy'],0.)

    def test_empty_and_full_masks(self):
        empty,_,_=rollout(self.item,dict(self.c,mask_old_threshold=10),1,'overlap',.1,action=2)
        self.assertEqual(empty['overlap_path_energy'],0.)
        self.assertGreater(empty['path_energy'],0.)
        full,_,_=rollout(self.item,dict(self.c,mask_old_threshold=0,mask_new_threshold=0),1,'overlap',.1,action=2)
        self.assertEqual(full['overlap_path_energy'],full['path_energy'])
        self.assertEqual(full['other_path_energy'],0.)

    def test_test_labels_do_not_affect_control(self):
        changed=copy.deepcopy(self.item)
        for task in changed[0]['tasks']:task['test']=(task['test'][0],task['test'][1]+100)
        a,ta,wa=rollout(self.item,self.c,1,'overlap',.1,self.policy())
        b,tb,wb=rollout(changed,self.c,1,'overlap',.1,self.policy())
        self.assertTrue(torch.equal(wa,wb))
        self.assertEqual(a['terminal_reward'],b['terminal_reward'])
        self.assertNotEqual(a['test_old_loss'],b['test_old_loss'])


if __name__=='__main__':unittest.main()

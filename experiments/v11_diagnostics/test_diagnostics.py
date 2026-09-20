import copy,unittest
import numpy as np
import torch
from core import Config,Environment,Learner,GatedAdam
from policies import warm_policy,RepairPolicy,FixedPolicy
from control import rollout

torch.set_num_threads(1)
class FakeData:
 def task(self,classes,role):
  rng=torch.Generator().manual_seed(int(classes[0])*100+{'train':1,'monitor':2,'eval':3}[role]);n=150
  return torch.randint(0,255,(n,3,32,32),dtype=torch.uint8,generator=rng),torch.arange(n)%5,torch.arange(n)
class Tests(unittest.TestCase):
 def test_shared_direction_and_moments(self):
  torch.manual_seed(1);a=Learner(Config());b=copy.deepcopy(a);oa=GatedAdam(a,Config());ob=GatedAdam(b,Config());old=copy.deepcopy(a.state_dict())
  for m in [a,b]:
   for p in m.parameters():
    if p.requires_grad:p.grad=torch.ones_like(p)
  oa.trunk_multiplier=.5;oa.head_multiplier=0;oa.step(torch.zeros(2,4));ob.step(torch.ones(2,4))
  for name,p in a.named_parameters():
   q=dict(b.named_parameters())[name]
   if name.startswith(('front.','norms.')):self.assertTrue(torch.equal(p,old[name]+.5*(q-old[name])))
   if name.startswith(('heads.','blocks.')):self.assertTrue(torch.equal(p,old[name]))
   if p.requires_grad:
    for key in ['step','exp_avg','exp_avg_sq']:self.assertTrue(torch.equal(oa.optimizer.state[p][key],ob.optimizer.state[q][key]))
 def test_monitor_stratification_no_dynamics_change(self):
  rows=[];states=[];memories=[];moments=[]
  for mode in ['random','stratified']:
   e=Environment(FakeData(),Config(tasks=3,steps=10,memory=200,monitor_mode=mode,target=0),123)
   r,_=rollout(e,FixedPolicy());rows.append(r);states.append(e.model.state_dict());memories.append(e.memory);moments.append(e.optim.optimizer.state_dict()['state'])
  self.assertEqual(rows[0]['matrix'],rows[1]['matrix'])
  for name,p in states[0].items():self.assertTrue(torch.equal(p,states[1][name]))
  self.assertEqual(moments[0].keys(),moments[1].keys())
  for k,state in moments[0].items():
   for name,value in state.items():self.assertTrue(torch.equal(value,moments[1][k][name]))
  for j in memories[0]:
   self.assertTrue(torch.equal(memories[0][j]['replay'][2],memories[1][j]['replay'][2]))
   counts=torch.bincount(memories[1][j]['monitor'][1],minlength=5);self.assertLessEqual(int(counts.max()-counts.min()),1)
 def test_integer_fixed_action_dtype(self):
  e=Environment(FakeData(),Config(tasks=2,steps=10,target=0),123);r,_=rollout(e,FixedPolicy(0));self.assertEqual(len(r["tasks"]),2)
 def test_warm_random_distribution(self):
  p=warm_policy();obs=(torch.randn(2,4,12),torch.randn(9));ds,_=p.distributions(obs)
  self.assertTrue(torch.allclose(ds[0].probs[:,1],torch.full((2,),.98)))
  self.assertTrue(torch.allclose(ds[2].probs[:,:,1],torch.full((2,4),.98)))
  self.assertTrue(torch.equal(p.act(obs,deterministic=True)[0],torch.full((2,4),.25)))
 def test_repair_support_probability_and_trunk(self):
  for trunk in [False,True]:
   p=RepairPolicy(trunk=trunk);obs=(torch.randn(2,4,12),torch.randn(9));m,z,lp,ent,v=p.act(obs);ds,_=p.distributions(obs)
   expected=ds[1].log_prob(z[0]).sum()+torch.distributions.Categorical(logits=ds[2].logits[torch.arange(2),z[0]]).log_prob(z[1]).sum()
   if trunk:expected+=torch.distributions.Categorical(logits=p.trunk_head(obs[1])).log_prob(z[2]);m=m[0]
   self.assertTrue(torch.allclose(lp,expected));self.assertTrue(((m!=.25).sum(1)<=1).all())
   (-lp).backward();self.assertGreater(float(p.object.weight.grad.abs().sum()),0)
 def test_reward_only_preserves_trajectories(self):
  rows=[]
  for mode in ['positive','balanced','signed']:
   e=Environment(FakeData(),Config(tasks=3,steps=10,target=0,reward_mode=mode),123);r,_=rollout(e,FixedPolicy());rows.append(r)
  self.assertTrue(all(r['matrix']==rows[0]['matrix'] and r['actions']==rows[0]['actions'] for r in rows))
if __name__=='__main__':unittest.main()

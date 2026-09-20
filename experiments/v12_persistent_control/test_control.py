import copy,io,sys,unittest
from pathlib import Path
import torch
from core import Config,Environment,Learner,GatedAdam
from policy import Policy,Baseline,TRUNK,prior
from control import advance,rollout,train_batch
torch.set_num_threads(1)
class Data:
 def task(self,classes,role):
  g=torch.Generator().manual_seed(int(classes[0])*100+{'train':1,'monitor':2,'eval':3}[role]);n=150
  return torch.randint(0,255,(n,3,32,32),dtype=torch.uint8,generator=g),torch.arange(n)%5,torch.arange(n)
class Tests(unittest.TestCase):
 def test_actual_displacement(self):
  torch.manual_seed(1);a=Learner(Config());b=copy.deepcopy(a);oa=GatedAdam(a,Config());ob=GatedAdam(b,Config());old=copy.deepcopy(a.state_dict())
  for model in [a,b]:
   for p in model.parameters():
    if p.requires_grad:p.grad=torch.ones_like(p)
  oa.trunk_multiplier=.5;oa.step(torch.zeros(2,4));ob.step(torch.ones(2,4))
  for n,p in a.named_parameters():
   q=dict(b.named_parameters())[n]
   if n.startswith(('front.','norms.')):self.assertTrue(torch.equal(p,old[n]+.5*(q-old[n])))
   if n.startswith('blocks.'):self.assertTrue(torch.equal(p,old[n]))
   if p.requires_grad:
    for k in ['step','exp_avg','exp_avg_sq']:self.assertTrue(torch.equal(oa.optimizer.state[p][k],ob.optimizer.state[q][k]))
 def test_hold_no_resampling_and_same_stop_checks(self):
  rr=[]
  for hold in [10,50]:
   e=Environment(Data(),Config(tasks=2,steps=70,target=0),22);r,tr=rollout(e,Baseline(),hold,True,True);rr.append(r)
   self.assertEqual(len(tr),7 if hold==10 else 2);self.assertEqual([t['duration'] for t in r['policy_trace']],[10]*7 if hold==10 else [50,20]);self.assertEqual(r['cost']['control_decisions'],len(tr))
  self.assertEqual(rr[0]['matrix'],rr[1]['matrix']);self.assertEqual(rr[0]['cost']['observation_examples'],rr[1]['cost']['observation_examples'])
 def test_restore_mid_hold(self):
  e=Environment(Data(),Config(tasks=2,steps=70,target=0),22)
  while e.k==0:advance(e,Baseline(),50)
  advance(e,Baseline(),50);advance(e,Baseline(),50)
  buffer=io.BytesIO();torch.save(e.clone(),buffer);buffer.seek(0);f=torch.load(buffer,weights_only=False)
  a,_=rollout(e,Baseline(),50);b,_=rollout(f,Baseline(),50)
  self.assertEqual(a['matrix'],b['matrix']);self.assertEqual(a['policy_trace'],b['policy_trace'])
  for n,v in e.model.state_dict().items():self.assertTrue(torch.equal(v,f.model.state_dict()[n]))
  for i,v in e.optim.optimizer.state_dict()['state'].items():
   for k,t in v.items():self.assertTrue(torch.equal(t,f.optim.optimizer.state_dict()['state'][i][k]))
 def test_sampling_probability_and_joint_support(self):
  obs=(torch.randn(2,4,12),torch.randn(9))
  for joint in [False,True]:
   p=Policy(joint=joint);changed=[];trunks=[];base=[]
   for _ in range(400):
    m,z,lp,_,_=p.act(obs);_,_,again,_,_=p.act(obs,z);self.assertTrue(torch.equal(lp,again));changed.append(int((m[0]!=0).sum()));trunks.append(m[1]);base.append(int(z[0])==0)
    if int(z[0])==0:self.assertAlmostEqual(float(lp.detach()),-0.69314718,places=5)
    else:
     expected=torch.log(torch.tensor(.5))+prior(3,torch.tensor(1),.5)[z[1]]+prior(6,torch.tensor(0),.5)[z[2]]+prior(6,z[2].expand(2),.8)[torch.arange(2),z[3]].sum()
     if joint:expected+=prior(6,z[3][:,None].expand(2,4),.8).gather(-1,z[4][...,None]).sum()
     else:
      h=p.features(obs);d=torch.distributions.Categorical(logits=p.object(h).squeeze(-1));expected+=d.log_prob(z[4]).sum()+prior(6,z[3],.8)[torch.arange(2),z[5]].sum()
     self.assertTrue(torch.allclose(lp,expected,atol=1e-6))
   self.assertGreater(sum(base),130);self.assertLess(sum(base),270);self.assertGreater(trunks.count(.5),200)
   self.assertTrue(max(changed)>2 if joint else max(changed)<=2)
 def test_grpo_updates_and_one_transition_per_hold(self):
  e=Environment(Data(),Config(tasks=2,steps=60,target=0),11)
  while e.k==0:advance(e,Baseline(),50)
  p=Policy(joint=True);opt=torch.optim.Adam(p.parameters(),lr=.001);r=train_batch(p,opt,e,50,4)
  self.assertEqual(r['decisions'],8);self.assertEqual(r['optimizer_steps'],2);self.assertGreater(r['parameter_change'],0)
if __name__=='__main__':unittest.main()

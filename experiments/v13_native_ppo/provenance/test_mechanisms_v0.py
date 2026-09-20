import copy,math,unittest
import numpy as np
import torch
from mechanisms import GatedAdamW,ActorCritic,gae,ppo_update
torch.set_num_threads(1)
class Tests(unittest.TestCase):
 def test_complete_displacement_and_moments(self):
  a=[torch.nn.Parameter(torch.tensor([1.,2.])) for _ in range(4)];b=copy.deepcopy(a);oa=GatedAdamW([[p] for p in a],.01,.1);ob=GatedAdamW([[p] for p in b],.01,.1);old=[p.detach().clone() for p in a]
  for p in a+b:p.grad=torch.tensor([.3,-.2])
  oa.step(torch.tensor([0.,.25,.5,1.]));ob.step(torch.ones(4))
  for i,(p,q,m) in enumerate(zip(a,b,[0.,.25,.5,1.])):
   self.assertTrue(torch.equal(p,old[i]+m*(q-old[i])))
   for key in ['step','exp_avg','exp_avg_sq']:self.assertTrue(torch.equal(oa.optimizer.state[p][key],ob.optimizer.state[q][key]))
 def test_optimizer_restore(self):
  a=torch.nn.Parameter(torch.tensor([1.,2.]));oa=GatedAdamW([[a]],.01);a.grad=torch.ones_like(a);oa.step(torch.tensor([.5]));b=torch.nn.Parameter(a.detach().clone());ob=GatedAdamW([[b]],.01);ob.optimizer.load_state_dict(copy.deepcopy(oa.optimizer.state_dict()))
  for p in [a,b]:p.grad=torch.tensor([.2,.4])
  oa.step(torch.tensor([.25]));ob.step(torch.tensor([.25]));self.assertTrue(torch.equal(a,b))
 def test_gae_task_boundary_and_terminal(self):
  a,r=gae([0.,1.],[.5,.7],[.7,0.],[False,True],[True,False]);np.testing.assert_allclose(a,[.485,.3]);np.testing.assert_allclose(r,[.985,1.])
  a,_=gae([0.,1.],[.5,.7],[.7,2.],[False,False],[True,False]);np.testing.assert_allclose(a,[2.385,2.3])
  a,r=gae([0,0,2],[.1,.2,.3],[.2,.3,0],[False,False,True],[True,True,False],lam=1);np.testing.assert_allclose(r,[2,2,2])
 def test_joint_probability_and_support(self):
  torch.manual_seed(3);p=ActorCritic([2,2]);obs=(torch.randn(2,64,8),torch.randn(8));changes=[]
  for _ in range(100):
   m,z,lp,_,v=p(obs);again=p(obs,z);self.assertTrue(torch.equal(lp,again[2]));self.assertTrue(torch.equal(m,again[0]));self.assertEqual(len(z),3)
   expected=sum(float(torch.where(q==1,torch.tensor(math.log(prob)),torch.tensor(math.log((1-prob)/2))).sum()) for q,prob in zip(z,[.9,.94,.96]));self.assertAlmostEqual(float(lp.detach()),expected,places=4);changes.append(int((m!=.5).sum()))
  self.assertGreater(max(changes),64);self.assertGreater(sum(n>2 for n in changes),60)
 def test_ppo_updates_actor_and_critic(self):
  torch.manual_seed(4);p=ActorCritic([2,2],experts=4);trajectories=[]
  for reward in [-.2,-2.]:
   transitions=[]
   for _ in range(8):
    obs=(torch.randn(2,4,8),torch.randn(8))
    with torch.no_grad():_,z,lp,_,v=p(obs)
    transitions.append({'obs':obs,'latent':z,'logp':float(lp),'value':float(v)})
   trajectories.append({'transitions':transitions,'reward':reward})
  old=copy.deepcopy(p.state_dict());r=ppo_update(p,torch.optim.Adam(p.parameters(),lr=1e-3),trajectories);self.assertGreater(r['parameter_change'],0);self.assertEqual(r['decisions'],16);self.assertFalse(torch.equal(old['critic.2.weight'],p.state_dict()['critic.2.weight']));self.assertFalse(torch.equal(old['residual.weight'],p.state_dict()['residual.weight']));self.assertAlmostEqual(r['epochs'][0]['kl'],0,places=6)
if __name__=='__main__':unittest.main()

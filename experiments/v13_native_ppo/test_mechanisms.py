import copy,math,unittest
import numpy as np
import torch
from mechanisms import GatedAdamW,gae,ppo_update
from sparse_policy import ActorCritic,Rule,shaped_rewards
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
 def test_sparse_probability_and_masks(self):
  torch.manual_seed(3);p=ActorCritic(experts=6);obs=(torch.randn(2,6,8),torch.randn(10))
  with torch.no_grad():
   for v in p.parameters():v.zero_()
  counts=set()
  for _ in range(100):
   m,z,lp,_,v=p(obs);again=p(obs,z);self.assertTrue(torch.equal(lp,again[2]));self.assertTrue(torch.equal(m,again[0]));pos=0;expected=0.
   for layer in range(2):
    base,count=int(z[pos]),int(z[pos+1]);pos+=2;counts.add(count);expected-=math.log(5)+math.log(3);chosen=[]
    for slot in range(count):
     who,mag=int(z[pos]),int(z[pos+1]);pos+=2;self.assertNotIn(who,chosen);self.assertNotEqual(mag,base);chosen.append(who);expected-=math.log(6-slot)+math.log(4)
    self.assertEqual(int((m[layer]!=[0,.25,.5,.75,1][base]).sum()),count)
   self.assertEqual(pos,len(z));self.assertAlmostEqual(float(lp.detach()),expected,places=4)
  self.assertEqual(counts,{0,1,2})
 def test_teacher_expressible_and_ties(self):
  loc=torch.zeros(2,6,8);loc[:,0,0]=.8;loc[:,2,0]=.8;obs=(loc,torch.zeros(10));rule=Rule([2,3]);m,z,*_=rule.act(obs);p=ActorCritic(experts=6);self.assertTrue(torch.equal(m,p(obs,z)[0]));self.assertEqual(int(z[2]),0);self.assertEqual(int(z[4]),2)
  zero=(torch.zeros_like(loc),obs[1]);m,z,*_=rule.act(zero);self.assertEqual(len(z),4);self.assertEqual(int(z[1]),0);self.assertEqual(int(z[3]),0)
 def test_cached_shaping_random_boundaries_and_restore(self):
  rng=np.random.default_rng(19);z=rng.uniform(0,.3,9).tolist();cache=copy.deepcopy({'z':z[4],'rng':rng.bit_generator.state,'task':1,'monitor_cursor':7});restored=copy.deepcopy(cache);self.assertEqual(restored['z'],z[4])
  rewards=shaped_rewards(z,.23,2);self.assertAlmostEqual(sum(rewards),-.23/.1-3*2+z[0]/.1,places=12)
  prefix=[(a-b)/.1 for a,b in zip(z[:4],z[1:5])];suffix=shaped_rewards([restored['z']]+z[5:],.23,2);np.testing.assert_allclose(prefix+suffix,rewards);self.assertTrue(any(r>0 for r in rewards));self.assertTrue(any(r<0 for r in rewards))
 def test_exact_value_advantage_invariance(self):
  z=np.array([.02,.06,.01]);values=np.array([-.7,-.8,-1.]);original=[0,0,-2.5];shaped=shaped_rewards(z,.25,0);a,_=gae(original,values,[values[1],values[2],0],[False,False,True],[True,True,False]);vv=values+z/.1;b,_=gae(shaped,vv,[vv[1],vv[2],0],[False,False,True],[True,True,False]);np.testing.assert_allclose(a,b)
 def test_ppo_updates_actor_and_critic(self):
  torch.manual_seed(4);p=ActorCritic(experts=4);trajectories=[]
  for final_F in [.02,.2]:
   transitions=[]
   for _ in range(8):
    obs=(torch.randn(2,4,8),torch.randn(10))
    with torch.no_grad():_,z,lp,_,v=p(obs)
    transitions.append({'obs':obs,'latent':z,'logp':float(lp),'value':float(v)})
   trajectories.append({'transitions':transitions,'rewards':shaped_rewards([.01*i for i in range(8)],final_F,0)})
  old=copy.deepcopy(p.state_dict());r=ppo_update(p,torch.optim.Adam(p.parameters(),lr=1e-3),trajectories);self.assertGreater(r['parameter_change'],0);self.assertEqual(r['decisions'],16);self.assertFalse(torch.equal(old['critic.2.weight'],p.state_dict()['critic.2.weight']));self.assertFalse(torch.equal(old['base_head.weight'],p.state_dict()['base_head.weight']));self.assertAlmostEqual(r['epochs'][0]['kl'],0,places=6)
if __name__=='__main__':unittest.main()

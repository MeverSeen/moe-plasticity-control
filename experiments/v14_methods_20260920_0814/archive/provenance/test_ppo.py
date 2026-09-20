import unittest,torch
from workflow import initialize
from ppo import ppo_update
from environment import Environment
from test_environment import FakeNative,data,spec,config
from structured_policy import Rule
class Tests(unittest.TestCase):
 def test_on_policy_ratio_and_real_update(self):
  torch.set_num_threads(1);p=initialize(1414);optimizer=torch.optim.Adam(p.parameters(),lr=1e-4);trajectories=[]
  for i in range(2):
   transitions=[]
   for j in range(2):
    obs=(torch.randn(16,64,10)*.1,torch.randn(10)*.1)
    with torch.no_grad():a,z,lp,en,v=p(obs);replayed=p(obs,z);self.assertEqual(float(lp),float(replayed[2]));self.assertTrue(torch.equal(a,replayed[0]))
    transitions.append({'obs':obs,'latent':z,'logp':float(lp),'value':float(v)})
   trajectories.append({'transitions':transitions,'rewards':[.1-i,-.3*i]})
  result=ppo_update(p,optimizer,trajectories);self.assertGreater(result['parameter_change'],0);self.assertEqual(result['decisions'],4);self.assertGreater(len(result['epochs']),0);self.assertTrue(all(torch.isfinite(x).all() for x in p.parameters()))
 def test_trend_cache_is_not_advanced_on_reread(self):
  e=Environment(FakeNative(),data(),spec(),config());e.prepare_prefix(__import__('pathlib').Path('/tmp/unused-method-checkpoint'));obs,z=e.observation();short=e.risk_short.clone();again,_=e.observation();self.assertTrue(torch.equal(short,e.risk_short));self.assertTrue(torch.equal(obs[0],again[0]));e._update(torch.ones(16,64));e.observation();self.assertEqual(tuple(e.risk_long.shape),(16,64));self.assertIn('recovery_long',e.state_data())
if __name__=='__main__':unittest.main()

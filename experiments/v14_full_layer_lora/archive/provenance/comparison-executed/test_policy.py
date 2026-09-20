import unittest
import torch
from structured_policy import ActorCritic,Rule
class Tests(unittest.TestCase):
 def test_full_layer_masked_action_likelihood(self):
  torch.manual_seed(1410);policy=ActorCritic();obs=(torch.randn(16,64,10),torch.randn(10));action,latent,logp,entropy,value=policy.act(obs);again=policy.act(obs,latent);self.assertEqual(tuple(action.shape),(16,64));self.assertTrue(torch.equal(action,again[0]));self.assertTrue(torch.allclose(logp,again[2]));self.assertTrue(torch.isfinite(logp+entropy+value));(-logp+value.square()).backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None),0)
  offset=0
  for layer in range(16):
   base,count=map(int,latent[offset:offset+2]);offset+=2;seen=[]
   for _ in range(count):
    expert,magnitude=map(int,latent[offset:offset+2]);offset+=2;self.assertNotIn(expert,seen);seen.append(expert);self.assertNotEqual(base,magnitude)
   self.assertLessEqual(count,2);self.assertLessEqual(int((action[layer]!=(base*.25)).sum()),2)
  self.assertEqual(offset,len(latent))
 def test_full_layer_rule_ties_and_shared_encoder(self):
  local=torch.zeros(16,64,10);local[:,:,0]=.8;obs=(local,torch.zeros(10));rule=Rule([4]*16);action,latent,*_=rule.act(obs);self.assertTrue(torch.equal(action[:,:2],torch.full((16,2),.75)));self.assertTrue(torch.equal(action[:,2:],torch.ones(16,62)));policy=ActorCritic();self.assertLess(sum(p.numel() for p in policy.parameters()),20000)
if __name__=='__main__':unittest.main()

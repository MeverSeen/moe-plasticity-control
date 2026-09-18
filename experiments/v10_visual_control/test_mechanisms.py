import copy
import unittest
import torch
from core import Config,Learner,GatedAdam
from control import Policy

torch.set_num_threads(1)
class Mechanisms(unittest.TestCase):
    def test_adam_displacement_and_moments(self):
        torch.manual_seed(1); cfg=Config(); a=Learner(cfg); b=copy.deepcopy(a); oa=GatedAdam(a,cfg); ob=GatedAdam(b,cfg)
        old=[p.clone() for p in a.blocks.parameters()]
        for m,o in [(a,oa),(b,ob)]:
            for p in m.parameters():
                if p.requires_grad: p.grad=torch.ones_like(p)*.2
        oa.step(torch.zeros(2,4)); ob.step(torch.ones(2,4))
        for p,x in zip(a.blocks.parameters(),old): self.assertTrue(torch.equal(p,x))
        for p,q in zip(a.blocks.parameters(),b.blocks.parameters()):
            self.assertTrue(torch.equal(oa.optimizer.state[p]['exp_avg'],ob.optimizer.state[q]['exp_avg']))
        self.assertTrue(any(not torch.equal(p,x) for p,x in zip(b.blocks.parameters(),old)))
    def test_inactive_expert_no_decay(self):
        m=Learner(Config()); o=GatedAdam(m,Config()); old=copy.deepcopy(m.state_dict()); o.step(torch.ones(2,4))
        for k,v in m.state_dict().items(): self.assertTrue(torch.equal(v,old[k]))
    def test_fractional_displacement(self):
        torch.manual_seed(2); a=Learner(Config()); b=copy.deepcopy(a); oa=GatedAdam(a,Config()); ob=GatedAdam(b,Config()); old=[p.clone() for p in a.blocks.parameters()]
        for m,o in [(a,oa),(b,ob)]:
            for p in m.parameters():
                if p.requires_grad:p.grad=torch.ones_like(p)
        oa.step(torch.full((2,4),.25)); ob.step(torch.ones(2,4))
        for p,q,x in zip(a.blocks.parameters(),b.blocks.parameters(),old): self.assertTrue(torch.allclose(p-x,.25*(q-x),atol=2e-8))
    def test_routing_semantics(self):
        x=torch.randint(0,255,(32,3,32,32),dtype=torch.uint8)
        for mode in ['fixed','dynamic']:
            m=Learner(Config(routing=mode)); _,before=m(x,0)
            with torch.no_grad():m.front[-1].bias.add_(torch.randn(48)*5)
            _,after=m(x,0)
            self.assertEqual(torch.equal(before,after),mode=='fixed')
            self.assertTrue(all(not p.requires_grad for p in m.routers.parameters()))
    def test_joint_probability(self):
        p=Policy(); obs=(torch.randn(2,4,12),torch.randn(9)); m,z,lp,en,v=p.act(obs)
        ds,_=p.distributions(obs)
        expected=ds[0].log_prob(z[0]).sum()+ds[1].log_prob(z[1]).sum()+torch.distributions.Categorical(logits=ds[2].logits[torch.arange(2),z[1]]).log_prob(z[2]).sum()
        self.assertTrue(torch.allclose(lp,expected)); self.assertTrue(torch.equal(m,p.act(obs,z)[0]))
        (-lp).backward(); self.assertGreater(float(p.object.weight.grad.abs().sum()),0)
    def test_expert_equivariance(self):
        p=Policy(interface='factorized'); obs=(torch.randn(2,4,12),torch.randn(9)); perm=torch.tensor([3,1,0,2]); a=p.distributions(obs)[0][0].probs; b=p.distributions((obs[0][:,perm],obs[1]))[0][0].probs
        self.assertTrue(torch.allclose(a[:,perm],b,atol=1e-6))
    def test_progress_masks_only_feedback(self):
        p=Policy(progress=True); loc=torch.randn(2,4,12); g=torch.randn(9); a=p.distributions((loc,g))[0][0].probs
        g[6:8]+=100;loc[:,:,1:3]+=100;loc[:,:,7:10]+=100
        b=p.distributions((loc,g))[0][0].probs; self.assertTrue(torch.equal(a,b))

class FakeData:
    def task(self,classes,role):
        generator=torch.Generator().manual_seed(int(classes[0])*100+{'train':1,'monitor':2,'eval':3,'test':4}[role])
        return torch.randint(0,255,(30,3,32,32),dtype=torch.uint8,generator=generator),torch.arange(30)%5,torch.arange(30)

class EnvironmentMechanisms(unittest.TestCase):
    def test_clone_carries_optimizer_memory_and_future_stream(self):
        from core import Environment
        cfg=Config(steps=20,interval=10,memory=20)
        a=Environment(FakeData(),cfg,900,'train');a.observation();a.update(torch.ones(2,4));b=a.clone()
        a.update(torch.full((2,4),.5));b.update(torch.full((2,4),.5))
        for p,q in zip(a.model.parameters(),b.model.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertEqual(a.cost['updates'],b.cost['updates'])
    def test_memory_budget_baseline_and_no_reacquisition(self):
        from core import Environment
        e=Environment(FakeData(),Config(steps=10,interval=10,memory=40),901,'train')
        e.observation();e.finish(1.2);baseline=e.memory[0]['baseline'].clone()
        e.observation();e.update(torch.ones(2,4));e.finish(1.2)
        self.assertLessEqual(sum(len(m[r][0]) for m in e.memory.values() for r in ['replay','monitor']),40)
        self.assertTrue(torch.equal(e.memory[0]['baseline'],baseline[:10]))
        with self.assertRaises(ValueError):e.dataset(0,'train')
        self.assertEqual(len(e.matrix),2);self.assertEqual(len(e.matrix[0]),4)
    def test_branch_cache_isolation(self):
        from core import Environment
        e=Environment(FakeData(),Config(memory=20),902);e.observation();b=e.clone();b.finish(1.2)
        self.assertEqual(e.k,0)
        self.assertIn((0,'monitor'),e.tasksets)
    def test_initial_success_and_failure_retained(self):
        from core import Environment
        from control import rollout
        for target,success in [(100.,True),(0.,False)]:
            e=Environment(FakeData(),Config(tasks=2,steps=10,target=target,memory=20),903)
            result,_=rollout(e,'fixed:1');self.assertEqual(result['success'],success)
            self.assertEqual(len(result['tasks']),2)
            self.assertEqual(result['tasks'][0]['initial_success'],success)

if __name__=='__main__':unittest.main()

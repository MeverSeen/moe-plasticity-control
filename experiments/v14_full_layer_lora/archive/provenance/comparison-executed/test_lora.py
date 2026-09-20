import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from lora import LoRALinear,insert_lora
from optimizer import GatedAdamW

class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(1400)
 def test_zero_output_initialization_and_input_gradient(self):
  base=nn.Linear(8,6,bias=False);layer=LoRALinear(base,4,4);x=torch.randn(3,8,requires_grad=True);self.assertTrue(torch.equal(layer(x),base(x)));self.assertGreater(float(layer.A.detach().abs().sum()),0);self.assertEqual(float(layer.B.detach().abs().sum()),0);layer(x).square().sum().backward();self.assertIsNone(base.weight.grad);self.assertGreater(float(layer.B.grad.abs().sum()),0);self.assertGreater(float(x.grad.abs().sum()),0)
 def test_full_native_scope_on_meta(self):
  layers=[]
  for _ in range(16):
   experts=[SimpleNamespace(gate_proj=nn.Linear(2048,1024,bias=False,device='meta'),up_proj=nn.Linear(2048,1024,bias=False,device='meta'),down_proj=nn.Linear(1024,2048,bias=False,device='meta')) for _ in range(64)];layers.append(SimpleNamespace(mlp=SimpleNamespace(experts=experts)))
  groups,adapters=insert_lora(SimpleNamespace(model=SimpleNamespace(layers=layers)));self.assertEqual(len(groups),1024);self.assertEqual(len(adapters),3072);self.assertEqual(sum(p.numel() for g in groups for p in g),37748736);self.assertTrue(all(not a.base.weight.requires_grad for a in adapters));self.assertTrue(all(len(g)==6 for g in groups))
 def test_gate_zero_one_fraction_and_moments(self):
  layers=[LoRALinear(nn.Linear(8,6,bias=False),2,2) for _ in range(3)];groups=[[p for a in layers for p in [a.A,a.B]]];opt=GatedAdamW(groups,lr=.03);x=torch.randn(4,8)
  for a in layers:a(x).square().mean().backward()
  opt.step(torch.ones(1));opt.zero_grad();learned=[p.clone() for p in opt.parameters];learned_output=[a(x).detach().clone() for a in layers]
  for p in opt.parameters:p.grad=torch.randn_like(p)
  state=copy.deepcopy(opt.optimizer.state_dict());grads=[p.grad.clone() for p in opt.parameters];reference=[nn.Parameter(p.clone()) for p in opt.parameters];adam=torch.optim.AdamW(reference,lr=.03,weight_decay=.01,foreach=False);adam.load_state_dict(copy.deepcopy(state))
  for p,g in zip(reference,grads):p.grad=g.clone()
  adam.step();opt.step(torch.zeros(1))
  self.assertTrue(all(torch.equal(p,v) for p,v in zip(opt.parameters,learned)));self.assertTrue(all(torch.equal(a(x),y) for a,y in zip(layers,learned_output)));self.assertTrue(any(not torch.equal(a(x),a.base(x)) for a in layers))
  for a,b in zip(opt.optimizer.state.values(),adam.state.values()):
   self.assertTrue(torch.equal(a['exp_avg'],b['exp_avg']));self.assertTrue(torch.equal(a['exp_avg_sq'],b['exp_avg_sq']))
  for multiplier in [1.,.25]:
   with torch.no_grad():
    for p,v in zip(opt.parameters,learned):p.copy_(v)
   opt.optimizer.load_state_dict(copy.deepcopy(state))
   for p,g in zip(opt.parameters,grads):p.grad=g.clone()
   opt.step(torch.tensor([multiplier]))
   for p,old,full in zip(opt.parameters,learned,reference):self.assertTrue(torch.allclose(p,old+multiplier*(full-old),atol=1e-7,rtol=1e-6))
 def test_unrouted_none_grad_skips_adam_decay(self):
  p=nn.Parameter(torch.ones(4));q=nn.Parameter(torch.ones(4));g=GatedAdamW([[p],[q]],lr=.1);p.grad=torch.ones_like(p);g.step(torch.ones(2));self.assertTrue(torch.equal(q,torch.ones(4)));self.assertNotIn(q,g.optimizer.state)
 def test_factor_scaling_not_linear_effective_weight(self):
  A=torch.randn(2,8);B=torch.randn(6,2);da=torch.randn_like(A);db=torch.randn_like(B);m=.5;actual=(B+m*db)@(A+m*da)-B@A;linear=m*((B+db)@(A+da)-B@A);self.assertFalse(torch.allclose(actual,linear));self.assertTrue(torch.allclose(actual-linear,(m*m-m)*db@da,atol=1e-5))
if __name__=='__main__':unittest.main()

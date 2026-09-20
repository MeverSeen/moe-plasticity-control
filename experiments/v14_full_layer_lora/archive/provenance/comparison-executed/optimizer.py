"""Apply one expert multiplier to the actual AdamW displacement of its six factors."""
import torch

class GatedAdamW:
 def __init__(self,groups,lr=2e-4,decay=.01):
  self.groups=groups;self.parameters=[p for group in groups for p in group];self.optimizer=torch.optim.AdamW(self.parameters,lr=lr,weight_decay=decay,betas=(.9,.999),eps=1e-8,foreach=False,fused=self.parameters[0].is_cuda);self.relative=torch.zeros(len(groups));self.active_parameters=0
 def zero_grad(self):self.optimizer.zero_grad(set_to_none=True)
 @torch.no_grad()
 def step(self,multipliers):
  m=torch.as_tensor(multipliers).detach().cpu().flatten();assert len(m)==len(self.groups) and torch.isfinite(m).all() and ((m>=0)&(m<=1)).all();self.active_parameters=sum(p.grad is not None for p in self.parameters)
  old=[[p.detach().clone() for p in group] for group in self.groups];self.optimizer.step();stats=[]
  for i,(group,previous) in enumerate(zip(self.groups,old)):
   ds=[];ws=[];scale=float(m[i])
   for p,before in zip(group,previous):
    delta=p-before;ds.append(delta.square().sum());ws.append(before.square().sum())
    if scale==0:p.copy_(before)
    elif scale!=1:p.copy_(before+scale*delta)
   stats.append((torch.stack(ds).sum()/(torch.stack(ws).sum()+1e-20)).sqrt())
  self.relative=torch.stack(stats).cpu()

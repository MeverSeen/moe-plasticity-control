"""Native expert displacement gating and finite-history PPO/GAE primitives."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ALPHABET=torch.tensor([0.,.25,.5,.75,1.])

class GatedAdamW:
 def __init__(self,groups,lr=2e-5,decay=.01):
  self.groups=groups;self.parameters=[p for group in groups for p in group]
  self.optimizer=torch.optim.AdamW(self.parameters,lr=lr,weight_decay=decay,foreach=False,fused=self.parameters[0].is_cuda)
  self.relative=torch.zeros(len(groups))
 def zero_grad(self):self.optimizer.zero_grad(set_to_none=True)
 @torch.no_grad()
 def step(self,multipliers):
  old=[[p.detach().clone() for p in group] for group in self.groups];self.optimizer.step();stats=[]
  for i,(group,previous) in enumerate(zip(self.groups,old)):
   ds=[];ws=[]
   for p,before in zip(group,previous):
    delta=p-before;ds.append(delta.square().sum());ws.append(before.square().sum());p.copy_(before+float(multipliers[i])*delta)
   stats.append((torch.stack(ds).sum()/(torch.stack(ws).sum()+1e-20)).sqrt())
  self.relative=torch.stack(stats).cpu()

def gae(rewards,values,next_values,terminated,continuation,gamma=1.,lam=.95):
 """Task boundaries continue; true episode ends don't bootstrap; cuts may bootstrap."""
 rewards=np.asarray(rewards,dtype=np.float64);values=np.asarray(values,dtype=np.float64);next_values=np.asarray(next_values,dtype=np.float64);out=np.zeros_like(rewards);last=0.
 for t in reversed(range(len(rewards))):
  delta=rewards[t]+gamma*(not terminated[t])*next_values[t]-values[t]
  last=delta+gamma*lam*bool(continuation[t])*(not terminated[t])*last;out[t]=last
 return out,out+values

def centered_prior(n,index,concentration):
 out=torch.full((*index.shape,n),math.log((1-concentration)/(n-1)),device=index.device)
 return out.scatter(-1,index[...,None],math.log(concentration))

class ActorCritic(nn.Module):
 def __init__(self,base_indices,experts=64):
  super().__init__();self.experts=experts;self.register_buffer('base_indices',torch.tensor(base_indices,dtype=torch.long))
  self.encode=nn.Sequential(nn.Linear(16,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh());self.combine=nn.Sequential(nn.Linear(96,32),nn.Tanh())
  self.global_mode=nn.Linear(32,3);self.layer_mode=nn.Linear(32,3);self.residual=nn.Linear(32,3);self.critic=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
  for head in [self.global_mode,self.layer_mode,self.residual]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
 def forward(self,obs,latent=None,deterministic=False):
  local,global_=obs;h=self.encode(torch.cat([local,global_.expand(*local.shape[:2],-1)],-1));layer=h.mean(1,keepdim=True).expand_as(h);glob=h.mean((0,1),keepdim=True).expand_as(h);h=self.combine(torch.cat([h,layer,glob],-1));g=h.mean((0,1));layer=h.mean(1)
  values=self.critic(g).squeeze(-1);logp=torch.zeros((),device=h.device);entropy=logp.clone();saved=[]
  def draw(logits):
   nonlocal logp,entropy
   d=Categorical(logits=logits);z=latent[len(saved)] if latent is not None else d.logits.argmax(-1) if deterministic else d.sample();saved.append(z);logp=logp+d.log_prob(z).sum();entropy=entropy+d.entropy().sum();return z
  anchor=draw(self.global_mode(g)+centered_prior(3,torch.tensor(1,device=h.device),.90))-1
  offsets=draw(self.layer_mode(layer)+centered_prior(3,torch.ones(2,dtype=torch.long,device=h.device),.94))-1
  residual=draw(self.residual(h)+centered_prior(3,torch.ones((2,self.experts),dtype=torch.long,device=h.device),.96))-1
  indices=(self.base_indices[:,None]+anchor+offsets[:,None]+residual).clamp(0,4)
  return ALPHABET.to(h.device)[indices],saved,logp,entropy,values

def ppo_update(policy,optimizer,trajectories,epochs=3,clip=.2,entropy_coef=.0005,target_kl=.03):
 transitions=[];advantages=[];targets=[]
 for trajectory in trajectories:
  tr=trajectory['transitions'];values=[t['value'] for t in tr];rewards=[0.]*(len(tr)-1)+[trajectory['reward']]
  aa,rr=gae(rewards,values,values[1:]+[0.],[False]*(len(values)-1)+[True],[True]*(len(values)-1)+[False]);transitions.extend(tr);advantages.extend(aa);targets.extend(rr)
 advantage=torch.tensor(advantages,dtype=torch.float32);returns=torch.tensor(targets,dtype=torch.float32);raw=advantage.clone();advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(.1)
 stats=[];original={k:v.detach().clone() for k,v in policy.state_dict().items()}
 for epoch in range(epochs):
  lp=[];entropy=[];pred=[]
  for t in transitions:
   _,_,l,e,v=policy(t['obs'],t['latent']);lp.append(l);entropy.append(e);pred.append(v)
  lp=torch.stack(lp);old=torch.tensor([t['logp'] for t in transitions]);ratio=(lp-old).exp();pred=torch.stack(pred);kl=((ratio-1)-(lp-old)).mean()
  if epoch and float(kl.detach())>target_kl:break
  actor=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean();value_loss=.5*(pred-returns).square().mean();loss=actor+.5*value_loss-entropy_coef*torch.stack(entropy).mean();optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
  stats.append({'epoch':epoch,'actor_loss':float(actor.detach()),'value_loss':float(value_loss.detach()),'kl':float(kl.detach()),'clip_fraction':float(((ratio-1).abs()>clip).float().mean()),'entropy':float(torch.stack(entropy).mean().detach()),'grad_norm':float(norm),'value_mean':float(pred.mean().detach()),'explained_variance':float(1-(returns-pred.detach()).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-8))})
 return {'epochs':stats,'advantage_std':float(raw.std(unbiased=False)),'parameter_change':sum(float((policy.state_dict()[k]-v).float().square().sum()) for k,v in original.items())**.5,'rewards':[t['reward'] for t in trajectories],'decisions':len(transitions)}

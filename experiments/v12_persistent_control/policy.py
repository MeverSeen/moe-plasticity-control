"""Shared context, hierarchical correlated exploration, exact latent log probability."""
import math
import torch
from torch import nn
from torch.distributions import Categorical
from encoder import Policy as Encoder
from core import ALPHABET

TRUNK=torch.tensor([.25,.5,1.])

def prior(n,index,p):
 out=torch.full((*index.shape,n),math.log((1-p)/(n-1)))
 return out.scatter(-1,index[...,None],math.log(p))

class Policy(Encoder):
 def __init__(self,base=0.,trunk=.5,joint=False):
  super().__init__();self.base=float(base);self.trunk=float(trunk);self.joint=joint
  self.base_index=int((ALPHABET==base).nonzero()[0]);self.trunk_index=int((TRUNK==trunk).nonzero()[0])
  self.keep=nn.Linear(32,2);self.trunk_head=nn.Linear(32,3);self.anchor=nn.Linear(32,6);self.layer_head=nn.Linear(32,6)
  for head in [self.actions,self.keep,self.trunk_head,self.anchor,self.layer_head]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
 def features(self,obs):
  loc,g=obs;h=self.encode(torch.cat([loc,g.expand(*loc.shape[:2],-1)],-1));layer=h.mean(1,keepdim=True).expand_as(h);glob=h.mean((0,1),keepdim=True).expand_as(h)
  return self.combine(torch.cat([h,layer,glob],-1))
 def act(self,obs,latent=None,deterministic=False):
  h=self.features(obs);g=h.mean((0,1));layer=h.mean(1);stored=[];lp=torch.tensor(0.);ent=torch.tensor(0.);pos=0
  def draw(logits):
   nonlocal lp,ent,pos
   d=Categorical(logits=logits);z=(d.logits.argmax(-1) if deterministic else d.sample()) if latent is None else latent[pos]
   stored.append(z);pos+=1;lp=lp+d.log_prob(z).sum();ent=ent+d.entropy().sum();return z
  keep=draw(self.keep(g)) # index0 is deterministic baseline; initial probability .5
  if int(keep)==0:
   m=torch.full(obs[0].shape[:2],self.base);return (m,self.trunk),stored,lp,ent,torch.tensor(0.)
  t=draw(self.trunk_head(g)+prior(3,torch.tensor(self.trunk_index),.5))
  anchor=draw(self.anchor(g)+prior(6,torch.tensor(self.base_index),.5))
  modes=draw(self.layer_head(layer)+prior(6,anchor.expand(len(layer)),.8))
  logits=self.actions(h)+prior(6,modes[:,None].expand(h.shape[:2]),.8)
  if self.joint:idx=draw(logits);m=ALPHABET[idx]
  else:
   who=draw(self.object(h).squeeze(-1));idx=draw(logits[torch.arange(len(layer)),who]);m=torch.full(h.shape[:2],self.base);m[torch.arange(len(layer)),who]=ALPHABET[idx]
  return (m,float(TRUNK[t])),stored,lp,ent,torch.tensor(0.)

class Baseline:
 def __init__(self,expert=0.,trunk=.5,rule='fixed',joint=True):self.base=float(expert);self.trunk=float(trunk);self.rule=rule;self.joint=joint
 def act(self,obs,latent=None,deterministic=False):
  loc,g=obs;m=torch.full(loc.shape[:2],self.base);t=self.trunk
  if self.rule!='fixed':
   # Prespecified simple rules, with equal observation access and valid interface actions.
   high=(g[2]>.25) if self.rule=='progress' else (g[6]<.03 and g[2]>.1)
   t=1. if high else .25 if self.rule=='feedback' and g[6]>.08 else self.trunk
   magnitude=1. if high else self.base
   if self.joint:m.fill_(magnitude)
   else:
    risk=loc[:,:,0]+loc[:,:,1]+loc[:,:,2];who=(loc[:,:,3]-risk).argmax(1);m[torch.arange(len(m)),who]=magnitude
  return (m,t),[],torch.tensor(0.),torch.tensor(0.),torch.tensor(0.)

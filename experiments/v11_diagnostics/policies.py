"""Probability-controlled initialization and deterministic-base sparse repairs."""
import math
import torch
from torch import nn
from torch.distributions import Categorical
from core import ALPHABET
from control import Policy

class FixedPolicy:
    def __init__(self,expert=.25,trunk=1.):self.expert=expert;self.trunk=trunk
    def act(self,obs,latent=None,deterministic=False):
        m=torch.full(obs[0].shape[:2],float(self.expert),dtype=torch.float32)
        return (m,self.trunk),[],torch.tensor(0.),torch.tensor(0.),torch.tensor(0.)

def initialize_head(head,index,probability):
    with torch.no_grad():
        head.weight.zero_();head.bias.fill_(math.log((1-probability)/(len(head.bias)-1)));head.bias[index]=math.log(probability)

def warm_policy(base=.25,p=.98):
    policy=Policy();index=int((ALPHABET==base).nonzero()[0]);initialize_head(policy.actions,index,p);return policy

class RepairPolicy(Policy):
    def __init__(self,base=.25,p=.9,trunk=False):
        super().__init__();self.base=base;self.with_trunk=trunk
        initialize_head(self.actions,int((ALPHABET==base).nonzero()[0]),p)
        if trunk:
            self.trunk_head=nn.Linear(9,3);initialize_head(self.trunk_head,2,.98)
    def act(self,obs,latent=None,deterministic=False):
        ds,value=self.distributions(obs);obj=ds[1]
        if latent is None:
            who=obj.logits.argmax(-1) if deterministic else obj.sample()
            mag=Categorical(logits=ds[2].logits[torch.arange(len(who)),who]);what=mag.logits.argmax(-1) if deterministic else mag.sample()
            latent=[who,what]
            if self.with_trunk:
                trunk=Categorical(logits=self.trunk_head(obs[1]));latent.append(trunk.logits.argmax(-1) if deterministic else trunk.sample())
        who,what=latent[:2];mag=Categorical(logits=ds[2].logits[torch.arange(len(who)),who])
        logp=obj.log_prob(who).sum()+mag.log_prob(what).sum();entropy=obj.entropy().sum()+mag.entropy().sum()
        m=torch.full(obs[0].shape[:2],float(self.base),dtype=torch.float32);m[torch.arange(len(who)),who]=ALPHABET[what]
        if self.with_trunk:
            trunk=Categorical(logits=self.trunk_head(obs[1]));logp+=trunk.log_prob(latent[2]);entropy+=trunk.entropy();m=(m,float(torch.tensor([.25,.5,1.])[latent[2]]))
        return m,latent,logp,entropy,value

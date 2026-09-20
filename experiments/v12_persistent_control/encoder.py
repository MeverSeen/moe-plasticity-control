"""Shared policies with exact latent-action probabilities and on-policy updates."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from core import ALPHABET

class Policy(nn.Module):
    def __init__(self,local=False,progress=False,interface='structured'):
        super().__init__(); self.local=local; self.progress=progress; self.interface=interface
        self.encode=nn.Sequential(nn.Linear(21,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.combine=nn.Sequential(nn.Linear(96,32),nn.Tanh())
        self.actions=nn.Linear(32,6); self.object=nn.Linear(32,1); self.value=nn.Linear(32,1)
    def distributions(self,obs):
        loc,glob=[x.clone() for x in obs]
        if self.local: glob[6:8]=0
        if self.progress:
            glob[6:8]=0; loc[:,:,1:3]=0; loc[:,:,7:10]=0
        h=self.encode(torch.cat([loc,glob.expand(*loc.shape[:2],-1)],-1))
        layer=h.mean(1,keepdim=True).expand_as(h); global_h=h.mean((0,1),keepdim=True).expand_as(h)
        if not self.local: h=self.combine(torch.cat([h,layer,global_h],-1))
        logits=self.actions(h)
        value=self.value(h.mean((0,1))).squeeze()
        if self.interface=='factorized': return [Categorical(logits=logits)],value
        if self.interface=='global': return [Categorical(logits=logits.mean((0,1)))],value
        if self.interface=='groups':
            # Historical usage threshold defines two groups; same alphabet, no 15-action range confound.
            mask=loc[:,:,0]>=.5
            group_logits=torch.stack([logits[mask].mean(0) if mask.any() else logits.mean((0,1)),logits[~mask].mean(0) if (~mask).any() else logits.mean((0,1))])
            return [Categorical(logits=group_logits)],value
        return [Categorical(logits=logits.mean(1)),Categorical(logits=self.object(h).squeeze(-1)),Categorical(logits=logits)],value
    def act(self,obs,latent=None,deterministic=False):
        ds,v=self.distributions(obs); shape=obs[0].shape[:2]
        if latent is None:
            latent=[]
            for i,d in enumerate(ds):
                if self.interface=='structured' and i==2:
                    logits=d.logits[torch.arange(shape[0]),latent[1]]; d=Categorical(logits=logits)
                latent.append(d.logits.argmax(-1) if deterministic else d.sample())
        logp=torch.tensor(0.); ent=torch.tensor(0.)
        for i,d in enumerate(ds):
            if self.interface=='structured' and i==2: d=Categorical(logits=d.logits[torch.arange(shape[0]),latent[1]])
            logp=logp+d.log_prob(latent[i]).sum(); ent=ent+d.entropy().sum()
        if self.interface=='structured':
            idx=latent[0][:,None].expand(shape).clone(); idx[torch.arange(shape[0]),latent[1]]=latent[2]
        elif self.interface=='factorized': idx=latent[0]
        elif self.interface=='global': idx=latent[0].expand(shape)
        else: idx=torch.where(obs[0][:,:,0]>=.5,latent[0][0],latent[0][1])
        return ALPHABET[idx],latent,logp,ent,v


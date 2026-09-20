"""Per-layer base, correction count, ordered distinct indices and non-base values.

Two permutations of a two-expert correction are distinct latent actions. The PPO
policy is explicitly this ordered latent distribution. No unused slots are drawn.
"""
import torch
from torch import nn
from torch.distributions import Categorical
ALPHABET=torch.tensor([0.,.25,.5,.75,1.])

def distribution(logits,valid,epsilon):
 assert valid.any();raw=logits.masked_fill(~valid,-torch.inf).softmax(-1);return Categorical(probs=(1-epsilon)*raw+epsilon*valid.float()/valid.sum())

def draw_action(obs,base_logits,count_logits,who_logits,magnitude_logits,epsilon,latent=None,deterministic=False):
 local,_=obs;layers,n=local.shape[:2];m=torch.empty((layers,n));saved=[];lp=torch.tensor(0.);entropy=torch.tensor(0.);valid5=torch.ones(5,dtype=torch.bool)
 def draw(logits,valid):
  nonlocal lp,entropy
  d=distribution(logits,valid,epsilon);z=latent[len(saved)] if latent is not None else d.probs.argmax() if deterministic else d.sample();assert valid[z];saved.append(z);lp=lp+d.log_prob(z);entropy=entropy+d.entropy();return int(z)
 for layer in range(layers):
  base=draw(base_logits(layer),valid5);count=draw(count_logits(layer,base),torch.arange(3)<=n);m[layer].fill_(float(ALPHABET[base]));valid_expert=torch.ones(n,dtype=torch.bool)
  for slot in range(count):
   who=draw(who_logits(layer,base,slot,valid_expert),valid_expert);valid_expert[who]=False
   valid_mag=valid5.clone();valid_mag[base]=False;value=draw(magnitude_logits(layer,base,who,slot),valid_mag);m[layer,who]=ALPHABET[value]
 if latent is not None:assert len(saved)==len(latent),'Unused or missing latent factors'
 return m,saved,lp,entropy

class ActorCritic(nn.Module):
 def __init__(self,experts=64,layers=16,epsilon=.02):
  super().__init__();self.experts=experts;self.layers=layers;self.epsilon=epsilon;self.encode=nn.Sequential(nn.Linear(21,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh());self.combine=nn.Sequential(nn.Linear(128,32),nn.Tanh());self.base_head=nn.Linear(32,5);self.layer_bias=nn.Parameter(torch.zeros(layers,5));self.count_head=nn.Linear(37,3);self.who_head=nn.Linear(32,1);self.who_skip=nn.Linear(10,1,bias=False);self.mag_head=nn.Linear(37,5);self.mag_bias=nn.Embedding(5,5);self.critic=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)
 def forward(self,obs,latent=None,deterministic=False):
  local,global_=obs;assert local.shape==(self.layers,self.experts,10);position=torch.linspace(0,1,self.layers)[:,None,None].expand(self.layers,self.experts,1);h=self.encode(torch.cat([local,position,global_.expand(*local.shape[:2],-1)],-1));layer=h.mean(1,keepdim=True).expand_as(h);maximum=h.max(1,keepdim=True).values.expand_as(h);glob=h.mean((0,1),keepdim=True).expand_as(h);h=self.combine(torch.cat([h,layer,maximum,glob],-1));layer=h.mean(1);g=h.mean((0,1))
  onehot=lambda base:torch.nn.functional.one_hot(torch.tensor(base),5).float()
  m,z,lp,entropy=draw_action(obs,lambda l:self.base_head(layer[l])+self.layer_bias[l],lambda l,b:self.count_head(torch.cat([layer[l],onehot(b)])),lambda l,b,slot,valid:self.who_head(h[l]).squeeze(-1)+self.who_skip(local[l]).squeeze(-1),lambda l,b,w,slot:self.mag_head(torch.cat([h[l,w],onehot(b)]))+self.mag_bias(torch.tensor(b)),self.epsilon,latent,deterministic)
  return m,z,lp,entropy,self.critic(g).squeeze(-1)
 act=forward

class Rule:
 def __init__(self,base_indices,epsilon=.02,mode='rule'):
  self.base=list(base_indices);self.epsilon=epsilon;self.mode=mode
 def act(self,obs,latent=None,deterministic=True):
  local,g=obs;n=local.shape[1];gap=float(g[2])*5;improvement=float(g[1])*5;potential=float(g[5])*.1;risk=(local[:,:,0]-local[:,:,1])/8
  bases=[max(0,b-1) if potential>.10 else min(4,b+1) if gap>.50 and improvement<.02 else b for b in self.base] if self.mode=='rule' else self.base
  def peaked(n,index):
   z=torch.full((n,),-torch.inf);z[index]=0.;return z
  def base(l):return peaked(5,bases[l])
  def count(l,b):return peaked(3,min(2,int((risk[l]>.02).sum())) if self.mode=='rule' and b>0 else 0)
  def who(l,b,slot,valid):
   scores=risk[l].masked_fill(~valid,-torch.inf);return peaked(n,int(scores.argmax())) # index order breaks exact ties
  def magnitude(l,b,w,slot):return peaked(5,b-1 if b>0 else 1)
  m,z,lp,entropy=draw_action(obs,base,count,who,magnitude,self.epsilon,latent,deterministic);return m,z,lp,entropy,torch.tensor(0.)

def shaped_rewards(cached_potentials,full_monitor_F,failures,scale=.1,kappa=3.):
 assert scale>0 and len(cached_potentials)>0
 z=list(map(float,cached_potentials));rewards=[(a-b)/scale for a,b in zip(z,z[1:])]+[(z[-1]-full_monitor_F)/scale-kappa*failures]
 expected=-full_monitor_F/scale-kappa*failures+z[0]/scale;assert abs(sum(rewards)-expected)<1e-8
 return rewards

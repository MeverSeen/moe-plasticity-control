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


def rule(obs,name):
    loc,g=obs; shape=loc.shape[:2]
    if name.startswith('fixed:'): return torch.full(shape,float(name.split(':')[1]))
    if name=='schedule': return torch.full(shape,1.5 if g[4]<.5 else .25)
    risk=loc[:,:,0]+loc[:,:,1]+loc[:,:,2]+loc[:,:,8]*.25
    if name=='local': return torch.where(risk>1.,.25,1.)
    if name=='coordinated':
        # Same observation access; current loss progress and measured old damage set quota.
        keep=max(1,shape[1]//2); result=torch.full(shape,.25)
        score=loc[:,:,3]-risk*(1+g[6].clamp(0,1))
        chosen=score.topk(keep,dim=1).indices; result.scatter_(1,chosen,1.5 if g[2]>.3 else 1.)
        return result
    raise ValueError(name)


def rollout(env,policy,stochastic=False,collect=False):
    transitions=[]; start=time.perf_counter(); action_hist=[]; trace=[]
    while env.k<env.cfg.tasks:
        obs=env.observation(); loss=float(obs[1][0])
        if loss<=env.cfg.target or env.t>=env.cfg.steps:
            env.finish(loss); continue
        if env.k==0: m=torch.ones_like(env.previous)
        elif isinstance(policy,str): m=rule(obs,policy)
        else:
            with torch.no_grad(): m,z,lp,en,v=policy.act(obs,deterministic=not stochastic)
            trace.append({'task':env.k,'step':env.t,'latent':[a.tolist() for a in z],'behavior_logp':float(lp),'observation':[a.tolist() for a in obs]})
            if collect: transitions.append({'obs':tuple(x.clone() for x in obs),'latent':[x.clone() for x in z],'old_logp':float(lp),'value':float(v)})
        action_hist.append(m.tolist()); env.update(m)
    # Reward uses retained old monitor only, with baselines at each sample's task completion.
    damage=[]
    for j,mem in env.memory.items():
        if j==env.cfg.tasks-1: continue
        ll,_,_=env.measure(j,mem['monitor']); damage.append(float((ll-mem['baseline']).mean().clamp_min(0)))
    result=env.result(); result['proxy_forgetting']=float(np.mean(damage)); result['wall_seconds']=time.perf_counter()-start
    result['actions']=action_hist; result['policy_trace']=trace
    result['reward']=-result['proxy_forgetting']/.1-3*sum(not t['success'] for t in result['tasks'])
    return result,transitions


def train_batch(policy,optimizer,roots,algorithm,group=4):
    results=[]; trajectories=[]; start=time.perf_counter(); copy_sec=0
    for root in roots:
        group_results=[]; group_traj=[]
        for _ in range(group):
            before=time.perf_counter(); env=root.clone(); copy_sec+=time.perf_counter()-before
            r,tr=rollout(env,policy,True,True); r['branch_cost']={k:r['cost'][k]-root.cost[k] for k in root.cost}; group_results.append(r); group_traj.append(tr)
        rewards=np.array([r['reward'] for r in group_results]); advantage=(rewards-rewards.mean())/max(rewards.std(),.1)
        for r,tr,a in zip(group_results,group_traj,advantage):
            for t in tr: t['return']=r['reward']; t['advantage']=float(a) if algorithm=='grpo' else r['reward']-t['value']
        results.extend(group_results); trajectories.extend(group_traj)
    flat=[t for tr in trajectories for t in tr]
    if not flat: return {'results':results,'decisions':0,'wall_seconds':time.perf_counter()-start}
    if algorithm=='ppo':
        aa=np.array([t['advantage'] for t in flat]); scale=max(aa.std(),.1)
        for t in flat: t['advantage']=(t['advantage']-aa.mean())/scale
    old=copy.deepcopy(policy.state_dict()); losses=[]; ratios=[]
    # Two epochs retain exact joint clipping; same schedule for GRPO/PPO.
    for _ in range(2):
        optimizer.zero_grad(); terms=[]
        for tr in trajectories:
            tt=[]
            for t in tr:
                _,_,lp,ent,value=policy.act(t['obs'],t['latent']); ratio=(lp-t['old_logp']).exp(); adv=t['advantage']
                term=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)-.002*ent
                if algorithm=='ppo': term=term+.1*(value-t['return'])**2
                tt.append(term); ratios.append(float(ratio.detach()))
            if tt: terms.append(torch.stack(tt).mean())
        loss=torch.stack(terms).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
    delta=sum(float((policy.state_dict()[k]-v).square().sum()) for k,v in old.items())**.5
    return {'results':results,'decisions':len(flat),'losses':losses,'parameter_change':delta,'ratio_min':min(ratios),'ratio_max':max(ratios),'copy_seconds':copy_sec,'wall_seconds':time.perf_counter()-start,'reward_std':float(np.std([r['reward'] for r in results]))}

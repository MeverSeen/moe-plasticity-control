"""Persistent actions with common ten-update stopping checks; GRPO only."""
import copy,time
import numpy as np
import torch

def advance(env,policy,hold,stochastic=False,collect=False):
 obs=env.observation();loss=float(obs[1][0]);transition=None
 if loss<=env.cfg.target or env.t>=env.cfg.steps:env.finish(loss);return transition
 if env.k==0:action=(torch.ones_like(env.previous),1.)
 else:
  if env.held_action is None or env.t>=env.held_until:
   with torch.no_grad():action,z,lp,entropy,_=policy.act(obs,deterministic=not stochastic)
   env.held_action=(action[0].clone(),action[1]);env.held_until=env.t+hold;env.cost['control_decisions']+=1
   env.control_trace.append({'task':env.k,'step':env.t,'duration':0,'requested_hold':hold,'experts':action[0].tolist(),'trunk':action[1],'latent':[v.tolist() for v in z],'behavior_logp':float(lp),'observation':[v.tolist() for v in obs]})
   if collect:transition={'obs':tuple(v.clone() for v in obs),'latent':[v.clone() for v in z],'old_logp':float(lp)}
  action=env.held_action
 before=env.t;env.update(action)
 if env.k>0:env.control_trace[-1]['duration']+=env.t-before
 return transition

def rollout(env,policy,hold,stochastic=False,collect=False):
 start=time.perf_counter();transitions=[]
 while env.k<env.cfg.tasks:
  t=advance(env,policy,hold,stochastic,collect)
  if t is not None:transitions.append(t)
 reward_tasks=[]
 for j,mem in env.memory.items():
  if j==env.cfg.tasks-1:continue
  ll,_,_=env.measure(j,mem['monitor']);delta=ll-mem['baseline'];reward_tasks.append({'task':j,'n':len(ll),'positive':float(delta.mean().clamp_min(0))})
 r=env.result();r['proxy_forgetting']=float(np.mean([v['positive'] for v in reward_tasks]));r['reward']=-r['proxy_forgetting']/.1-3*sum(not v['success'] for v in r['tasks']);r['reward_tasks']=reward_tasks;r['policy_trace']=env.control_trace;r['hold']=hold;r['wall_seconds']=time.perf_counter()-start
 return r,transitions

def train_batch(policy,opt,root,hold,group=4):
 results=[];trajs=[];start=time.perf_counter()
 for _ in range(group):
  e=root.clone();r,tr=rollout(e,policy,hold,True,True);r['branch_cost']={k:r['cost'][k]-root.cost[k] for k in root.cost};results.append(r);trajs.append(tr)
 rewards=np.array([r['reward'] for r in results]);adv=(rewards-rewards.mean())/max(rewards.std(),.1);old=copy.deepcopy(policy.state_dict());ratios=[];losses=[]
 if any(trajs):
  for _ in range(2):
   opt.zero_grad();terms=[]
   for tr,a in zip(trajs,adv):
    pieces=[]
    for t in tr:
     _,_,lp,entropy,_=policy.act(t['obs'],t['latent']);ratio=(lp-t['old_logp']).exp();ratios.append(float(ratio.detach()));pieces.append(-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)-.002*entropy)
    if pieces:terms.append(torch.stack(pieces).mean())
   loss=torch.stack(terms).mean();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();losses.append(float(loss.detach()))
 return {'results':results,'reward_std':float(rewards.std()),'advantages':adv.tolist(),'decisions':sum(map(len,trajs)),'optimizer_steps':len(losses),'losses':losses,'ratio_range':[min(ratios),max(ratios)] if ratios else [],'parameter_change':sum(float((policy.state_dict()[k]-v).square().sum()) for k,v in old.items())**.5,'wall_seconds':time.perf_counter()-start}

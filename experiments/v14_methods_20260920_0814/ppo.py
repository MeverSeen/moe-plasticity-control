import numpy as np
import torch

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


def ppo_update(policy,optimizer,trajectories,epochs=3,clip=.2,entropy_coef=0.,target_kl=.03):
 transitions=[];advantages=[];targets=[]
 for trajectory in trajectories:
  tr=trajectory['transitions'];values=[t['value'] for t in tr];rewards=trajectory['rewards'];assert len(rewards)==len(tr)
  if not tr:continue
  aa,rr=gae(rewards,values,values[1:]+[0.],[False]*(len(values)-1)+[True],[True]*(len(values)-1)+[False]);transitions.extend(tr);advantages.extend(aa);targets.extend(rr)
 if not transitions:return {'skipped':'No policy decisions in completed histories','decisions':0,'parameter_change':0.,'rewards':[sum(t['rewards']) for t in trajectories]}
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
 with torch.no_grad():
  new=torch.stack([policy(t['obs'],t['latent'])[2] for t in transitions]);old=torch.tensor([t['logp'] for t in transitions]);ratio=(new-old).exp();post={'kl_old_to_current':float((ratio-1-(new-old)).mean()),'clip_fraction':float(((ratio-1).abs()>clip).float().mean())}
  if all('ref_logp' in t for t in transitions):
   ref=torch.tensor([t['ref_logp'] for t in transitions]);d=ref-new;post['kl_current_to_clone_importance_estimate']=float((ratio*(d.exp()-1-d)).mean())
 return {'epochs':stats,'post_update':post,'advantage_std':float(raw.std(unbiased=False)),'parameter_change':sum(float((policy.state_dict()[k]-v).float().square().sum()) for k,v in original.items())**.5,'rewards':[sum(t['rewards']) for t in trajectories],'decisions':len(transitions)}

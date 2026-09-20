"""Post hoc probability-change diagnosis on saved initial stochastic observations.
No learner/data rollout or checkpoint selection; exact latent KL by chain rule.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical,kl_divergence
from learn import factory
from diagnose import save

def audit(path,arm,plan):
 start=time.perf_counter();initial=factory(arm,plan['base']);final=factory(arm,plan['base'])
 initial.load_state_dict(torch.load(path/'epoch0.pt',weights_only=True));final.load_state_dict(torch.load(path/f"epoch{plan['rounds']}.pt",weights_only=True))
 raw=json.loads((path/'audit.json').read_text())['0']['stochastic'];observations=[tuple(torch.tensor(v) for v in t['observation']) for r in raw for t in r['policy_trace']];index=0;repair=arm.startswith('repair');records=[]
 with torch.no_grad():
  for obs in observations:
   ds0,_=initial.distributions(obs);ds1,_=final.distributions(obs)
   kl=(kl_divergence(ds0[1],ds1[1])+(ds0[1].probs*kl_divergence(ds0[2],ds1[2])).sum(-1)).sum()
   if not repair:kl+=kl_divergence(ds0[0],ds1[0]).sum()
   base_probs=[];margins=[]
   for p,ds in [(initial,ds0),(final,ds1)]:
    layer=(ds[1].probs*ds[2].probs[:,:,index]).sum(-1)
    if not repair:layer*=ds[0].probs[:,index]
    prob=layer.prod()
    if arm=='repair_trunk':prob*=Categorical(logits=p.trunk_head(obs[1])).probs[2]
    base_probs.append(float(prob));logits=ds[2].logits;margins.append(float((logits[:,:,index]-logits[:,:,1:].max(-1).values).min()))
   if arm=='repair_trunk':kl+=kl_divergence(Categorical(logits=initial.trunk_head(obs[1])),Categorical(logits=final.trunk_head(obs[1])))
   m0=initial.act(obs,deterministic=True)[0];m1=final.act(obs,deterministic=True)[0]
   if isinstance(m0,tuple):changed=not torch.equal(m0[0],m1[0]) or m0[1]!=m1[1]
   else:changed=not torch.equal(m0,m1)
   records.append({'latent_kl_initial_to_final':max(0,float(kl)),'initial_all_base_probability':base_probs[0],'final_all_base_probability':base_probs[1],'initial_magnitude_base_margin':margins[0],'final_magnitude_base_margin':margins[1],'greedy_changed':changed})
 summary={k:float(np.mean([r[k] for r in records])) for k in records[0]};summary['latent_kl_max']=max(r['latent_kl_initial_to_final'] for r in records);summary['final_base_margin_min']=min(r['final_magnitude_base_margin'] for r in records)
 save(path/'policy_change.json',{'scope':'post hoc, saved initial stochastic audit observations; no new learner rollout; exact latent KL, not marginal allocation KL; all-base includes trunk1 when controlled','observations':len(records),'policy_forward_calls':4*len(records),'summary':summary,'seconds':time.perf_counter()-start});print(path.name,summary,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--learning',type=Path,default=Path('runs/v11-learning'));p.add_argument('--plan',type=Path,default=Path(__file__).with_name('learning_plan.json'));a=p.parse_args();plan=json.loads(a.plan.read_text());torch.set_num_threads(1)
 for arm in plan['arms']:
  for seed in plan['controller_seeds']:
   path=a.learning/f'{arm}_{seed}'
   if (path/'cost.json').exists() and not (path/'policy_change.json').exists():audit(path,arm,plan)
if __name__=='__main__':main()

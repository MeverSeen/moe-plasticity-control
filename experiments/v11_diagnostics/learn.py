"""Independent CPU learning jobs, frozen plan, paired greedy/stochastic evaluation."""
import argparse,hashlib,json,resource,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from core import Config,Data,Environment
from control import Policy,rollout,train_batch
from policies import FixedPolicy,RepairPolicy,warm_policy
from diagnose import save

def factory(arm,base):
 if arm=='scratch_grpo':return Policy()
 if arm=='warm_grpo':return warm_policy(base)
 return RepairPolicy(base,trunk=arm=='repair_trunk')

def evaluate(data,cfg,seeds,policy,action_seed,stochastic=False):
 rows=[]
 for seed in seeds:
  e=Environment(data,cfg,seed,'dev');torch.manual_seed(action_seed+seed)
  r,_=rollout(e,policy,stochastic=stochastic);r['action_seed']=action_seed+seed;r['execution']='stochastic' if stochastic else 'greedy';rows.append(r)
 return rows

def score(rows):return (-float(np.mean([r['success'] for r in rows])),float(np.mean([r['forgetting'] for r in rows])),float(np.mean([r['final_loss'] for r in rows])))

def distribution_audit(policy,obs,base):
 counts={str(float(m)):0 for m in [0,.25,.5,.75,1,1.5]};unchanged=[];trunk_counts={}
 with torch.no_grad(),torch.random.fork_rng():
  torch.manual_seed(1169999)
  for _ in range(1000):
   m,*_=policy.act(obs)
   if isinstance(m,tuple):
    t=str(float(m[1]));trunk_counts[t]=trunk_counts.get(t,0)+1;m=m[0]
   unchanged.append(bool((m==base).all()))
   for v in m.flatten().tolist():counts[str(float(v))]+=1
  greedy=policy.act(obs,deterministic=True)[0];greedy=greedy[0] if isinstance(greedy,tuple) else greedy
 return {'trunk_counts':trunk_counts,'draws':1000,'expert_draws':8000,'multiplier_counts':counts,'whole_allocation_base_fraction':float(np.mean(unchanged)),'base_expert_fraction':counts[str(float(base))]/8000,'mean_multiplier':sum(float(k)*v for k,v in counts.items())/8000,'greedy':greedy.tolist()}

def main(args):
 torch.set_num_threads(1);plan=json.loads(args.plan.read_text());out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();data=Data(args.data);cfg=Config(**plan['config'])
 for name,digest in plan['source_sha256'].items():
  if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen source changed: '+name)
 if args.arm=='repair_balanced':cfg.reward_mode='balanced'
 if args.arm not in plan['arms'] or args.seed not in plan['controller_seeds']:raise ValueError('not in locked plan')
 save(out/'job.json',{'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),'arm':args.arm,'seed':args.seed,'config':asdict(cfg),'base':plan['base']})
 roots=[]
 for seed in plan['train_roots']:
  e=Environment(data,cfg,seed,'train')
  while e.k==0:
   o=e.observation();l=float(o[1][0])
   if l<=cfg.target or e.t>=cfg.steps:e.finish(l)
   else:e.update(torch.ones(cfg.layers,cfg.experts))
  roots.append(e)
 save(out/'prefix_cost.json',[e.cost for e in roots]);torch.manual_seed(args.seed);policy=factory(args.arm,plan['base']);opt=torch.optim.Adam(policy.parameters(),lr=plan['policy_lr']);algo='ppo' if args.arm=='repair_ppo' else 'grpo'
 obs=roots[0].clone().observation();save(out/'initial_distribution.json',distribution_audit(policy,obs,plan['base']))
 history=[];dev={}
 for u in range(plan['rounds']+1):
  if u in plan['checkpoints']:
   torch.save(policy.state_dict(),out/f'epoch{u}.pt');dev[u]={}
   for mode in ['greedy','stochastic']:
    dev[u][mode]=evaluate(data,cfg,plan['selection_roots'],policy,plan['selection_action_seed'],mode=='stochastic')
   save(out/'development.json',dev);print(args.arm,args.seed,'dev',u,score(dev[u]['greedy']),score(dev[u]['stochastic']),flush=True)
  if u==plan['rounds']:break
  root=roots[u%len(roots)];torch.manual_seed(args.seed*10000+u)
  batch=train_batch(policy,opt,[root],algo,plan['group']);batch['round']=u;batch['root']=root.seed;history.append(batch);save(out/'training.json',history)
  print(args.arm,args.seed,'round',u,'seconds',round(batch['wall_seconds'],2),flush=True)
 selected={mode:min(dev,key=lambda u:score(dev[u][mode])) for mode in ['greedy','stochastic']}
 # Mode-specific deployment selection is explicit. Cross-mode evaluation of each selected checkpoint is kept.
 checkpoints=sorted(set([0,plan['rounds']]+list(selected.values())))
 locked={'selected':selected,'checkpoints':checkpoints,'model_sha256':{u:hashlib.sha256((out/f'epoch{u}.pt').read_bytes()).hexdigest() for u in checkpoints},'scope':'new histories within already used development class pool, not new-class final confirmation'};save(out/'locked_selection.json',locked)
 audit={}
 for u in checkpoints:
  policy.load_state_dict(torch.load(out/f'epoch{u}.pt',weights_only=True));audit[u]={}
  for mode in ['greedy','stochastic']:
   audit[u][mode]=evaluate(data,cfg,plan['audit_roots'],policy,plan['audit_action_seed'],mode=='stochastic')
  save(out/'audit.json',audit)
 save(out/'final_distribution.json',distribution_audit(policy,obs,plan['base']))
 save(out/'cost.json',{'wall_seconds':time.perf_counter()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'training_updates':sum(r['branch_cost']['updates'] for b in history for r in b['results']),'training_seconds':sum(b['wall_seconds'] for b in history),'prefix_updates':sum(e.cost['updates'] for e in roots),'dev_updates':sum(r['cost']['updates'] for modes in dev.values() for rows in modes.values() for r in rows),'audit_updates':sum(r['cost']['updates'] for modes in audit.values() for rows in modes.values() for r in rows)})
 print('complete',args.arm,args.seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',default='.cache/datasets/cifar-complete.tar.gz');main(p.parse_args())

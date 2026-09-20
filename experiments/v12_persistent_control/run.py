"""Frozen learner-update budgets, mode-specific selection, independent dev replication."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from core import Config,Data,Environment
from policy import Policy,Baseline
from control import advance,rollout,train_batch
from calibrate import save,score

def evaluate(data,cfg,roots,policy,hold,action_seed,stochastic=False):
 rows=[]
 for root in roots:
  e=Environment(data,cfg,root,'dev');torch.manual_seed(root+action_seed);r,_=rollout(e,policy,hold,stochastic);r['action_seed']=root+action_seed;r['execution']='stochastic' if stochastic else 'greedy';rows.append(r)
 return rows

def sample_audit(policy,obs,base,trunk):
 result={'draws':1000,'keep':0,'all_base':0,'multi_expert':0,'joint_trunk_and_experts':0,'trunk_counts':{},'expert_counts':{},'changed_expert_histogram':{}}
 with torch.no_grad(),torch.random.fork_rng():
  torch.manual_seed(1269999)
  for _ in range(result['draws']):
   (m,t),z,*_=policy.act(obs);changed=int((m!=base).sum());result['keep']+=int(z[0])==0;result['all_base']+=changed==0 and t==trunk;result['multi_expert']+=changed>2;result['joint_trunk_and_experts']+=changed>0 and t!=trunk
   for dic,key in [('trunk_counts',str(t)),('changed_expert_histogram',str(changed))]:result[dic][key]=result[dic].get(key,0)+1
   for value in m.flatten().tolist():key=str(value);result['expert_counts'][key]=result['expert_counts'].get(key,0)+1
 return result

def main(a):
 torch.set_num_threads(1);plan=json.loads(a.plan.read_text());out=a.out;out.mkdir(parents=True,exist_ok=False)
 assert a.seed in plan['controller_seeds'] and a.arm in plan['arms']
 for name,h in plan['sources'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h,name
 start=time.perf_counter();data=Data('.cache/datasets/cifar-complete.tar.gz');cfg=Config(**plan['config']);hold=plan['arms'][a.arm]['hold'];joint=plan['arms'][a.arm]['joint'];base,trunk=plan['base']['expert'],plan['base']['trunk']
 save(out/'job.json',{'arm':a.arm,'controller_seed':a.seed,'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'hold':hold,'joint':joint})
 roots=[]
 for seed in plan['train_roots']:
  e=Environment(data,cfg,seed,'train')
  while e.k==0:advance(e,Baseline(base,trunk),hold)
  roots.append(e)
 save(out/'prefix_cost.json',[r.cost for r in roots]);torch.manual_seed(a.seed);p=Policy(base,trunk,joint);opt=torch.optim.Adam(p.parameters(),lr=plan['policy_lr']);obs=roots[0].clone().observation();save(out/'initial_samples.json',sample_audit(p,obs,base,trunk))
 history=[];dev={};progress={};updates=0
 def checkpoint(label):
  torch.save(p.state_dict(),out/(label+'.pt'));progress[label]={'training_updates':updates,'groups':len(history),'optimizer_steps':sum(b['optimizer_steps'] for b in history)};dev[label]={mode:evaluate(data,cfg,plan['selection_roots'],p,hold,plan['selection_action_seed'],mode=='stochastic') for mode in ['greedy','stochastic']};save(out/'development.json',dev);save(out/'checkpoint_work.json',progress);print(a.arm,a.seed,label,updates,{m:score(r) for m,r in dev[label].items()},flush=True)
 checkpoint('initial')
 for u in range(plan['max_groups']):
  root=roots[u%len(roots)];torch.manual_seed(a.seed*10000+u);b=train_batch(p,opt,root,hold,plan['group']);b.update({'group':u,'root':root.seed});history.append(b);updates+=sum(r['branch_cost']['updates'] for r in b['results']);save(out/'training.json',history);print(a.arm,a.seed,'group',u,'updates',updates,flush=True)
  if updates>=plan['mid_updates'] and 'mid' not in dev:checkpoint('mid')
  if updates>=plan['training_updates']:break
 assert updates>=plan['training_updates'],'Safety group cap reached; report incomplete, never silently call it matched budget.'
 checkpoint('terminal');save(out/'final_samples.json',sample_audit(p,obs,base,trunk))
 selected={mode:min(dev,key=lambda k:score(dev[k][mode])) for mode in ['greedy','stochastic']};checkpoints=[s for s in ['initial','mid','terminal'] if s in {'initial','terminal',*selected.values()}]
 save(out/'locked_selection.json',{'selected':selected,'audit_checkpoints':checkpoints,'model_sha256':{s:hashlib.sha256((out/(s+'.pt')).read_bytes()).hexdigest() for s in dev}})
 audit={}
 for label in checkpoints:
  p.load_state_dict(torch.load(out/(label+'.pt'),weights_only=True));audit[label]={m:evaluate(data,cfg,plan['audit_roots'],p,hold,plan['audit_action_seed'],m=='stochastic') for m in ['greedy','stochastic']};save(out/'audit.json',audit)
 cost={'wall_seconds':time.perf_counter()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'training_updates':updates,'training_groups':len(history),'training_branches':sum(len(b['results']) for b in history),'training_decisions':sum(b['decisions'] for b in history),'policy_optimizer_steps':sum(b['optimizer_steps'] for b in history),'prefix_updates':sum(e.cost['updates'] for e in roots),'training_cost':{k:sum(r['branch_cost'][k] for b in history for r in b['results']) for k in roots[0].cost},'evaluation_cost':{k:sum(r['cost'][k] for dataset in [dev,audit] for modes in dataset.values() for rows in modes.values() for r in rows) for k in roots[0].cost}};save(out/'cost.json',cost);print('complete',a.arm,a.seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--seed',type=int,required=True);main(p.parse_args())

"""Version-8 protocol, paired action diagnosis, supervised and PPO comparisons."""
import argparse
import copy
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
import control as l
import sequence_env as e
import ppo

SPLITS=dict(train=list(range(810000,810064)),dev=list(range(820000,820024)),test=list(range(830000,830064)))
PILOT=dict(train=list(range(850000,850008)),dev=list(range(860000,860004)),test=list(range(870000,870008)))

def plans():
    result=[dict(name=f'fixed_{a}',kind='fixed',action=a) for a in range(15)]
    result += [dict(name=f'lr_{scale}',kind='lr',scale=scale,action=4) for scale in [.25,.5,.75,1.,1.25,1.5,2.]]
    for early in [0,4,5,9,10,14]:
        for late in [0,4,5,9,10,14]:
            if early==late: continue
            for kind,thresholds in [('time',[.3,.6]),('block',[1,2]),('progress',[.5,.8])]:
                for threshold in thresholds:
                    result.append(dict(name=f'{kind}_{early}_{late}_{threshold}',kind=kind,early=early,late=late,threshold=threshold))
    return result

def plan_chooser(plan):
    def choose(env):
        kind=plan['kind']
        if kind in ['fixed','lr']: return plan['action']
        value=env.local_steps/env.c['segment_steps'] if kind=='time' else env.stage
        if kind=='progress': value=1-env.current_loss/max(env.initial_loss,1e-8)
        return plan['early'] if value<plan['threshold'] else plan['late']
    return choose

def evaluate_plan(item,config,plan,events=False):
    cfg=dict(config,lr=config['lr']*plan.get('scale',1.))
    return e.rollout(item,cfg,plan_chooser(plan),'within',events)

def diagnose(states):
    counts=np.zeros(15,int); improved=0; gains=[]
    for s in states:
        values=[]
        for a in s['branches']:
            hit=np.mean([r['success'] for r in a]); good=[r['forgetting'] for r in a if r['success']]
            values.append((-hit,np.mean(good) if good else float('inf')))
        old=min(range(5),key=lambda a:values[a]); new=min(range(15),key=lambda a:values[a]); counts[new]+=1
        improved+=values[new]<values[old]
        if values[old][0]==values[new][0] and np.isfinite(values[old][1]): gains.append(values[old][1]-values[new][1])
    fixed=[]
    for a in range(15):
        rows=[r for s in states for r in s['branches'][a]]
        fixed.append(dict(action=a,metrics=l.metrics(rows)))
    best=min(fixed,key=lambda r:l.selection_key(r['metrics']))['action']
    ties=0
    for s in states:
        def key(a):
            rows=s['branches'][a]; good=[r['forgetting'] for r in rows if r['success']]
            return (-np.mean([r['success'] for r in rows]),np.mean(good) if good else float('inf'))
        ties+=key(best)==min(key(a) for a in range(15))
    return dict(states=len(states),new_actions_strictly_improve=int(improved),best_action_counts=counts.tolist(),
                mean_equal_reach_forgetting_gain=float(np.mean(gains)) if gains else None,
                fixed_action_values=fixed,best_single_action=best,best_single_action_matches_state_best=int(ties),
                interpretation='Empirical training-state value under common (0,1) continuation; descriptive, not global oracle. Fixed action here is first intervention only; full-trajectory fixed baselines are separately tuned.')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
    if args.out.exists(): parser.error('Use a fresh output directory')
    out=args.out; out.mkdir(parents=True); start=time.time()
    config=copy.deepcopy(e.CONFIG); splits=PILOT if args.pilot else SPLITS; seeds=[881] if args.pilot else [881,882,883]
    source=Path(__file__).parent
    hashes={p.name:l.file_hash(p) for p in source.glob('*.py')}
    protocol=dict(config=config,splits=splits,seeds=seeds,actions=e.ACTIONS,features=e.FEATURES,
        source_hashes=hashes,protocol_text=(source/'README.md').read_text(),p_min=.9,noninferiority_margin=.05,
        supervised_checkpoints=[0,50,100] if args.pilot else [0,150,450],ppo_batches=3 if args.pilot else 24,pilot=args.pilot)
    l.save(out/'plan.json',protocol); l.save(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads()))
    train=[e.make_item(s,config) for s in splits['train']]; dev=[e.make_item(s,config) for s in splits['dev']]
    states=l.collect(out/'data',train,config,'within',1 if args.pilot else 2)
    l.save(out/'action_diagnosis.json',diagnose(states))
    candidates=[]
    for idx,plan in enumerate(plans()):
        rows=[evaluate_plan(item,config,plan) for item in dev]
        candidates.append(dict(plan=plan,metrics=l.metrics(rows),rows=rows))
        if idx%20==0: print(f'baseline {idx}/{len(plans())}',flush=True)
    baselines={}
    for name,kinds in [('fixed',['fixed']),('global_lr',['lr']),('schedule',['fixed','time','block','progress'])]:
        pool=[r for r in candidates if r['plan']['kind'] in kinds]
        baselines[name]=min(pool,key=lambda r:l.selection_key(r['metrics']))
    l.save(out/'baseline_candidates.json',candidates); l.save(out/'baselines.json',baselines)
    l.save(out/'baseline_cost.json',dict(rollouts=len(candidates)*len(dev),updates=sum(r['steps'] for c in candidates for r in c['rows']),work=sum(r['work'] for c in candidates for r in c['rows'])))
    registry={}
    for actions in [5,15]:
        subset=[dict(s,branches=s['branches'][:actions]) for s in states]
        for feedback in [False,True]:
            mode='feedback' if feedback else 'progress'
            for seed in seeds:
                stem=f'a{actions}_{mode}_{seed}'; folder=out/'models'/f'sup_{stem}'
                l.fit(folder,subset,dev,config,'within',feedback,seed,protocol['supervised_checkpoints'],[0.,.1,1.],actions)
                registry[f'sup_{stem}']=dict(path=str((folder/'selected.pt').relative_to(out)),kind='sup')
                registry[f'sup0_{stem}']=dict(path=str((folder/'epoch0.pt').relative_to(out)),kind='sup')
                pfolder=out/'models'/f'ppo_{stem}'
                ppo.fit(pfolder,folder/'selected.pt',subset,train,dev,config,seed,protocol['ppo_batches'],args.pilot)
                for prefix,file in [('ppo','selected'),('warm','warm0'),('random','random0'),('last',f"ppo{protocol['ppo_batches']}")]:
                    registry[f'{prefix}_{stem}']=dict(path=str((pfolder/f'{file}.pt').relative_to(out)),kind='ppo')
    assert hashes=={p.name:l.file_hash(p) for p in source.glob('*.py')}
    policy_hashes={r['path']:l.file_hash(out/r['path']) for r in registry.values()}
    l.save(out/'locked_protocol.json',dict(protocol,registry=registry,baselines=baselines,policy_hashes=policy_hashes,lock_time=time.time()))
    print('LOCKED: creating final test worlds now',flush=True)
    models={name:((l if r['kind']=='sup' else ppo).load(out/r['path']),r['kind']) for name,r in registry.items()}
    result={name:[] for name in models}; result.update({name:[] for name in baselines})
    for i,seed in enumerate(splits['test']):
        item=e.make_item(seed,config)
        for name,((model,ckpt),kind) in models.items():
            result[name].append(e.rollout(item,config,(l if kind=='sup' else ppo).chooser(model,ckpt),'within',True))
        for name,b in baselines.items(): result[name].append(evaluate_plan(item,config,b['plan'],True))
        if (i+1)%4==0: print(f'test {i+1}/{len(splits["test"])}',flush=True)
    l.save(out/'test_online.json',result)
    l.save(out/'total_cost.json',dict(seconds=time.time()-start,test_rollouts=sum(map(len,result.values())),test_updates=sum(r['steps'] for rows in result.values() for r in rows),test_work=sum(r['work'] for rows in result.values() for r in rows),offline_branch_cost_5_equivalent_fraction=1/3,offline_extra_15_action_fraction=2/3))
    assert hashes=={p.name:l.file_hash(p) for p in source.glob('*.py')}
    print(f'Complete {time.time()-start:.1f}s',flush=True)

if __name__=='__main__': main()

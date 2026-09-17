"""Descriptive training/selection diagnostics; never changes policies or protocol."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    selections={}; offline=dict(branch_queries=0,branch_updates=0,branch_work=0.,dev_rollouts=0,dev_updates=0,dev_work=0.,ppo_episodes=0,ppo_updates=0,ppo_work=0.,scorer_optimizer_steps=0,distillation_optimizer_steps=0,ppo_optimizer_steps=0)
    for folder in sorted((run/'models').iterdir()):
        selected=json.loads((folder/'selection.json').read_text()); candidates=json.loads((folder/'candidates.json').read_text())
        curve=json.loads((folder/'training.json').read_text());cost=json.loads((folder/'cost.json').read_text())
        offline['scorer_optimizer_steps']+=max((r.get('epoch',0) for r in curve),default=0)
        offline['distillation_optimizer_steps']+=cost.get('distillation_epochs',0)
        offline['ppo_optimizer_steps']+=cost.get('ppo_gradient_steps',0)
        selections[folder.name]=dict(selected=selected,dev_candidates=len(candidates),
            dev_feasible_candidates=sum(c['metrics']['reach']>=.9 for c in candidates),
            selected_dev_feasible=selected['dev']['reach']>=.9,
            train_curve=curve)
        offline['dev_rollouts']+=cost['dev_rollouts']
        offline['dev_updates']+=cost.get('updates',cost.get('dev_updates',0));offline['dev_work']+=cost.get('work',cost.get('dev_work',0))
        offline['ppo_episodes']+=cost.get('train_episodes',0);offline['ppo_updates']+=cost.get('train_updates',0);offline['ppo_work']+=cost.get('train_work',0.)
    branch=json.loads((run/'data/within_cost.json').read_text());base=json.loads((run/'baseline_cost.json').read_text())
    offline.update(branch_queries=branch['branches'],branch_updates=branch['future_updates'],branch_work=branch['future_work'],
        extra_action_queries=branch['branches']*2//3,old5_equivalent_queries=branch['branches']//3,
        baseline_dev_rollouts=base['rollouts'],baseline_dev_updates=base['updates'],baseline_dev_work=base['work'])
    states=[json.loads(s) for s in (run/'data/within_branches.jsonl').read_text().splitlines()]
    offline['old5_branch_updates']=sum(r['steps']-s['global_steps'] for s in states for rows in s['branches'][:5] for r in rows)
    offline['extra10_branch_updates']=sum(r['steps']-s['global_steps'] for s in states for rows in s['branches'][5:] for r in rows)
    best_fixed=json.loads((run/'action_diagnosis.json').read_text())['best_single_action']
    stages={}
    for stage in [0,1,2]:
        pool=[s for s in states if s['stage']==stage];count=0;win_counts=[0]*15;no_success=0;fixed_explained=0;reach_improved=0
        for s in pool:
            values=[]
            for rows in s['branches']:
                hit=np.mean([r['success'] for r in rows]);good=[r['forgetting'] for r in rows if r['success']]
                values.append((-hit,float(np.mean(good)) if good else float('inf')))
            old=min(range(5),key=lambda a:values[a]);new=min(range(15),key=lambda a:values[a])
            count+=int(values[new]<values[old]);win_counts[new]+=1
            no_success+=int(values[new][0]==0)
            fixed_explained+=int(values[new]<values[old] and values[best_fixed]==values[new])
            reach_improved+=int(values[new][0]<values[old][0])
        stages[stage]=dict(states=len(pool),new_action_improvements=count,best_action_counts=win_counts,all_actions_fail=no_success,best_fixed_matches_improved_state_best=fixed_explained,reach_improvements=reach_improved)
    plan=json.loads((run/'plan.json').read_text())
    offline['shared_pretraining_updates']=sum((1+seed%3)*plan['config']['base_steps'] for seeds in plan['splits'].values() for seed in seeds)
    behavior_path=run/'data/behavior_audit.json'
    if behavior_path.exists():
        behavior=json.loads(behavior_path.read_text())
        offline['collection_behavior_updates']=behavior['updates'];offline['collection_behavior_work']=behavior['work']
        offline['additional_collection_audit_updates']=behavior['updates']+sum((1+s%3)*plan['config']['base_steps'] for s in plan['splits']['train'])
    report=dict(selections=selections,offline_cost=offline,action_diagnosis_by_stage=stages,
                interpretation='Descriptive only; p_min=.9 is development point feasibility, not a certified population probability. Branch labels condition on fixed continuation; best-action counts are not online gains.')
    (run/'analysis').mkdir(exist_ok=True)
    (run/'analysis/training_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(offline,indent=2))
if __name__=='__main__':main()

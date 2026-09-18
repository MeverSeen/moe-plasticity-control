"""Paired roots and controller seeds; all checkpoints and failures remain visible."""
import argparse,collections,json
from pathlib import Path
import numpy as np
from diagnose import save

def read(p):return json.loads(p.read_text())
def metrics(rows):
 return {k:float(np.mean([r[k] for r in rows])) for k in ['success','forgetting','signed_forgetting','backward_improvement','final_loss','final_accuracy']}|{'episodes':len(rows),'updates_mean':float(np.mean([r['cost']['updates'] for r in rows])),'terminal_monitor_loss':float(np.mean([t['loss'] for r in rows for t in r['tasks']])),'initially_successful_tasks':sum(t['initial_success'] for r in rows for t in r['tasks'])}

def comparison(left,right):
 """Arrays indexed controller seed x root; baseline rows can be duplicated as paired controls."""
 assert len(left)==len(right)
 for la,rb in zip(left,right):assert [r['seed'] for r in la]==[r['seed'] for r in rb]
 rng=np.random.default_rng(1180000);result={}
 for metric in ['success','forgetting','final_loss']:
  d=np.array([[float(a[metric])-float(b[metric]) for a,b in zip(la,rb)] for la,rb in zip(left,right)]);ns,nr=d.shape
  ri=rng.integers(nr,size=(10000,nr));si=rng.integers(ns,size=(10000,ns));cross=d[si[:,:,None],ri[:,None,:]].mean((1,2));root=d.mean(0)[ri].mean(1)
  result[metric]={'difference':float(d.mean()),'per_seed':d.mean(1).tolist(),'per_root':d.mean(0).tolist(),'crossed_seed_root_bootstrap95':np.quantile(cross,[.025,.975]).tolist(),'root_bootstrap95_conditioned_on_seeds':np.quantile(root,[.025,.975]).tolist()}
 common=[(a,b) for la,rb in zip(left,right) for a,b in zip(la,rb) if a['success'] and b['success']]
 result['common_success']={'pairs':len(common),'forgetting_difference':float(np.mean([a['forgetting']-b['forgetting'] for a,b in common])) if common else None,'warning':'success-conditioned descriptive comparison; does not establish feasibility noninferiority'}
 return result

def actions(rows):
 counts=collections.Counter();trunk=collections.Counter();alloc=collections.Counter()
 for r in rows:
  first=r['tasks'][0]['steps']//10
  for m in r['actions'][first:]:
   if isinstance(m,dict):trunk[str(m['trunk'])]+=1;m=m['experts']
   else:trunk['1.0']+=1
   counts.update(str(x) for line in m for x in line);alloc[str(m)]+=1
 n=sum(counts.values())
 return {'expert_counts':dict(counts),'trunk_counts':dict(trunk),'allocation_counts':dict(alloc),'base_expert_fraction':counts['0.0']/n if n else None}

def main(a):
 plan=read(a.plan);base=read(a.baselines/'results.json');jobs={};summary={};contrasts={};ledger=collections.Counter();train_diagnostics={}
 for arm in plan['arms']:
  jobs[arm]=[]
  for seed in plan['controller_seeds']:
   path=a.learning/f'{arm}_{seed}';j={n:read(path/(n+'.json')) for n in ['audit','development','training','locked_selection','cost','initial_distribution','final_distribution','prefix_cost']};j['seed']=seed;jobs[arm].append(j)
   for k in ['training_updates','prefix_updates','dev_updates','audit_updates','wall_seconds','training_seconds']:ledger[k]+=j['cost'][k]
   ledger['training_branches']+=sum(len(b['results']) for b in j['training']);ledger['dev_episodes']+=sum(len(rows) for modes in j['development'].values() for rows in modes.values());ledger['audit_episodes']+=sum(len(rows) for modes in j['audit'].values() for rows in modes.values());ledger['prefix_tasks']+=len(j['prefix_cost'])
   train_diagnostics[f'{arm}_{seed}']={'selected':j['locked_selection']['selected'],'initial_distribution':j['initial_distribution'],'final_distribution':j['final_distribution'],'rounds':[{'round':b['round'],'root':b['root'],'reward_std':b.get('reward_std'),'parameter_change':b.get('parameter_change'),'ratio_min':b.get('ratio_min'),'ratio_max':b.get('ratio_max'),'sequence_success':float(np.mean([r['success'] for r in b['results']])),'reward':float(np.mean([r['reward'] for r in b['results']])),'forgetting':float(np.mean([r['forgetting'] for r in b['results']]))} for b in j['training']]}
  summary[arm]={}
  for mode in ['greedy','stochastic']:
   groups={stage:[j['audit'][str(j['locked_selection']['selected'][mode]) if stage=='selected' else str(0 if stage=='initial' else plan['rounds'])][mode] for j in jobs[arm]] for stage in ['initial','selected','terminal']}
   summary[arm][mode]={stage:metrics([r for rows in gg for r in rows])|{'per_seed':[metrics(rows) for rows in gg],'actions':actions([r for rows in gg for r in rows])} for stage,gg in groups.items()}
   summary[arm][mode]['selected_epochs']=[j['locked_selection']['selected'][mode] for j in jobs[arm]]
   summary[arm][mode]['development_curve']={str(epoch):metrics([r for j in jobs[arm] for r in j['development'][str(epoch)][mode]]) for epoch in plan['checkpoints']}
   for stage in ['selected','terminal']:contrasts[f'{arm}/{mode}/{stage}-initial']=comparison(groups[stage],groups['initial'])
   for name,stages in base.items():contrasts[f'{arm}/{mode}/selected-{name}']=comparison(groups['selected'],[stages['audit'] for _ in jobs[arm]])
   summary[arm][mode]['greedy_terminal_matrix_changes_per_seed']=[sum(x['matrix']!=y['matrix'] for x,y in zip(j['audit']['0']['greedy'],j['audit'][str(plan['rounds'])]['greedy'])) for j in jobs[arm]]
 # Mechanism contrasts use the same fixed epoch, avoiding differential checkpoint selection.
 for left,right in [('warm_grpo','scratch_grpo'),('repair_grpo','warm_grpo'),('repair_ppo','repair_grpo'),('repair_balanced','repair_grpo'),('repair_trunk','repair_grpo')]:
  for mode in ['greedy','stochastic']:
   for epoch in ['0',str(plan['rounds'])]:contrasts[f'{left}-{right}/{mode}/epoch{epoch}']=comparison([j['audit'][epoch][mode] for j in jobs[left]],[j['audit'][epoch][mode] for j in jobs[right]])
 ledger['fixed_episodes']=sum(len(rows) for stages in base.values() for rows in stages.values());ledger['fixed_updates']=sum(r['cost']['updates'] for stages in base.values() for rows in stages.values() for r in rows)
 ledger['all_learning_stage_updates']=sum(ledger[k] for k in ['training_updates','prefix_updates','dev_updates','audit_updates','fixed_updates'])
 result={'scope':'Exploratory development replication: same 20 development classes, 12 new order/init roots, 4 controller seeds, one paired action stream per root. 48 cells are not 48 independent histories. CIs descriptive and unadjusted for multiple comparisons. Negative forgetting differences favor left; positive success differences favor left. No final-class or GPU claims.','plan':plan,'baselines':{k:{s:metrics(rows) for s,rows in v.items()} for k,v in base.items()},'summary':summary,'contrasts':contrasts,'training_diagnostics':train_diagnostics,'ledger':dict(ledger)};save(a.out,result)
 print(json.dumps({'baselines':result['baselines'],'ledger':dict(ledger),'selected':{arm:{m:{k:v[k] for k in ['success','forgetting','final_loss']} for m,mm in modes.items() for v in [mm['selected']]} for arm,modes in summary.items()}},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=Path(__file__).with_name('learning_plan.json'));p.add_argument('--learning',type=Path,default=Path('runs/v11-learning'));p.add_argument('--baselines',type=Path,default=Path('runs/v11-baselines'));p.add_argument('--out',type=Path,default=Path('runs/v11-learning/analysis.json'));main(p.parse_args())

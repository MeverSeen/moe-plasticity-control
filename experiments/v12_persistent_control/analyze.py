"""Paired development replication, actual work and duration-weighted actions."""
import collections,gzip,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def metrics(rows):
 keys=['success','forgetting','signed_forgetting','final_loss','final_accuracy'];out={k:float(np.mean([r[k] for r in rows])) for k in keys}
 out.update({'executions':len(rows),'updates':float(np.mean([r['cost']['updates'] for r in rows])),'decisions':float(np.mean([r['cost']['control_decisions'] for r in rows])),'stop_loss':float(np.mean([t['loss'] for r in rows for t in r['tasks']])),'failures_by_task':[sum(not r['tasks'][k]['success'] for r in rows) for k in range(4)]});return out
def compare(left,right):
 rng=np.random.default_rng(1280000);out={}
 for a,b in zip(left,right):assert [r['seed'] for r in a]==[r['seed'] for r in b]
 for key in ['success','forgetting','final_loss','updates','decisions']:
  value=lambda r: r['cost'][{'updates':'updates','decisions':'control_decisions'}[key]] if key in ['updates','decisions'] else r[key]
  d=np.array([[float(value(a))-float(value(b)) for a,b in zip(aa,bb)] for aa,bb in zip(left,right)]);s,n=d.shape;si=rng.integers(s,size=(10000,s));ri=rng.integers(n,size=(10000,n));bs=d[si[:,:,None],ri[:,None,:]].mean((1,2));out[key]={'mean':float(d.mean()),'seed_means':d.mean(1).tolist(),'root_means':d.mean(0).tolist(),'crossed_bootstrap95':np.quantile(bs,[.025,.975]).tolist()}
 common=[(a,b) for aa,bb in zip(left,right) for a,b in zip(aa,bb) if a['success'] and b['success']];out['common_success']={'pairs':len(common),'F_difference':float(np.mean([a['forgetting']-b['forgetting'] for a,b in common])) if common else None};return out
def behavior(rows,base,trunk):
 hist={k:collections.Counter() for k in ['durations','trunk_decisions','trunk_updates','expert_decisions','expert_updates','allocations','changed_experts']};keep=joint=total=controlled=keep_updates=joint_updates=0
 for r in rows:
  for t in r['policy_trace']:
   m=np.array(t['experts']);n=int((m!=base).sum());duration=t['duration'];total+=1;keep+=n==0 and t['trunk']==trunk;joint+=n>2;controlled+=duration;keep_updates+=duration*(n==0 and t['trunk']==trunk);joint_updates+=duration*(n>2)
   hist['durations'][str(duration)]+=1;hist['changed_experts'][str(n)]+=1;hist['allocations'][str(t['experts'])+'/'+str(t['trunk'])]+=1
   for name,w in [('decisions',1),('updates',duration)]:
    hist['trunk_'+name][str(t['trunk'])]+=w
    for v in m.flatten():hist['expert_'+name][str(float(v))]+=w
 return {k:dict(v) for k,v in hist.items()}|{'decisions':total,'all_base_fraction':keep/max(total,1),'more_than_two_experts_fraction':joint/max(total,1),'unique_allocations':len(hist['allocations']),'controlled_updates':controlled,'mean_realized_hold':controlled/max(total,1),'all_base_update_fraction':keep_updates/max(controlled,1),'more_than_two_experts_update_fraction':joint_updates/max(controlled,1)}
def main():
 plan=read(ROOT/'plan.json');folder=Path('runs/v12-learning');base=read(Path('runs/v12-baselines/results.json'));jobs={};summary={};contrasts={};costs={};rows_by_arm={};base_exp=plan['base']['expert'];base_trunk=plan['base']['trunk']
 for arm in plan['arms']:
  jobs[arm]=[];summary[arm]={};rows_by_arm[arm]={};costs[arm]=[]
  for seed in plan['controller_seeds']:
   d=folder/f'{arm}_{seed}';job={name:read(d/(name+'.json')) for name in ['training','audit','development','cost','locked_selection','checkpoint_work','initial_samples','final_samples']};job['seed']=seed;jobs[arm].append(job);costs[arm].append(job['cost'])
  for mode in ['greedy','stochastic']:
   groups={stage:[j['audit'][j['locked_selection']['selected'][mode] if stage=='selected' else stage][mode] for j in jobs[arm]] for stage in ['initial','selected','terminal']};rows_by_arm[arm][mode]=groups
   summary[arm][mode]={stage:metrics([r for rows in gg for r in rows])|{'per_seed':[metrics(rows)|{'behavior':behavior(rows,base_exp,base_trunk)} for rows in gg],'behavior':behavior([r for rows in gg for r in rows],base_exp,base_trunk)} for stage,gg in groups.items()};summary[arm][mode]['selected_checkpoints']=[j['locked_selection']['selected'][mode] for j in jobs[arm]]
   for stage in ['selected','terminal']:
    contrasts[f'{arm}/{mode}/{stage}-initial']=compare(groups[stage],groups['initial'])
    for name in ['fixed_best','fixed_v11','fixed_trunk1',arm+'_progress',arm+'_feedback']:contrasts[f'{arm}/{mode}/{stage}-{name}']=compare(groups[stage],[base[name]['audit'] for _ in groups[stage]])
 for left,right in [('B','A'),('C','B')]:
  for mode in ['greedy','stochastic']:
   for stage in ['initial','selected','terminal']:contrasts[f'{left}-{right}/{mode}/{stage}']=compare(rows_by_arm[left][mode][stage],rows_by_arm[right][mode][stage])
 train={}
 for arm,jj in jobs.items():
  train[arm]=[]
  for j in jj:
   batches=j['training'];rr=[r for b in batches for r in b['results']];train[arm].append({'seed':j['seed'],'metrics':metrics(rr),'behavior':behavior(rr,base_exp,base_trunk),'checkpoint_work':j['checkpoint_work'],'initial_samples':j['initial_samples'],'final_samples':j['final_samples'],'groups':[{'root':b['root'],'updates':sum(r['branch_cost']['updates'] for r in b['results']),'decisions':b['decisions'],'optimizer_steps':b['optimizer_steps'],'reward_std':b['reward_std'],'parameter_change':b['parameter_change'],'ratio_range':b['ratio_range']} for b in batches]})
 result={'scope':'Exploratory reused development classes; four controller seeds, twelve roots, one action stream per root. Crossed bootstrap intervals unadjusted and not confirmation; degenerate success intervals are not noninferiority proof. B-C includes changed structured exploration support/magnitude. Actual learner budgets may overshoot by one complete group.','plan':plan,'summary':summary,'baselines':{k:{stage:metrics(rows) for stage,rows in stages.items()} for k,stages in base.items()},'contrasts':contrasts,'costs':costs,'training':train};save(folder/'analysis.json',result)
 print(json.dumps({'selected':{arm:{m:{k:v['selected'][k] for k in ['success','forgetting','final_loss','updates','decisions']} for m,v in modes.items()} for arm,modes in summary.items()},'costs':{a:[{k:c[k] for k in ['training_updates','training_groups','training_decisions','policy_optimizer_steps']} for c in cc] for a,cc in costs.items()}},indent=2))
if __name__=='__main__':main()

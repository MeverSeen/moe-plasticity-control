"""Root-clustered descriptive analysis; candidate ranking never uses final test."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from audit_v10 import ranking

def clustered_contrast(left,right):
 a={(g['root'],g['stage'],g['replicate']):g for g in left};b={(g['root'],g['stage'],g['replicate']):g for g in right};values={}
 for key in a.keys()&b.keys():
  if a[key]['selected'] is not None and b[key]['selected'] is not None:values.setdefault(key[0],[]).append(a[key]['selected']-b[key]['selected'])
 d=np.array([np.mean(v) for _,v in sorted(values.items())]);rng=np.random.default_rng(11051)
 return {'root_differences':{str(k):float(np.mean(v)) for k,v in sorted(values.items())},'mean':float(d.mean()),'root_bootstrap95':np.quantile(d[rng.integers(len(d),size=(10000,len(d)))].mean(1),[.025,.975]).tolist(),'negative_favors':'left estimator','eligible_roots':len(d)}

def estimate(delta,labels,mode='positive'):
 if mode=='balanced' and len(set(labels))==5:value=np.mean([np.mean(delta[labels==c]) for c in range(5)])
 else:value=np.mean(delta)
 return float(value if mode=='signed' else max(0,value))

def summary(groups):
 valid=[g for g in groups if g['selected'] is not None];roots=sorted({g['root'] for g in valid});d=np.array([np.mean([g['selected']-g['random_expectation'] for g in valid if g['root']==r]) for r in roots]);rng=np.random.default_rng(11050)
 same=sum(g['concordant'] for g in valid);opp=sum(g['discordant'] for g in valid)
 return {'groups':len(groups),'roots':len(roots),'concordance':same/(same+opp) if same+opp else None,'non_tied_pairs':same+opp,'ties':sum(g['ties'] for g in groups),'selected_eval':np.mean([g['selected'] for g in valid]),'random_eval':np.mean([g['random_expectation'] for g in valid]),'selection_regret':np.mean([g['selection_regret'] for g in valid]),'selected_minus_random':float(d.mean()),'root_bootstrap95':np.quantile(d[rng.integers(len(d),size=(10000,len(d)))].mean(1),[.025,.975]).tolist(),'failure_branch_observations':sum(g['failures'] for g in groups),'valid_groups':len(valid)}

def analyze(out):
 raw=json.loads((out/'reward_diagnostic.json').read_text());methods={};coverage=[];tasks=[];selection_failure={}
 for method in ['retained_positive','retained_balanced','retained_signed','full_monitor_privileged','resample_random','resample_stratified']:
  groups=[];reps=20 if method.startswith('resample') else 1;penalty_chosen=[];all_choices=[]
  for group in raw:
   for rep in range(reps):
    rows=[]
    for b in group['branches']:
     task_est=[]
     for task in b['privileged_task_data']:
      if method.startswith('retained'):data=task['retained'];idx=np.arange(len(data['labels']))
      else:
       data=task['monitor'];labels=np.array(data['labels']);rng=np.random.default_rng(group['root']*1000+group['stage']*100+task['task']*30+rep)
       if method=='resample_random':idx=rng.permutation(len(labels))[:25]
       elif method=='resample_stratified':idx=np.concatenate([rng.permutation(np.where(labels==c)[0])[:5] for c in range(5)])
       else:idx=np.arange(len(labels))
      labels=np.array(data['labels'])[idx];delta=(np.array(data['final'])-data['completion'])[idx];mode=method.split('_')[-1] if method.startswith('retained') else 'positive';task_est.append(estimate(delta,labels,mode))
      if method=='retained_positive':
       ed=np.array(task['eval']['final'])-task['eval']['completion'];counts=np.bincount(labels,minlength=5);coverage.append({'root':group['root'],'stage':group['stage'],'branch':b['branch'],'task':task['task'],'counts':counts.tolist(),'missing_classes':int((counts==0).sum())});tasks.append({'root':group['root'],'stage':group['stage'],'branch':b['branch'],'task':task['task'],'proxy_signed':float(delta.mean()),'eval_signed':float(ed.mean()),'proxy_positive':max(0,float(delta.mean())),'eval_positive':max(0,float(ed.mean())),'delta_sd':float(delta.std()),'within_sample_se':float(delta.std()/len(delta)**.5),'full_monitor_signed':float(np.mean(np.array(task['monitor']['final'])-task['monitor']['completion']))})
     rows.append({'success':b['success'],'proxy_forgetting':float(np.mean(task_est)),'forgetting':b['forgetting'],'failures':sum(not t['success'] for t in b['tasks'])})
    g=ranking(rows);g.update({'root':group['root'],'stage':group['stage'],'replicate':rep});groups.append(g)
    scores=[r['proxy_forgetting']/.1+3*r['failures'] for r in rows];choices=[r for r,s in zip(rows,scores) if abs(s-min(scores))<=1e-8]
    penalty_chosen.append(float(np.mean([r['failures'] for r in choices])));all_choices.append({'root':group['root'],'stage':group['stage'],'replicate':rep,'selected_success':float(np.mean([r['success'] for r in choices])),'selected_eval':float(np.mean([r['forgetting'] for r in choices])),'random_success':float(np.mean([r['success'] for r in rows])),'random_eval':float(np.mean([r['forgetting'] for r in rows]))})
  methods[method]={'summary':summary(groups),'groups':groups};selection_failure[method]={'penalized_selected_failed_task_mean':float(np.mean(penalty_chosen)),'selected_success':float(np.mean([r['selected_success'] for r in all_choices])),'selected_eval':float(np.mean([r['selected_eval'] for r in all_choices])),'random_success':float(np.mean([r['random_success'] for r in all_choices])),'random_eval':float(np.mean([r['random_eval'] for r in all_choices])),'rows':all_choices,'scope':'all branches included, same failure penalty as training; exact ties averaged; not success-filtered'}
 # Task-level rankings are descriptive, never treated as independent replicates.
 task_rank={}
 for measure in ['proxy_signed','proxy_positive']:
  gs=[]
  for root,stage,task in sorted({(r['root'],r['stage'],r['task']) for r in tasks}):
   rr=[{'success':True,'proxy_forgetting':r[measure],'forgetting':r['eval_signed' if measure.endswith('signed') else 'eval_positive']} for r in tasks if (r['root'],r['stage'],r['task'])==(root,stage,task)]
   g=ranking(rr);g['root']=root;gs.append(g)
  task_rank[measure]=summary(gs)
 scope=json.loads((out/'scope_diagnostic.json').read_text());sc={}
 for mode in sorted({r['mode'] for r in scope}):
  rr=[r for r in scope if r['mode']==mode];sc[mode]={'n':len(rr),'success':float(np.mean([r['success'] for r in rr])),'mean_steps':np.mean([r['steps'] for r in rr]),'one_old_delta':np.mean([r['one_step']['old_loss']-r['initial']['old_loss'] for r in rr]),'one_new_delta':np.mean([r['one_step']['new_loss']-r['initial']['new_loss'] for r in rr]),'one_topk_drift':np.mean([r['one_step']['topk_drift'] for r in rr]),'end_cumulative_forgetting':np.mean([r['end']['cumulative_forgetting'] for r in rr]),'end_new_loss':np.mean([r['end']['new_loss'] for r in rr]),'end_topk_drift':np.mean([r['end']['topk_drift'] for r in rr])}
 hashes_equal=all(len({r['one_step_moment_hash'] for r in scope if (r['root'],r['stage'])==key})==1 for key in {(r['root'],r['stage']) for r in scope})
 contrasts={left+'_minus_'+right:clustered_contrast(methods[left]['groups'],methods[right]['groups']) for left,right in [('retained_balanced','retained_positive'),('retained_signed','retained_positive'),('resample_stratified','resample_random')]}
 for mode in sc:
  rr=[r for r in scope if r['mode']==mode];sc[mode]['one_step_positive_old_damage_states']=sum(r['one_step']['old_loss']>r['initial']['old_loss']+1e-8 for r in rr);sc[mode]['one_step_old_delta_by_root']={str(root):float(np.mean([r['one_step']['old_loss']-r['initial']['old_loss'] for r in rr if r['root']==root])) for root in sorted({r['root'] for r in rr})}
 gaps=[x for g in methods['retained_positive']['groups'] for x in g['gaps']];sampling={'median_proxy_pair_gap':float(np.median([x[0] for x in gaps])),'median_eval_pair_gap':float(np.median([x[1] for x in gaps])),'median_task_within_sample_se':float(np.median([t['within_sample_se'] for t in tasks])),'note':'Within-sample SE is descriptive only: task SE and sequence pair gaps are not independent or identically scaled.'}
 result={'actual_groups':len(raw),'actual_roots':len({g['root'] for g in raw}),'actual_branches':sum(len(g['branches']) for g in raw),'actual_failure_branches':sum(not r['success'] for g in raw for r in g['branches']),'reward':methods,'paired_estimator_contrasts':contrasts,'sampling_scale':sampling,'coverage':coverage,'task_deltas':tasks,'task_rank':task_rank,'all_branch_penalized_selection':selection_failure,'scope':sc,'single_step_moments_equal_all_modes':hashes_equal,'scope_warning':'multistep recomputes candidates on each path; one-step losses are not an additive long-term causal decomposition','cost':json.loads((out/'cost.json').read_text())};(out/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({'reward':{k:v['summary'] for k,v in methods.items()},'contrasts':contrasts,'scope':sc,'moment_match':hashes_equal},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();analyze(a.out)

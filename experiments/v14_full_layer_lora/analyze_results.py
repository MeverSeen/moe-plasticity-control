"""Analyze only executed fixed-menu v14 endpoints; keep unexecuted methods explicit."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path

METHODS=[('uniform1','fixed','Standard uniform1'),('static050','fixed','Predeclared static0.5'),('rule','rule','Same-interface rule'),('strong_static','fixed','Development-selected strong static'),('ppo','learned','Shared PPO controller')]
def mean(xs):return statistics.mean(xs) if xs else None
def write_csv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def analyze(directory):
 main=[];task_rows=[];general_rows=[];cost_rows=[];raw={};initial=json.loads((directory/'initial_general.json').read_text()) if (directory/'initial_general.json').exists() else None
 for name,kind,label in METHODS:
  file=directory/(name+'.json');r=json.loads(file.read_text()) if file.exists() else None
  base={'method':name,'class':kind,'label':label,'status':'not_executed','roots':0,'complete_histories':0,'successful_histories':None,'failed_tasks':None,'F':None,'signed_F':None,'final_answer_CE':None,'final_mean_task_EM':None,'positive_score_forgetting':None,'actual_updates':None,'path_updates':None,'control_decisions':None,'actual_seconds_with_general':None,'path_seconds_with_general':None}
  if r is None:
   base['reason']='Strong-static development calibration not run' if name=='strong_static' else 'PPO not trained in this platform/engineering allocation' if name=='ppo' else 'No completed endpoint recorded'
   partial=directory/(name+'__incomplete.json')
   if partial.exists():rr=json.loads(partial.read_text());base.update(status='incomplete',roots=1,failed_tasks=rr['n_fail'],actual_updates=rr['cost_actual']['updates'],reason=rr['reason'])
   main.append(base);continue
  raw[name]=r;ev=r['evaluation'];m=ev['matrix'];s=ev['score_matrix'];assert r['history_completed'];assert all(m[k][j] is not None for k in range(3) for j in range(k+1));assert all(s[k][j] is not None for k in range(3) for j in range(k+1))
  F=mean([max(m[-1][j]-m[j][j],0) for j in range(2)]);assert abs(F-ev['forgetting'])<1e-9;assert r['cost_path']['updates']==sum(t['steps'] for t in r['tasks']);assert sum(t['duration'] for t in r['trace'])==sum(t['steps'] for t in r['tasks'][1:]);assert abs(r['reward']['telescope_error'])<1e-8
  base.update(status='completed_development_short',roots=1,complete_histories=1,successful_histories=int(r['history_success']),failed_tasks=r['n_fail'],F=F,signed_F=ev['signed_forgetting'],final_answer_CE=ev['final_loss'],final_mean_task_EM=mean(s[-1]),positive_score_forgetting=mean([max(s[j][j]-s[-1][j],0) for j in range(2)]),actual_updates=r['cost_actual']['updates'],path_updates=r['cost_path']['updates'],control_decisions=r['cost_actual']['control_decisions'],actual_seconds_with_general=r.get('cost_actual_including_final_general',r['cost_actual'])['wall_seconds'],path_seconds_with_general=r.get('cost_path_including_general',r['cost_path'])['wall_seconds'],reason='Single predeclared development root; no cross-platform causal inference')
  main.append(base)
  for k in range(3):
   for j in range(k+1):task_rows.append({'method':name,'after_task_position':k,'evaluated_task_position':j,'task':r['root']['tasks'][j],'answer_CE':m[k][j],'normalized_EM':s[k][j],'completion_CE':m[j][j],'completion_EM':s[j][j],'signed_CE_change_from_completion':m[k][j]-m[j][j],'signed_EM_change_from_completion':s[k][j]-s[j][j],'learning_updates':r['tasks'][j]['steps'],'learning_target':r['tasks'][j]['target'],'learning_terminal_monitor_CE':r['tasks'][j]['monitor_loss'],'learning_success':r['tasks'][j]['success'],'initial_success':r['tasks'][j]['initial_success']})
  for ability,v in r.get('general_ability',{}).items():general_rows.append({'method':name,'ability':ability,**{key:v[key] for key in ['n','initial_accuracy','final_accuracy','accuracy_delta','initial_answer_nll','final_answer_nll','answer_nll_delta','changed_correctness']}})
  for scope,key in [('actual','cost_actual_including_final_general'),('path','cost_path_including_general')]:
   cost=r.get(key,r['cost_actual' if scope=='actual' else 'cost_path']);cost_rows.append({'method':name,'scope':scope,'updates':cost['updates'],'train_input_tokens':cost['train_input_tokens'],'replay_input_tokens':cost['replay_input_tokens'],'monitor_input_tokens':cost['monitor_input_tokens'],'eval_CE_input_tokens':cost['eval_input_tokens'],'generation_input_tokens':cost['generation_input_tokens'],'generation_output_tokens':cost['generation_output_tokens'],'generation_decode_step_tokens':cost['generation_decode_step_tokens'],'general_input_tokens':cost['general_eval_input_tokens'],'copies':cost['copies'],'copy_seconds':cost['copy_seconds'],'inner_train_forward_backward_seconds':cost['train_forward_seconds']+cost['train_backward_seconds'],'inner_replay_forward_backward_seconds':cost['replay_forward_seconds']+cost['replay_backward_seconds'],'optimizer_seconds':cost['optimizer_seconds'],'monitor_forward_seconds':cost['monitor_forward_seconds'],'eval_CE_forward_seconds':cost['eval_forward_seconds'],'generation_seconds':cost['generation_seconds'],'general_seconds':cost['general_eval_seconds'],'online_control_seconds':r['online_control_seconds'],'observation_seconds_including_monitor':r['observation_seconds'],'outer_training_status':'not_executed','wall_seconds':cost['wall_seconds']})
 paired=[]
 if 'uniform1' in raw:
  reference=raw['uniform1']
  for name,r in raw.items():
   if name=='uniform1':continue
   assert r['checkpoint']['prefix_sha256']==reference['checkpoint']['prefix_sha256'];common=bool(r['history_success'] and reference['history_success']);paired.append({'candidate':name,'reference':'uniform1','paired_roots':1,'candidate_success':r['history_success'],'reference_success':reference['history_success'],'common_success_roots':int(common),'F_difference_on_common_success':r['evaluation']['forgetting']-reference['evaluation']['forgetting'] if common else None,'path_update_difference_on_common_success':r['cost_path']['updates']-reference['cost_path']['updates'] if common else None})
 return {'main':main,'tasks':task_rows,'general':general_rows,'costs':cost_rows,'paired':paired,'clone_diagnostic':{'status':'not_executed'},'scope':'One development root and fixed menu; general independent subsets. No trained PPO, no calibrated strong-static reference, no confirmation or independent-future replication.','raw_endpoint_sha256':{name:hashlib.sha256((directory/(name+'.json')).read_bytes()).hexdigest() for name in raw}}

def plot(result,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[r for r in result['main'] if r['status']=='completed_development_short']
 if not rows:return
 plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,3,figsize=(12,3.7));labels=[r['method'] for r in rows];colors=['#386cb0','#7fc97f','#fdc086'][:len(rows)]
 for ax,key,title in zip(axes,['F','final_mean_task_EM','path_updates'],['Cumulative old-task CE forgetting','Final mean generated exact match','Complete-path learner updates']):
  values=[r[key] for r in rows];ax.bar(labels,values,color=colors);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=20);ax.spines[['top','right']].set_visible(False)
  for i,v in enumerate(values):ax.text(i,v,(f'{v:.6f}' if key=='F' else f'{v:.3f}') if key!='path_updates' else str(v),ha='center',va='bottom',fontsize=9)
  ax.margins(y=.2)
 fig.suptitle('v14: one predeclared development history; no PPO or confirmation',fontsize=12);fig.tight_layout();fig.savefig(out/'short_comparison.png',dpi=180);plt.close(fig)
 general=result['general']
 if general:
  abilities=list(dict.fromkeys(r['ability'] for r in general));fig,ax=plt.subplots(figsize=(10,4));width=.23
  for k,name in enumerate(labels):
   values=[next(r['accuracy_delta']*100 for r in general if r['method']==name and r['ability']==a) for a in abilities];ax.bar([i+(k-1)*width for i in range(len(abilities))],values,width,label=name,color=colors[k])
  ax.axhline(0,color='#333333',linewidth=.7);ax.set_xticks(range(len(abilities)),[{'knowledge_mmlu_subset':'MMLU subset (228)','reasoning_bbh_logical5_subset':'BBH logical5 (128)','reading_boolq_subset':'BoolQ subset (128)','commonsense_piqa_subset':'PIQA subset (128)'}[a] for a in abilities]);ax.set_ylabel('Accuracy change (percentage points)');ax.set_title('Independent option-letter evaluation; limited subsets, no score-based selection');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'general_changes.png',dpi=180);plt.close(fig)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);r=analyze(args.results)
 for key in ['main','tasks','general','costs']:write_csv(args.output/(key+'.csv'),r[key])
 (args.output/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');plot(r,args.output);print(json.dumps({'completed':[x['method'] for x in r['main'] if x['complete_histories']],'paired':r['paired']}))
if __name__=='__main__':main()

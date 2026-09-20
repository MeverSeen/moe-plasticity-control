"""Offline three-class report; incomplete planned roots stay visible."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def mean(v):return float(np.mean(v)) if v else None

def analyze(root):
 plan=read(root/'method_protocol.json');fixed=read(root/'fixed_lock.json');methods=['uniform1']+([] if fixed['selected']=='uniform1' else [fixed['selected']])+['rule','ppo_selected'];rows=[];per_task=[];general=[];by={n:{} for n in methods};cost=[]
 for phase in ['train','development','confirmation','target_reference']:
  for path in sorted((root/'results'/phase).glob('*.json')):
   if path.stem.endswith('__general'):continue
   r=read(path)
   if not r.get('history_completed',False):continue
   for key,value in r['cost_actual'].items():cost.append({'phase':phase,'file':path.name,'kind':'actual_history','metric':key,'value':value})
   if phase!='confirmation':continue
   name=r['candidate']['name'];rid=r['root']['id'];by[name][rid]=r
   for k,t in enumerate(r['tasks']):
    per_task.append({'method':name,'root':rid,**t,'final_CE':r['evaluation']['matrix'][-1][k],'completion_CE':r['evaluation']['matrix'][k][k],'final_EM':r['evaluation']['score_matrix'][-1][k],'completion_EM':r['evaluation']['score_matrix'][k][k]})
   for ability,g in r.get('general_ability',{}).items():general.append({'method':name,'root':rid,'ability':ability,**{k:v for k,v in g.items() if isinstance(v,(int,float,str,bool))}})
   for key,value in r.get('general_cost_actual',{}).items():cost.append({'phase':phase,'file':path.name,'kind':'endpoint_general','metric':key,'value':value})
 common=[s['id'] for s in plan['roots']['confirmation'] if all(s['id'] in by[n] and by[n][s['id']]['success'] for n in methods)]
 for name in methods:
  rr=list(by[name].values());paired=[by[name][i] for i in common];rows.append({'method':name,'class':'PPO' if name=='ppo_selected' else 'rule' if name=='rule' else 'fixed','planned_roots':len(plan['roots']['confirmation']),'completed_roots':len(rr),'successful_roots':sum(r['success'] for r in rr),'failed_completed_roots':sum(not r['success'] for r in rr),'uncompleted_roots':len(plan['roots']['confirmation'])-len(rr),'common_success_roots':len(common),'F_common_success':mean([r['evaluation']['forgetting'] for r in paired]),'final_mean_EM_common':mean([mean(r['evaluation']['score_matrix'][-1]) for r in paired]),'path_updates_common':sum(r['cost_path']['updates'] for r in paired),'F_completed':mean([r['evaluation']['forgetting'] for r in rr]),'signed_F_completed':mean([r['evaluation']['signed_forgetting'] for r in rr]),'final_mean_CE':mean([r['evaluation']['final_loss'] for r in rr]),'final_mean_EM':mean([mean(r['evaluation']['score_matrix'][-1]) for r in rr]),'final_mean_F1':mean([mean([r['evaluation']['score_details'][f'2/{j}']['token_f1'] for j in range(3)]) for r in rr]),'path_updates':sum(r['cost_path']['updates'] for r in rr),'actual_branch_updates':sum(r['cost_actual']['updates'] for r in rr),'decisions':sum(len(r['trace']) for r in rr),'actual_history_seconds':sum(r['cost_actual']['wall_seconds'] for r in rr),'logical_path_seconds':sum(r['cost_path']['wall_seconds'] for r in rr),'fraction_decisions_all_one':mean([float(np.all(np.asarray(t['experts'])==1)) for r in rr for t in r['trace']]),'mean_expert_multiplier':mean([float(np.mean(t['experts'])) for r in rr for t in r['trace']]),'endpoint_general_seconds':sum(r.get('general_cost_actual',{}).get('wall_seconds',0) for r in rr)})
 learning=read(root/'learning.json') if (root/'learning.json').exists() else []
 train=[read(p) for p in sorted((root/'results/train').glob('*.json')) if '__incomplete' not in p.stem];train_visits={s['id']:sum(r['root']['id']==s['id'] for r in train) for s in plan['roots']['train']}
 pairs=[]
 for n in methods:
  if n=='uniform1':continue
  for rid in common:pairs.append({'root':rid,'candidate':n,'F_difference_vs_uniform1':by[n][rid]['evaluation']['forgetting']-by['uniform1'][rid]['evaluation']['forgetting'],'updates_difference_vs_uniform1':by[n][rid]['cost_path']['updates']-by['uniform1'][rid]['cost_path']['updates']})
 confirmation_histories=[{'root':spec['id'],'method':name,'status':'completed' if spec['id'] in by[name] else 'budget_missing','F':by[name][spec['id']]['evaluation']['forgetting'] if spec['id'] in by[name] else None,'final_CE':by[name][spec['id']]['evaluation']['final_loss'] if spec['id'] in by[name] else None,'final_EM':mean(by[name][spec['id']]['evaluation']['score_matrix'][-1]) if spec['id'] in by[name] else None,'path_updates':by[name][spec['id']]['cost_path']['updates'] if spec['id'] in by[name] else None} for spec in plan['roots']['confirmation'] for name in methods]
 for phase in ['train','development','confirmation']:
  for path in (root/'prefixes'/phase).glob('*.json'):
   prefix=read(path)
   for kind in ['cost','snapshot_cost']:
    for key,value in prefix[kind].items():cost.append({'phase':phase,'file':path.name,'kind':'prefix_'+kind,'metric':key,'value':value})
 for batch in learning:
  for key,value in [('optimizer_steps',batch['optimizer_steps']),('controller_optimization_seconds',batch['wall_seconds'])]:cost.append({'phase':'train','file':'batch_'+str(batch['batch']),'kind':'PPO_optimization','metric':key,'value':value})
 return {'confirmation_histories':confirmation_histories,'main':rows,'tasks':per_task,'general':general,'costs':cost,'paired':pairs,'training_histories':[{'root':r['root']['id'],'candidate':r['candidate']['name'],'success':r['success'],'monitor_F':r['monitor']['forgetting'],'evaluation_F':r['evaluation']['forgetting'],'terminal_objective':r['reward']['terminal_objective'],'initial_z':r['reward']['initial_z'],'shaped_return':r['reward']['shaped_return'],'decisions':len(r['trace']),'actual_updates':r['cost_actual']['updates'],'actual_seconds':r['cost_actual']['wall_seconds']} for r in train],'training':{'per_root_fresh_histories':train_visits,'histories':len(train),'decisions':sum(len(r['trace']) for r in train),'ppo_batches':len(learning),'optimizer_steps':sum(x['optimizer_steps'] for x in learning),'batch_statistics':learning},'stage_costs':read(root/'stage_costs.json'),'summary':read(root/'summary.json') if (root/'summary.json').exists() else read(root.parent/'recovered_summary.json'),'common_success_ids':common,'interpretation':'One PPO seed, small held-out-task root set; general subsets previously exposed in pilot. No claim of stable superiority or sufficient training from a null result.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);result=analyze(a.archive);(a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 for key in ['main','tasks','general','costs','paired','training_histories','confirmation_histories']:
  rows=result[key]
  if rows:
   fields=list(dict.fromkeys(k for row in rows for k in row));f=(a.output/(key+'.csv')).open('w',newline='');writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows);f.close()
 print(json.dumps({'main':result['main'],'training':{k:v for k,v in result['training'].items() if k!='batch_statistics'}}))
if __name__=='__main__':main()

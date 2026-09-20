"""One predeclared development root, three fixed/rule paths; no outer learning."""
import copy,gc,hashlib,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from lora_native import NativeLearner
from lora import frozen_hash
from environment import Environment,Config,BudgetStop,cost_difference
from structured_policy import Rule
from evaluation import GeneralEvaluator,compare_general

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_adapter(root,name,learner):
 path=root/'checkpoints'/(name+'.pt');path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');start=time.perf_counter();torch.save({'A_B':[p.detach().cpu() for p in learner.parameters],'base_revision':'6d84c48581ece794365f2b8e9cfb043c68ade9c5','protocol_sha256':sha(root/'protocol.json'),'learner_updates_operational':learner.cost['updates'],'purpose':'Adapter-only deployment checkpoint, factors in deterministic layer/expert/gate-up-down/A-B order; not optimizer resume state'},tmp);tmp.replace(path);return {'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'save_seconds':time.perf_counter()-start}

def main():
 root=Path(os.environ['V14_WORK_ROOT']);allocation=Path(os.environ['V13_ALLOCATION_ROOT']);out=root/'results/short_comparison';out.mkdir(parents=True,exist_ok=False);plan=json.loads((root/'comparison_plan.json').read_text());started=time.time();os.environ['V13_STAGE_DEADLINE_EPOCH']=str(min(started+plan['wall_cap_seconds'],float(os.environ['V13_DEADLINE_EPOCH'])-2700));cfg=Config(**plan['environment']);learner=NativeLearner(str(allocation/'model'));data=torch.load(allocation/'dataset.pt',weights_only=False);answers=torch.load(root/'answer_lookup.pt',weights_only=False);general=GeneralEvaluator(json.loads((root/'general_examples.json').read_text()),json.loads((root/'general_manifest.json').read_text()));spec=copy.deepcopy(plan['root']);emergency=root/'comparison_emergency.pt';full=torch.ones(16,64);initial_general=None;rows=[]
 def progress(stage,**kw):save(root/'comparison_progress.json',{'stage':stage,'epoch':time.time(),'cost':learner.cost,**kw});print(stage,json.dumps(kw),flush=True)
 try:
  progress('initial_general');initial_general=general.run(learner);save(out/'initial_general.json',initial_general);initial_adapter=save_adapter(root,'initial',learner);save(out/'initial_adapter.json',initial_adapter)
  reference=Environment(learner,data,spec,cfg,reference=True).run(Rule([4]*16,mode='fixed'),checkpoint_path=emergency);reference.update(phase='target_reference',candidate={'name':'reference30','mode':'fixed','base_indices':[4]*16});save(out/'target_reference.json',reference)
  spec['targets']=[float(np.median([x['loss'] for x in curve if x['step'] in [20,25,30]])) for curve in reference['loss_curves']];save(out/'targets.json',{'root':spec,'definition':plan['target_rule'],'plan_sha256':sha(root/'comparison_plan.json')});progress('targets_locked',targets=spec['targets'])
  prefix,info=Environment(learner,data,spec,cfg,answer_lookup=answers).prepare_prefix(emergency);save(out/'prefix.json',info)
  for name,base,mode in [('uniform1',4,'fixed'),('static050',2,'fixed'),('rule',4,'rule')]:
   policy=Rule([base]*16,mode=mode);e=Environment.from_state(learner,data,cfg,prefix['environment'],prefix['learner'],answer_lookup=answers);progress('history_running',candidate=name)
   try:r=e.run(policy,checkpoint_path=emergency)
   except BudgetStop:
    incomplete={'protocol_revision':'v14_full_layer_expert_lora_v1','phase':'development_short','root':spec,'candidate':{'name':name,'base_indices':[base]*16,'mode':mode,'execution':'greedy'},'history_completed':False,'history_success':False,'n_fail':sum(x['success'] is False for x in e.records),'tasks':e.records,'unexecuted_tasks':spec['tasks'][e.k:],'trace':e.trace,'target_spec':spec['targets'],'reason':'wall-clock reserve interrupted history; unfinished not fabricated as task failure','cost_actual':cost_difference(learner.cost,e.cost_start)};save(out/(name+'__incomplete.json'),incomplete);raise
   r.update(phase='development_short',candidate={'name':name,'base_indices':[base]*16,'mode':mode,'execution':'greedy'},comparison_plan_sha256=sha(root/'comparison_plan.json'),target_spec={'targets':spec['targets'],'rule':plan['target_rule']});r['checkpoint']['adapter']=save_adapter(root,name,learner);save(out/(name+'.json'),r);progress('history_complete',candidate=name,success=r['success'],path_updates=r['cost_path']['updates'],F=r['evaluation']['forgetting'])
   final_general=general.run(learner);save(out/(name+'__general.json'),final_general);r['general_ability']=compare_general(initial_general,final_general);r['general_evaluation_cost']={'initial_shared':initial_general['cost'],'final_actual':final_general['cost']};r['cost_actual_including_final_general']=r['cost_actual'].copy();r['cost_path_including_general']=r['cost_path'].copy()
   for key,source in [('general_eval_input_tokens','input_tokens'),('general_eval_padded_tokens','padded_tokens'),('general_eval_target_tokens','target_tokens_scored'),('general_eval_seconds','wall_seconds')]:r['cost_actual_including_final_general'][key]+=final_general['cost'][source];r['cost_path_including_general'][key]+=initial_general['cost'][source]+final_general['cost'][source]
   r['cost_actual_including_final_general']['wall_seconds']+=final_general['cost']['wall_seconds'];r['cost_path_including_general']['wall_seconds']+=initial_general['cost']['wall_seconds']+final_general['cost']['wall_seconds'];save(out/(name+'.json'),r);rows.append({'candidate':name,'success':r['success'],'updates_path':r['cost_path']['updates'],'F':r['evaluation']['forgetting'],'final_answer_loss':r['evaluation']['final_loss'],'final_task_EM':r['evaluation']['score_matrix'][-1],'general_ability':r['general_ability']});gc.collect()
  learner.check_frozen();base=frozen_hash(learner.model);smoke=json.loads((root/'smoke_result.json').read_text());assert base==smoke['base_hash'];progress('complete');save(out/'summary.json',{'status':'completed','phase':'development_short','profiles':rows,'native_cost':learner.cost,'wall_seconds':time.time()-started,'initial_general_shared':True,'frozen_base_hash':base,'scope':'One development root; static050 was predeclared, not a calibrated strong-static winner. Rule is same-information/interface. No learned controller or clone executed. No cross-platform causal comparison.'})
 except BudgetStop:progress('budget_stop');save(out/'unfinished_summary.json',{'status':'incomplete','completed_candidates':[x['candidate'] for x in rows],'native_cost':learner.cost,'wall_seconds':time.time()-started});raise
 finally:save(root/'comparison_cost.json',{'native_cost':learner.cost,'wall_seconds':time.time()-started,'finished_epoch':time.time(),'stage_cap_seconds':plan['wall_cap_seconds']})
if __name__=='__main__':
 try:main()
 except BudgetStop as e:print(str(e),flush=True);raise SystemExit(124)

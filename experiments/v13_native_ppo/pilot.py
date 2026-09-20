import copy,hashlib,json,os,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from native import NativeLearner
from environment import Environment,Config,BudgetStop
from sparse_policy import Rule

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def main():
 root=Path(os.environ['V13_ALLOCATION_ROOT']);out=root/'results/pilot';out.mkdir(parents=True,exist_ok=False);start=time.time();manifest=json.loads((root/'data_manifest.json').read_text());data=torch.load(root/'dataset.pt',weights_only=False);cfg=Config();learner=NativeLearner(str(root/'model'));spec={'id':'dev_1301000','seed':1301000,'split':'development','tasks':manifest['splits']['development'][:3],'targets':None};fixed=Rule([4,4],mode='fixed')
 reference=Environment(learner,data,spec,cfg,reference=True).run(fixed,checkpoint_path=root/'emergency.pt');reference.update({'phase':'pilot','candidate':{'name':'reference40','base_indices':[4,4],'execution':'greedy'}});save(out/'dev_1301000__reference40.json',reference)
 targets=[float(np.median([point['loss'] for point in curve if point['step'] in [30,35,40]])) for curve in reference['loss_curves']];spec['targets']=targets;save(out/'dev_1301000__targets.json',{'root':spec,'rule':'median fixed-reference monitor CE at learner updates30,35,40; no policy-dependent initial-loss target','reference_improvements':[curve[0]['loss']-target for curve,target in zip(reference['loss_curves'],targets)]});print('TARGETS',targets,flush=True)
 env=Environment(learner,data,spec,cfg);prefix,info=env.prepare_prefix(root/'emergency.pt');save(out/'dev_1301000__prefix.json',info)
 profiles=[]
 for name,policy in [('fixed1',fixed),('fixed075',Rule([3,3],mode='fixed')),('provisional_rule1',Rule([4,4]))]:
  e=Environment.from_state(learner,data,cfg,prefix['environment'],prefix['learner']);r=e.run(policy,action_seed=1301000,teacher=Rule([4,4]),checkpoint_path=root/'emergency.pt');r.update({'phase':'pilot','candidate':{'name':name,'base_indices':policy.base,'execution':'greedy'}});save(out/(spec['id']+'__'+name+'.json'),r);profiles.append({'candidate':name,'success':r['success'],'updates':r['cost_path']['updates'],'decisions':len(r['trace']),'steps':[t['steps'] for t in r['tasks']],'F':r['evaluation']['forgetting'],'monitor_F':r['monitor']['forgetting'],'seconds':r['cost_actual']['wall_seconds']});print('PILOT',profiles[-1],flush=True)
 save(out/'summary.json',{'config':asdict(cfg),'profiles':profiles,'native_cost':learner.cost,'wall_seconds':time.time()-start,'reference_updates':reference['cost_actual']['updates'],'deadline_utc':'2026-09-20T13:25:00Z','scope':'Engineering/development only; provisional rule and H=5 are not formal PPO or confirmation.'})
if __name__=='__main__':
 try:main()
 except BudgetStop as e:print(str(e),flush=True);raise SystemExit(124)

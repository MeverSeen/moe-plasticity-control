"""Bounded sequential v13 stages; artifacts are append-only per executed history."""
import argparse,copy,gc,hashlib,json,os,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from environment import Environment,Config,BudgetStop,training_trajectory
from native import NativeLearner
from sparse_policy import ActorCritic,Rule
from mechanisms import ppo_update
from pilot import save

FIXED=[('fixed025',[1,1]),('fixed050',[2,2]),('fixed075',[3,3]),('fixed100',[4,4]),('fixed050_100',[2,4]),('fixed100_050',[4,2])]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checkpoint(path,policy,extra=None):
 payload={'policy':policy.state_dict(),'epsilon':policy.epsilon,'extra':extra or {}};tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path);return digest(path)
def load_policy(path):
 s=torch.load(path,weights_only=False);p=ActorCritic(epsilon=s['epsilon']);p.load_state_dict(s['policy']);return p

def roots(manifest):
 out={}
 for split,n,seed in [('train',8,1310000),('development',2,1301000),('confirmation',2,1320000)]:
  names=manifest['splits'][split];out[split]=[{'id':f'{split}_{seed+i}','seed':seed+i,'split':split,'tasks':names[i*3:i*3+3],'targets':None} for i in range(n)]
 return out

class Workflow:
 def __init__(self,root):
  self.root=root;self.started=time.time();self.manifest=json.loads((root/'data_manifest.json').read_text());self.plan=json.loads((root/'execution_plan.json').read_text());self.cfg=Config(**self.plan['environment']);self.roots=roots(self.manifest);self.data=torch.load(root/'dataset.pt',weights_only=False);self.learner=NativeLearner(str(root/'model'));self.emergency=root/'emergency.pt';self.state={}
 def progress(self,stage,**kw):save(self.root/'workflow_progress.json',{'stage':stage,'utc_epoch':time.time(),'native_cost':self.learner.cost,**kw})
 def target(self,spec):
  spec=copy.deepcopy(spec);path=self.root/'targets'/(spec['id']+'.json')
  if path.exists():return json.loads(path.read_text())['root']
  ref=Environment(self.learner,self.data,spec,self.cfg,reference=True).run(Rule([4,4],mode='fixed'),checkpoint_path=self.emergency)
  ref.update(phase='target_reference',candidate={'name':'reference40','base_indices':[4,4],'execution':'greedy'});save(self.root/'results/target_reference'/(spec['id']+'.json'),ref)
  spec['targets']=[float(np.median([x['loss'] for x in curve if x['step'] in [30,35,40]])) for curve in ref['loss_curves']]
  save(path,{'root':spec,'definition':self.plan['target_rule'],'initial_improvement':[curve[0]['loss']-v for curve,v in zip(ref['loss_curves'],spec['targets'])]});self.progress('targets',root_id=spec['id']);return spec
 def prefix(self,spec,phase):
  e=Environment(self.learner,self.data,spec,self.cfg);prefix,info=e.prepare_prefix(self.emergency);save(self.root/'prefixes'/phase/(spec['id']+'.json'),info);return prefix
 def run(self,spec,policy,name,phase,prefix=None,stochastic=False,controller_path=None,reference=None,teacher=None,seed=None):
  path=self.root/'results'/phase/(spec['id']+'__'+name+('__stochastic' if stochastic else '__greedy')+'.json');assert not path.exists(),path
  e=Environment(self.learner,self.data,spec,self.cfg) if prefix is None else Environment.from_state(self.learner,self.data,self.cfg,prefix['environment'],prefix['learner'])
  r=e.run(policy,stochastic,spec['seed']+17 if seed is None else seed,teacher,reference,self.emergency,self.state)
  r['phase']=phase;r['candidate']={'name':name,'execution':'stochastic' if stochastic else 'greedy','epsilon':policy.epsilon}
  if isinstance(policy,Rule):r['candidate'].update(base_indices=policy.base,mode=policy.mode)
  if controller_path is not None:r['checkpoint']['controller_sha256']=digest(controller_path)
  r['target_spec']={'targets':spec['targets'],'rule':self.plan['target_rule']};r['execution_plan_sha256']=digest(self.root/'execution_plan.json');save(path,r);self.progress(phase,root_id=spec['id'],candidate=name,success=r['success'],updates=r['cost_path']['updates'],F=r['evaluation']['forgetting']);print('HISTORY',phase,spec['id'],name,r['success'],r['cost_path']['updates'],r['evaluation']['forgetting'],flush=True);return r
 def calibration(self):
  rows={name:[] for name,_ in FIXED}
  for raw in self.roots['development']:
   spec=self.target(raw);prefix=self.prefix(spec,'calibration')
   for name,base in FIXED:rows[name].append(self.run(spec,Rule(base,mode='fixed'),name,'calibration',prefix))
   del prefix;gc.collect()
  def score(rr):return [-sum(x['success'] for x in rr),sum(x['failure_count'] for x in rr),float(np.mean([x['evaluation']['forgetting'] for x in rr])),float(np.mean([x['evaluation']['final_loss'] for x in rr])),sum(x['cost_path']['updates'] for x in rr)]
  name,base=min(FIXED,key=lambda pair:score(rows[pair[0]]));lock={'name':name,'base_indices':base,'selection_rule':self.plan['fixed_selection'],'scores':{k:score(v) for k,v in rows.items()},'selected_at_epoch':time.time(),'feasible':all(x['success'] for x in rows[name])};save(self.root/'fixed_lock.json',lock)
  teacher=Rule(base);rr=[]
  for raw in self.roots['development']:
   spec=self.target(raw);prefix=self.prefix(spec,'rule_validation');rr.append(self.run(spec,teacher,'rule','rule_validation',prefix));del prefix;gc.collect()
  gate={'fixed_feasible':lock['feasible'],'rule_feasible':all(x['success'] for x in rr),'ppo_allowed_after_clone':lock['feasible'] and all(x['success'] for x in rr)};save(self.root/'rule_gate.json',gate);self.progress('calibration_complete',gate=gate)
 def fit_clone(self,policy,observations,teacher,epochs,seed):
  torch.set_num_threads(1);torch.manual_seed(seed);optimizer=torch.optim.Adam(policy.parameters(),lr=.002);pairs=[]
  for obs in observations:
   with torch.no_grad():latent=teacher.act(obs,deterministic=True)[1]
   pairs.append((obs,latent))
  assert pairs;history=[];rng=np.random.default_rng(seed)
  for epoch in range(epochs):
   ix=rng.choice(len(pairs),size=min(16,len(pairs)),replace=False);loss=torch.stack([-policy(*pairs[int(i)])[2] for i in ix]).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
   if epoch%50==0:history.append({'epoch':epoch,'joint_nll':float(loss.detach())})
  with torch.no_grad():matches=sum(torch.equal(policy.act(o,deterministic=True)[0],teacher.act(o,deterministic=True)[0]) for o in observations)
  torch.set_num_threads(8);return {'observations':len(observations),'epochs':epochs,'loss_history':history,'greedy_action_match':matches/len(observations),'selection':'diagnostic only; acceptance uses independent closed-loop development runs'}
 def clone(self):
  gate=json.loads((self.root/'rule_gate.json').read_text())
  if not gate['ppo_allowed_after_clone']:save(self.root/'clone_gate.json',{'accepted':False,'reason':'Fixed or rule failed the predeclared development feasibility gate; expensive PPO not executed.'});return
  fixed=json.loads((self.root/'fixed_lock.json').read_text());teacher=Rule(fixed['base_indices']);obs=[];torch.manual_seed(1301);policy=ActorCritic();trains=[self.target(s) for s in self.roots['train'][:2]]
  for spec in trains:
   r=self.run(spec,teacher,'rule','clone_teacher');obs.extend(tuple(torch.tensor(v) for v in t['observation']) for t in r['trace'])
  reports=[]
  for attempt in range(3):
   fit=self.fit_clone(policy,obs,teacher,400,1301+attempt);cp=self.root/'controllers'/f'clone_attempt{attempt}.pt';cp.parent.mkdir(exist_ok=True);checkpoint(cp,policy,{'fit':fit,'attempt':attempt});checks=[]
   for raw in self.roots['development']:
    spec=self.target(raw);prefix=self.prefix(spec,f'clone_validation_{attempt}');rule=self.run(spec,teacher,'rule',f'clone_validation_{attempt}',prefix);r=self.run(spec,policy,'clone',f'clone_validation_{attempt}',prefix,controller_path=cp,teacher=teacher);checks.append({'root':spec['id'],'success':r['success'],'rule_success':rule['success'],'F_difference':r['evaluation']['forgetting']-rule['evaluation']['forgetting'],'loss_difference':r['evaluation']['final_loss']-rule['evaluation']['final_loss'],'path_updates_ratio':r['cost_path']['updates']/max(rule['cost_path']['updates'],1)});del prefix;gc.collect()
   accepted=all(x['success'] and x['rule_success'] and x['F_difference']<=.02 and x['loss_difference']<=.05 and x['path_updates_ratio']<=1.25 for x in checks);reports.append({'attempt':attempt,'fit':fit,'closed_loop_checks':checks,'accepted':accepted,'controller':cp.name,'sha256':digest(cp)});save(self.root/'clone_gate.json',{'accepted':accepted,'attempts':reports,'criteria':self.plan['clone_gate']})
   if accepted:
    final=self.root/'controllers/clone.pt';checkpoint(final,policy,{'accepted_attempt':attempt});self.progress('clone_accepted',attempt=attempt);return
   if attempt<2:
    for spec in trains:
     r=self.run(spec,policy,'clone',f'clone_repair_{attempt}',controller_path=cp,teacher=teacher);obs.extend(tuple(torch.tensor(v) for v in t['observation']) for t in r['trace'])
  self.progress('clone_rejected')
 def train(self):
  if not json.loads((self.root/'clone_gate.json').read_text())['accepted']:save(self.root/'ppo_status.json',{'executed':False,'reason':'Clone feasibility gate did not pass'});return
  specs=[self.target(s) for s in self.roots['train']];policy=load_policy(self.root/'controllers/clone.pt');reference=copy.deepcopy(policy);optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4);torch.manual_seed(1301);started=time.time();history=[];visits={s['id']:0 for s in specs};limit=self.plan['ppo']['maximum_histories'];buffer=[];current=self.root/'controllers/ppo_000.pt';checkpoint(current,policy,{'optimizer':optimizer.state_dict(),'completed_histories':0});stop_reason='history_cap'
  for i in range(limit):
   if time.time()-started>=self.plan['ppo']['wall_cap_seconds'] or time.time()>float(os.environ['V13_DEADLINE_EPOCH'])-self.plan['confirmation_archive_reserve_seconds']:stop_reason='wall_or_reserve_cap';break
   spec=specs[i%len(specs)];visits[spec['id']]+=1;self.state={'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'buffer':buffer,'completed_histories':i,'visits':visits,'training_started_epoch':started};r=self.run(spec,policy,'ppo',f'train_{i:03}',stochastic=True,controller_path=current,reference=reference,seed=13010000+i);buffer.append(training_trajectory(r))
   if len(buffer)==2:
    stats=ppo_update(policy,optimizer,buffer);history.append({'histories':i+1,'statistics':stats});buffer=[];current=self.root/'controllers'/f'ppo_{i+1:03}.pt';checkpoint(current,policy,{'optimizer':optimizer.state_dict(),'completed_histories':i+1,'visits':visits});save(self.root/'ppo_learning.json',history)
  save(self.root/'ppo_status.json',{'executed':True,'completed_histories':sum(visits.values()),'visits':visits,'updates':len(history),'unused_rollouts_in_final_buffer':len(buffer),'wall_seconds':time.time()-started,'stop_reason':stop_reason,'checkpoints':[p.name for p in sorted((self.root/'controllers').glob('ppo_*.pt'))]});self.state={};self.progress('ppo_complete')
 def evaluate(self):
  status=json.loads((self.root/'ppo_status.json').read_text()) if (self.root/'ppo_status.json').exists() else {'executed':False}
  if not status['executed']:return
  candidates=status['checkpoints'];indices=sorted(set([0,len(candidates)//2,len(candidates)-1]));menu=[]
  for index in indices:
   cp=self.root/'controllers'/candidates[index]
   for stochastic in [False,True]:menu.append((cp,stochastic))
  rows={}
  for raw in self.roots['development']:
   spec=self.target(raw);prefix=self.prefix(spec,'development_selection')
   for cp,stochastic in menu:
    key=cp.stem+('_stochastic' if stochastic else '_greedy');r=self.run(spec,load_policy(cp),cp.stem,'development_selection',prefix,stochastic,cp);rows.setdefault(key,[]).append(r)
   del prefix;gc.collect()
  def score(rr):return [-sum(x['success'] for x in rr),sum(x['failure_count'] for x in rr),float(np.mean([x['evaluation']['forgetting'] for x in rr])),float(np.mean([x['evaluation']['final_loss'] for x in rr])),sum(x['cost_path']['updates'] for x in rr)]
  cp,stochastic=min(menu,key=lambda v:score(rows[v[0].stem+('_stochastic' if v[1] else '_greedy')]));lock={'controller':cp.name,'sha256':digest(cp),'stochastic':stochastic,'selection_rule':self.plan['checkpoint_selection'],'scores':{k:score(v) for k,v in rows.items()},'locked_at_epoch':time.time()};save(self.root/'confirmation_lock.json',lock)
  fixed=json.loads((self.root/'fixed_lock.json').read_text());clonepath=self.root/'controllers/clone.pt'
  for raw in self.roots['confirmation']:
   spec=self.target(raw);prefix=self.prefix(spec,'confirmation')
   self.run(spec,Rule(fixed['base_indices'],mode='fixed'),fixed['name'],'confirmation',prefix)
   self.run(spec,Rule(fixed['base_indices']),'rule','confirmation',prefix)
   self.run(spec,load_policy(clonepath),'clone','confirmation',prefix,False,clonepath)
   self.run(spec,load_policy(cp),'ppo_selected','confirmation',prefix,stochastic,cp)
   del prefix;gc.collect()
  self.progress('confirmation_complete')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calibration','clone','train','evaluate']);args=parser.parse_args();root=Path(os.environ['V13_ALLOCATION_ROOT']);w=Workflow(root)
 try:getattr(w,args.stage)()
 finally:save(root/('stage_'+args.stage+'_cost.json'),{'native_cost':w.learner.cost,'wall_seconds':time.time()-w.started,'ended_epoch':time.time()})
if __name__=='__main__':
 try:main()
 except BudgetStop as e:print(str(e),flush=True);raise SystemExit(124)

"""Time-bounded single-seed PPO on the frozen v14 full-layer LoRA platform."""
import copy,gc,hashlib,json,os,time,math
from pathlib import Path
import numpy as np
import torch
from environment import Environment,Config,BudgetStop,training_trajectory,cost_difference
from lora_native import NativeLearner
from lora import frozen_hash
from structured_policy import ActorCritic,Rule
from ppo import ppo_update
from evaluation import GeneralEvaluator,compare_general

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def initialize(seed):
 torch.manual_seed(seed);p=ActorCritic()
 with torch.no_grad():
  p.base_head.weight.zero_();p.base_head.bias.fill_(math.log(.025));p.base_head.bias[4]=math.log(.9)
  p.count_head.weight.zero_();p.count_head.bias.copy_(torch.tensor([math.log(.9),math.log(.05),math.log(.05)]))
 return p

def rank(r):return [r['failure_count'],r['evaluation']['forgetting'],r['evaluation']['final_loss'],r['cost_path']['updates']]

class Workflow:
 def __init__(self):
  self.root=Path(os.environ['V14_METHOD_ROOT']);self.alloc=Path(os.environ['V13_ALLOCATION_ROOT']);self.pilot=self.alloc/'v14';self.plan=json.loads((self.root/'method_protocol.json').read_text());self.started=time.time();self.cfg=Config(**self.plan['environment']);self.data=torch.load(self.alloc/'dataset.pt',weights_only=False);self.answers=torch.load(self.pilot/'answer_lookup.pt',weights_only=False);self.learner=NativeLearner(str(self.alloc/'model'));self.general=GeneralEvaluator(json.loads((self.pilot/'general_examples.json').read_text()),json.loads((self.pilot/'general_manifest.json').read_text()));self.policy=initialize(self.plan['ppo']['seed']);self.optim=torch.optim.Adam(self.policy.parameters(),lr=self.plan['ppo']['lr']);self.learning=[];self.visits={s['id']:0 for s in self.plan['roots']['train']};self.costs={};self.rows=[];self.state={};self.emergency=self.root/'emergency.pt';self.final_deadline=float(os.environ['V13_DEADLINE_EPOCH'])-2700;self.initial_general=json.loads((self.pilot/'results/short_comparison/initial_general.json').read_text());self.initial_hash=sha(self.pilot/'checkpoints/initial.pt');self.controller('ppo_initial')
 def progress(self,stage,**kw):
  value={'stage':stage,'epoch':time.time(),'native_cost':self.learner.cost,'visits':self.visits,**kw};save(self.root/'progress.json',value);print(json.dumps(value),flush=True)
 def controller(self,name):
  p=self.root/'controllers'/(name+'.pt');p.parent.mkdir(exist_ok=True);torch.save({'policy':self.policy.state_dict(),'optimizer':self.optim.state_dict(),'visits':self.visits,'protocol_sha256':sha(self.root/'method_protocol.json')},p);return {'path':p.name,'sha256':sha(p),'parameters':sum(p.numel() for p in self.policy.parameters())}
 def adapter(self,name):
  p=self.root/'adapters'/(name+'.pt');p.parent.mkdir(exist_ok=True);started=time.perf_counter();torch.save({'A_B':[p.detach().cpu() for p in self.learner.parameters],'base_revision':self.plan['base_revision'],'protocol_sha256':sha(self.root/'protocol.json'),'method_protocol_sha256':sha(self.root/'method_protocol.json'),'purpose':'deployment adapter only; no Adam state'},p);return {'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'save_hash_seconds':time.perf_counter()-started}
 def phase(self,name,seconds,fn):
  start=time.time();before=copy.deepcopy(self.learner.cost);os.environ['V13_STAGE_DEADLINE_EPOCH']=str(min(start+seconds,self.final_deadline));self.progress(name+'_start',stage_cap_seconds=seconds);status='completed'
  try:fn()
  except BudgetStop as exc:status='budget_stop';self.progress(name+'_budget_stop',reason=str(exc))
  finally:
   self.costs[name]={'status':status,'wall_seconds':time.time()-start,'native_cost':cost_difference(self.learner.cost,before),'ended_epoch':time.time()};save(self.root/'stage_costs.json',self.costs)
 def target(self,raw):
  spec=copy.deepcopy(raw)
  if spec['split']=='development':
   saved=json.loads((self.pilot/'results/short_comparison/targets.json').read_text());assert saved['root']['seed']==spec['seed'] and saved['root']['tasks']==spec['tasks'];spec['targets']=saved['root']['targets'];save(self.root/'targets'/(spec['id']+'.json'),{'root':spec,'reused_from_pilot':True,'source_sha256':sha(self.pilot/'results/short_comparison/targets.json')});return spec
  e=Environment(self.learner,self.data,spec,self.cfg,reference=True);r=e.run(Rule([4]*16,mode='fixed'),checkpoint_path=self.emergency);r['phase']='target_reference';save(self.root/'results/target_reference'/(spec['id']+'.json'),r);spec['targets']=[float(np.median([x['loss'] for x in curve if x['step'] in [20,25,30]])) for curve in r['loss_curves']];save(self.root/'targets'/(spec['id']+'.json'),{'root':spec,'target_rule':self.plan['target_rule']});self.progress('targets_ready',root_id=spec['id'],targets=spec['targets']);return spec
 def prefix(self,spec,phase,scores):
  e=Environment(self.learner,self.data,spec,self.cfg,answer_lookup=self.answers if scores else None);prefix,info=e.prepare_prefix(self.emergency);save(self.root/'prefixes'/phase/(spec['id']+'.json'),info);return prefix
 def run(self,spec,policy,name,phase,prefix,stochastic=False,index=0,general=False):
  path=self.root/'results'/phase/(spec['id']+'__'+name+'.json');assert not path.exists();e=Environment.from_state(self.learner,self.data,self.cfg,prefix['environment'],prefix['learner'],answer_lookup=None if phase=='train' else self.answers);self.progress('history_start',phase=phase,root_id=spec['id'],candidate=name)
  try:r=e.run(policy,stochastic,self.plan['ppo']['action_seed']+index,checkpoint_path=self.emergency,extra=self.state)
  except BudgetStop:
   save(path.with_name(path.stem+'__incomplete.json'),{'history_completed':False,'root':spec,'phase':phase,'candidate':name,'tasks':e.records,'trace':e.trace,'cost_actual':cost_difference(self.learner.cost,e.cost_start),'reason':'predeclared stage or allocation reserve; not fabricated task failure'});raise
  r.update(phase=phase,candidate={'name':name,'execution':'stochastic' if stochastic else 'greedy','epsilon':policy.epsilon},method_protocol_sha256=sha(self.root/'method_protocol.json'))
  if isinstance(policy,Rule):r['candidate'].update(base_indices=policy.base,mode=policy.mode)
  else:r['candidate']['controller_sha256']=self.controller('ppo_current')['sha256']
  save(path,r)
  if general:
   e._guard(self.emergency,self.state);r['checkpoint']['adapter']=self.adapter(spec['id']+'__'+name);g=self.general.run(self.learner);save(path.with_name(path.stem+'__general.json'),g);r['general_ability']=compare_general(self.initial_general,g);r['general_cost_actual']=g['cost'];r['general_initial_shared_reused']={'sha256':sha(self.pilot/'results/short_comparison/initial_general.json'),'cost_in_original_pilot':self.initial_general['cost']};save(path,r)
  self.rows.append({'phase':phase,'root':spec['id'],'method':name,'success':r['success'],'F':r['evaluation']['forgetting'],'updates':r['cost_path']['updates'],'decisions':len(r['trace'])});self.progress('history_complete',result=self.rows[-1]);return r
 def train(self):
  specs=[];prefixes={}
  for raw in self.plan['roots']['train']:
   spec=self.target(raw);specs.append(spec);prefixes[spec['id']]=self.prefix(spec,'train',False)
  buffer=[];durations=[];deadline=float(os.environ['V13_STAGE_DEADLINE_EPOCH']);reason='history_cap'
  for i in range(self.plan['ppo']['maximum_histories']):
   # Admit complete pairs only; no outcome-dependent extension or root choice.
   if not buffer:
    estimate=max(self.plan['ppo']['minimum_pair_reserve_seconds'],2*max(durations[-3:] or [0])*1.25+30)
    if time.time()+estimate>=deadline:reason='predeclared_pair_admission_reserve';break
   spec=specs[i%len(specs)];self.state={'policy':self.policy.state_dict(),'optimizer':self.optim.state_dict(),'buffer':buffer,'visits':self.visits};start=time.time();r=self.run(spec,self.policy,f'ppo_rollout_{i:03}','train',prefixes[spec['id']],True,i);durations.append(time.time()-start);self.visits[spec['id']]+=1;buffer.append(training_trajectory(r))
   if len(buffer)==2:
    torch.set_num_threads(1);t=time.perf_counter();stats=ppo_update(self.policy,self.optim,buffer,epochs=self.plan['ppo']['epochs'],clip=.2,entropy_coef=0.,target_kl=.03);torch.set_num_threads(8);entry={'batch':len(self.learning)+1,'completed_histories':sum(self.visits.values()),'visits':dict(self.visits),'statistics':stats,'optimizer_steps':len(stats.get('epochs',[])),'wall_seconds':time.perf_counter()-t};self.learning.append(entry);buffer=[];entry['checkpoint']=self.controller(f'ppo_batch_{len(self.learning):03}');save(self.root/'learning.json',self.learning);self.progress('ppo_update',batch=entry['batch'],decisions=stats['decisions'],optimizer_steps=entry['optimizer_steps'])
  self.state={};self.controller('ppo_final');save(self.root/'training_status.json',{'visits':self.visits,'complete_histories':sum(self.visits.values()),'rollout_batches':len(self.learning),'optimizer_steps':sum(x['optimizer_steps'] for x in self.learning),'unused_complete_rollouts':len(buffer),'stop_reason':reason,'durations':durations});del prefixes;gc.collect()
 def development(self):
  # The pilot's deterministic fixed runs are exact reusable baselines, not new runs.
  fixed={name:json.loads((self.pilot/'results/short_comparison'/(name+'.json')).read_text()) for name in ['uniform1','static050']};best=min(fixed,key=lambda n:rank(fixed[n]));self.fixed_name=best;self.fixed_base=4 if best=='uniform1' else 2
  save(self.root/'fixed_lock.json',{'selected':best,'base_index':self.fixed_base,'menu':['uniform1','static050'],'selection_rule':self.plan['fixed_selection'],'scores':{n:rank(r) for n,r in fixed.items()},'reused_pilot_paths':{n:sha(self.pilot/'results/short_comparison'/(n+'.json')) for n in fixed},'scope':'best in two-candidate development menu; not globally optimal strong-static','locked_epoch':time.time()})
  spec=self.target(self.plan['roots']['development'][0]);prefix=self.prefix(spec,'development',True)
  self.run(spec,Rule([4]*16),'rule','development',prefix)
  self.run(spec,self.policy,'ppo_final','development',prefix)
  # No checkpoint/seed/decoder search: final completed PPO batch is the registered deployable policy.
  self.lock();del prefix;gc.collect()
 def lock(self):
  p=self.controller('ppo_selected');save(self.root/'confirmation_lock.json',{'controller':p,'selection':'last fully completed PPO batch; greedy frozen deployment, independent of confirmation/general scores','fixed_name':self.fixed_name,'fixed_base':self.fixed_base,'rule_base':4,'locked_epoch':time.time()})
 def confirmation(self):
  if not (self.root/'confirmation_lock.json').exists():self.lock()
  for raw in self.plan['roots']['confirmation']:
   spec=self.target(raw);prefix=self.prefix(spec,'confirmation',True);menu=[('uniform1',Rule([4]*16,mode='fixed'))]
   if self.fixed_name!='uniform1':menu.append((self.fixed_name,Rule([self.fixed_base]*16,mode='fixed')))
   menu.extend([('rule',Rule([4]*16)),('ppo_selected',self.policy)])
   for name,policy in menu:self.run(spec,policy,name,'confirmation',prefix,general=True)
   del prefix;gc.collect()
 def main(self):
  self.phase('train',self.plan['stage_caps_seconds']['train'],self.train)
  # A budget cut cannot expose a partially applied PPO optimization to deployment.
  self.controller('ppo_final');save(self.root/'training_status_final.json',{'visits':self.visits,'complete_histories':sum(self.visits.values()),'rollout_batches':len(self.learning),'optimizer_steps':sum(x['optimizer_steps'] for x in self.learning),'decisions':sum(x['statistics']['decisions'] for x in self.learning)})
  self.phase('development',self.plan['stage_caps_seconds']['development'],self.development)
  self.phase('confirmation',self.plan['stage_caps_seconds']['confirmation'],self.confirmation)
  self.learner.check_frozen();base=frozen_hash(self.learner.model);assert base==self.plan['frozen_base_hash'];save(self.root/'summary.json',{'status':'completed' if all(v['status']=='completed' for v in self.costs.values()) else 'bounded_incomplete','rows':self.rows,'stage_costs':self.costs,'native_cost':self.learner.cost,'wall_seconds':time.time()-self.started,'frozen_base_hash':base,'ended_epoch':time.time(),'learning_batches':len(self.learning),'training_visits':self.visits});self.progress('complete')
if __name__=='__main__':Workflow().main()

"""Three-task native continual learner with cached boundary potentials."""
import copy,hashlib,json,os,random,signal,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import torch
from structured_policy import shaped_rewards,Rule
from evaluation import task_generation_score

CHECKPOINT_REQUEST=False
def request_checkpoint(signum,frame):
 global CHECKPOINT_REQUEST
 CHECKPOINT_REQUEST=True
signal.signal(signal.SIGUSR1,request_checkpoint)
class BudgetStop(Exception):pass

@dataclass
class Config:
 tasks:int=3
 max_updates:int=60
 reference_updates:int=30
 hold:int=5
 microbatch:int=2
 accumulation:int=1
 replay_microbatch:int=1
 stop_examples:int=8
 old_observation_examples:int=2
 evaluation_batch:int=8
 memory_tokens:int=256000
 reward_scale:float=.1
 failure_kappa:float=3.

def cost_difference(now,before):return {k:now[k]-before[k] for k in now}
def token_prefix(rows,budget):
 result=[];total=0
 for row in rows:
  if total+row['input_tokens']>budget:break
  result.append(row);total+=row['input_tokens']
 return result
def hash_tree(value):
 h=hashlib.sha256()
 def visit(x):
  if isinstance(x,torch.Tensor):h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for key in sorted(x,key=str):h.update(str(key).encode());visit(x[key])
  elif isinstance(x,(list,tuple)):
   h.update(str(len(x)).encode())
   for v in x:visit(v)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

class Environment:
 def __init__(self,learner,dataset,spec,cfg=None,reference=False,answer_lookup=None):
  self.learner=learner;self.dataset=dataset;self.spec=copy.deepcopy(spec);self.cfg=cfg or Config();assert len(spec['tasks'])==self.cfg.tasks;self.reference=reference;self.answer_lookup=answer_lookup;self.cost_start=copy.deepcopy(learner.cost);self.started=time.perf_counter();learner.reset();torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
  self.k=0;self.step=0;self.memory={};self.records=[];self.matrix=[[None]*self.cfg.tasks for _ in range(self.cfg.tasks)];self.curve=[];self.previous=torch.ones(16,64);self.trace=[];self.held=None;self.held_remaining=0;self.cached_obs=None;self.cached_z=None;self.last_potential=0.;self.initial_z=None;self.monitor_cursor=0;self.last_decision_loss=None;self.last_old_loss=None;self.prefix_info=None;self.action_rng_started=False;self.score_matrix=[[None]*self.cfg.tasks for _ in range(self.cfg.tasks)];self.score_details={};self.control_seconds=0.;self.observation_seconds=0.;self.route_observation=None;self._initialise_task()
 def path_update_index(self):return sum(r['steps'] for r in self.records)+self.step
 def target(self):return self.spec['targets'][self.k] if self.spec.get('targets') else 0.
 def _initialise_task(self):
  rows=self.dataset[self.spec['tasks'][self.k]]['monitor'];rng=np.random.default_rng(self.spec['seed']*131+self.k);indices=rng.choice(len(rows),self.cfg.stop_examples,replace=False);self.stop_rows=[rows[int(i)] for i in indices];self.curve.append([]);self._measure_current();self.initial_loss=self.current_loss
 def _measure(self,rows,role='monitor',routes=False):
  losses=[];stats={}
  self.learner.model.eval()
  with torch.no_grad():
   for start in range(0,len(rows),self.cfg.evaluation_batch):
    ll,rr=self.learner.loss(rows[start:start+self.cfg.evaluation_batch],role,routes);losses.extend(ll.cpu().tolist())
    if routes:
     for layer,(use,mass,n) in rr.items():
      if layer not in stats:stats[layer]=[torch.zeros(64),torch.zeros(64),0]
      stats[layer][0]+=use;stats[layer][1]+=mass;stats[layer][2]+=n
  return losses,stats
 def _measure_current(self):
  ll,self.current_routes=self._measure(self.stop_rows,routes=True);self.current_loss=float(np.mean(ll));self.current_routes_at_update=self.path_update_index();self.curve[self.k].append({'step':self.step,'loss':self.current_loss})
 def _done_task(self):return self.step>=self.cfg.reference_updates if self.reference else self.current_loss<=self.target() or self.step>=self.cfg.max_updates
 def _finish(self):
  k=self.k;quota=self.cfg.memory_tokens//(2*(k+1))
  for memory in self.memory.values():
   memory['replay']=token_prefix(memory['replay'],quota);memory['monitor']=token_prefix(memory['monitor'],quota);memory['baseline']=memory['baseline'][:len(memory['monitor'])]
  memory={}
  for role,source in [('replay','train'),('monitor','monitor')]:
   rows=list(self.dataset[self.spec['tasks'][k]][source]);random.Random(self.spec['seed']+k*79+len(role)).shuffle(rows);memory[role]=token_prefix(rows,quota)
  memory['baseline'],_=self._measure(memory['monitor']);self.memory[k]=memory
  for j in range(k+1):
   rows=self.dataset[self.spec['tasks'][j]]['eval'];ll,_=self._measure(rows,'eval');self.matrix[k][j]=float(np.mean(ll))
   if self.answer_lookup is not None and not self.reference:
    score=task_generation_score(self.learner,rows,self.answer_lookup[self.spec['tasks'][j]],self.cfg.evaluation_batch);self.score_matrix[k][j]=score['normalized_exact_match'];self.score_details[f'{k}/{j}']=score
  self.records.append({'task':self.spec['tasks'][k],'position':k,'steps':self.step,'target':None if self.reference else self.target(),'initial_loss':self.initial_loss,'monitor_loss':self.current_loss,'success':None if self.reference else self.current_loss<=self.target(),'initial_success':False if self.reference else self.step==0 and self.current_loss<=self.target(),'memory_tokens':{r:sum(x['input_tokens'] for x in memory[r]) for r in ['replay','monitor']}})
  self.k+=1;self.step=0;self.held=None;self.held_remaining=0;self.cached_obs=None;self.cached_z=None;self.last_decision_loss=None;self.last_old_loss=None
  if self.k<self.cfg.tasks:self._initialise_task()
 def observation(self):
  if self.cached_obs is not None:return self.cached_obs,self.cached_z
  observation_started=time.perf_counter();old_usage=[];old_mass=[];old_losses=[];damages=[];old_tokens=0
  for j,memory in sorted(self.memory.items()):
   rows=memory['monitor'];start=(self.spec['seed']*97+self.k*13+self.step*2+j*31)%len(rows);indices=[(start+i)%len(rows) for i in range(min(self.cfg.old_observation_examples,len(rows)))];sample=[rows[i] for i in indices];ll,rr=self._measure(sample,routes=True);old_losses.append(float(np.mean(ll)));damages.append(max(float(np.mean(np.array(ll)-np.array(memory['baseline'])[indices])),0.));old_usage.append(torch.stack([rr[l][0]/max(rr[l][2],1) for l in range(16)]));old_mass.append(torch.stack([rr[l][1]/max(rr[l][2],1) for l in range(16)]));old_tokens+=rr[0][2]
  z=sum(damages)/(self.cfg.tasks-1);ou=torch.stack(old_usage).mean(0) if old_usage else torch.zeros(16,64);om=torch.stack(old_mass).mean(0) if old_mass else torch.zeros(16,64);cu=torch.stack([self.current_routes[l][0]/self.current_routes[l][2] for l in range(16)]);cm=torch.stack([self.current_routes[l][1]/self.current_routes[l][2] for l in range(16)]);old_loss=float(np.mean(old_losses)) if old_losses else 0.
  local=torch.stack([ou*8,cu*8,om*64,cm*64,self.learner.optim.relative.reshape(16,64)*100,self.previous,torch.full((16,64),min(old_tokens/2048,1)),torch.full((16,64),min(self.current_routes[0][2]/2048,1)),torch.full((16,64),float(self.path_update_index()-self.current_routes_at_update)/max(self.cfg.hold,1)),torch.zeros(16,64)],-1)
  failures=sum(r['success'] is False for r in self.records);global_=torch.tensor([self.current_loss/5,0. if self.last_decision_loss is None else (self.last_decision_loss-self.current_loss)/5,(self.current_loss-self.target())/5,old_loss/5,0. if self.last_old_loss is None else (self.last_old_loss-old_loss)/5,z/self.cfg.reward_scale,self.k/(self.cfg.tasks-1),(self.cfg.max_updates-self.step)/self.cfg.max_updates,failures/self.cfg.tasks,(self.last_potential-z)/self.cfg.reward_scale])
  self.last_decision_loss=self.current_loss;self.last_old_loss=old_loss;self.last_potential=z;self.cached_obs=(local,global_);self.cached_z=z;self.monitor_cursor+=1;self.observation_seconds+=time.perf_counter()-observation_started;self.route_observation={'measured_at_update':self.path_update_index(),'current_cache_age_updates':self.path_update_index()-self.current_routes_at_update,'old_cache_age_updates':0,'current_tokens_each_layer':[self.current_routes[l][2] for l in range(16)],'current_observed_experts_each_layer':[int((self.current_routes[l][0]>0).sum()) for l in range(16)],'old_sample_tokens_per_layer':old_tokens,'old_tasks_observed':len(old_usage),'missing_old_memory':not bool(old_usage)};return self.cached_obs,z
 def _update(self,action):
  rows=self.dataset[self.spec['tasks'][self.k]]['train'];rng=np.random.default_rng(self.spec['seed']*1000003+self.k*10007+self.step);current=[];replay=[]
  for i in range(self.cfg.accumulation):
   current.append([rows[int(x)] for x in rng.integers(len(rows),size=self.cfg.microbatch)])
   if self.memory:
    j=sorted(self.memory)[(self.step+i)%len(self.memory)];old=self.memory[j]['replay'];replay.append([old[int(x)] for x in rng.integers(len(old),size=self.cfg.replay_microbatch)])
  self.learner.step(current,replay,action);self.step+=1;self.previous=action.clone();self.cached_obs=None;self.cached_z=None;self._measure_current()
 def save_interrupt(self,path,extra=None):
  state={'learner':self.learner.snapshot(),'environment':self.state_data(),'config':asdict(self.cfg),'extra':extra or {},'native_cost':self.learner.cost};temporary=Path(str(path)+'.tmp');torch.save(state,temporary);temporary.replace(path)
 def _guard(self,checkpoint_path,extra):
  if CHECKPOINT_REQUEST or time.time()>=min(float(os.environ.get('V13_DEADLINE_EPOCH','inf'))-180,float(os.environ.get('V13_STAGE_DEADLINE_EPOCH','inf'))):
   self.save_interrupt(checkpoint_path,extra);raise BudgetStop('Saved recoverable state at allocation reserve')
 def state_data(self):return copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in ['learner','dataset','cfg','cost_start','started','answer_lookup']})
 @classmethod
 def from_state(cls,learner,dataset,cfg,state,learner_state,answer_lookup=None):
  obj=cls.__new__(cls);obj.answer_lookup=answer_lookup;obj.learner=learner;obj.dataset=dataset;obj.cfg=cfg;obj.cost_start=copy.deepcopy(learner.cost);obj.started=time.perf_counter();obj.__dict__.update(copy.deepcopy(state));learner.restore(learner_state);return obj
 def prepare_prefix(self,checkpoint_path):
  while self.k==0:
   self._guard(checkpoint_path,{})
   if self._done_task():self._finish();break
   self._update(torch.ones(16,64))
  before=copy.deepcopy(self.learner.cost);execution_seconds=time.perf_counter()-self.started;state={'learner':self.learner.snapshot(),'environment':self.state_data(),'config':asdict(self.cfg),'base_revision':'6d84c48581ece794365f2b8e9cfb043c68ade9c5','adapter_protocol':'rank4_alpha4_dropout0_seed1400_all16x64'};start=time.perf_counter();fingerprint=hash_tree(state);fingerprint_seconds=time.perf_counter()-start
  info={'sha256':fingerprint,'root':self.spec,'cost':cost_difference(before,self.cost_start),'snapshot_cost':cost_difference(self.learner.cost,before),'fingerprint_seconds':fingerprint_seconds,'execution_wall_seconds':execution_seconds,'wall_seconds':time.perf_counter()-self.started};state['environment']['prefix_info']=info;return state,info
 def run(self,policy,stochastic=False,action_seed=0,teacher=None,reference_policy=None,checkpoint_path=Path('emergency.pt'),extra=None):
  if not self.action_rng_started:torch.manual_seed(action_seed);self.action_rng_started=True
  while self.k<self.cfg.tasks:
   self._guard(checkpoint_path,extra)
   if self._done_task():self._finish();continue
   if self.k==0:self._update(torch.ones(16,64));continue
   if self.held is None:
    obs,z=self.observation()
    if self.trace and self.trace[-1]['reward'] is None:self.trace[-1]['reward']=(self.trace[-1]['potential']-z)/self.cfg.reward_scale
    if self.initial_z is None:self.initial_z=z
    control_started=time.perf_counter()
    with torch.no_grad():
     action,latent,lp,entropy,value=policy.act(obs,deterministic=not stochastic);reference_lp=float(reference_policy.act(obs,latent)[2]) if reference_policy is not None else None;teacher_action=teacher.act(obs,deterministic=True)[0] if teacher is not None else None
    self.control_seconds+=time.perf_counter()-control_started
    self.trace.append({'route_observation':copy.deepcopy(self.route_observation),'task':self.k,'step':self.step,'duration':0,'experts':action.tolist(),'latent':[int(v) for v in latent],'policy_logp':float(lp),'ref_logp':reference_lp,'value':float(value),'entropy':float(entropy),'observation':[v.tolist() for v in obs],'potential':z,'reward':None,'rule_action_match':None if teacher_action is None else bool(torch.equal(action,teacher_action))});self.held=action.clone();self.held_remaining=self.cfg.hold
   self._update(self.held);self.trace[-1]['duration']+=1;self.held_remaining-=1
   if self.held_remaining==0:self.held=None
  monitor_signed=[];monitor_positive=[]
  for j in range(self.cfg.tasks-1):
   memory=self.memory[j];ll,_=self._measure(memory['monitor']);delta=float(np.mean(np.array(ll)-np.array(memory['baseline'])));monitor_signed.append(delta);monitor_positive.append(max(delta,0.))
  full_F=float(np.mean(monitor_positive));failures=sum(r['success'] is False for r in self.records);raw_objective=-full_F/self.cfg.reward_scale-self.cfg.failure_kappa*failures
  if self.trace:
   self.trace[-1]['reward']=(self.trace[-1]['potential']-full_F)/self.cfg.reward_scale-self.cfg.failure_kappa*failures;expected=shaped_rewards([t['potential'] for t in self.trace],full_F,failures,self.cfg.reward_scale,self.cfg.failure_kappa);assert np.allclose(expected,[t['reward'] for t in self.trace],atol=1e-10);shaped_return=sum(expected)
  else:self.initial_z=0.;shaped_return=raw_objective
  evaluation_signed=[self.matrix[-1][j]-self.matrix[j][j] for j in range(self.cfg.tasks-1)];self.learner.check_frozen();cost=cost_difference(self.learner.cost,self.cost_start);cost.update({'control_decisions':len(self.trace),'wall_seconds':time.perf_counter()-self.started});path_cost=cost.copy()
  if self.prefix_info:
   for k,v in self.prefix_info['cost'].items():path_cost[k]+=v
   path_cost['wall_seconds']+=self.prefix_info['execution_wall_seconds']
  result={'protocol_revision':'v14_full_layer_expert_lora_v1','root':self.spec,'reference_calibration':self.reference,'checkpoint':{'base_revision':'6d84c48581ece794365f2b8e9cfb043c68ade9c5','adapter_protocol':'rank4_alpha4_dropout0_seed1400_all16x64','prefix_sha256':None if self.prefix_info is None else self.prefix_info['sha256']},'batch_stream':{'id':0,'scheme':'root/task/local-update deterministic train and replay streams','independent_future':False},'tasks':self.records,'success':None if self.reference else all(r['success'] for r in self.records),'failure_count':failures,'unexecuted_tasks':[],'evaluation':{'matrix':self.matrix,'score_matrix':self.score_matrix,'score_details':self.score_details,'forgetting':float(np.maximum(evaluation_signed,0).mean()),'signed_forgetting':float(np.mean(evaluation_signed)),'final_loss':float(np.mean(self.matrix[-1]))},'monitor':{'forgetting':full_F,'signed_forgetting':float(np.mean(monitor_signed)),'full_memory':True,'denominator':self.cfg.tasks-1,'retained_tokens':{role:sum(x['input_tokens'] for m in self.memory.values() for x in m[role]) for role in ['replay','monitor']}},'reward':{'terminal_objective':raw_objective,'shaped_return':shaped_return,'initial_z':self.initial_z,'scale':self.cfg.reward_scale,'kappa':self.cfg.failure_kappa,'telescope_error':shaped_return-raw_objective-self.initial_z/self.cfg.reward_scale,'zero_decision_terminal_event':not bool(self.trace)},'cost_actual':cost,'cost_path':path_cost,'trace':self.trace,'loss_curves':self.curve}
  result.update({'online_control_seconds':self.control_seconds,'observation_seconds':self.observation_seconds,'score_evaluation_cost_path':{key:sum(v['cost'][key] for v in self.score_details.values()) for key in ['input_tokens','padded_prompt_tokens','generated_tokens','wall_seconds']},'history_completed':True,'history_success':result['success'],'n_fail':failures});assert abs(result['reward']['telescope_error'])<1e-8;return result

def training_trajectory(result):
 tr=[]
 for t in result['trace']:
  q={'obs':tuple(torch.tensor(v) for v in t['observation']),'latent':[torch.tensor(v) for v in t['latent']],'logp':t['policy_logp'],'value':t['value']}
  if t['ref_logp'] is not None:q['ref_logp']=t['ref_logp']
  tr.append(q)
 return {'transitions':tr,'rewards':[t['reward'] for t in result['trace']]}

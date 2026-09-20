"""Authorized <=10-minute correctness reproduction; never a method confirmation."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(os.environ['V14_METHOD_ROOT']);sys.path.insert(0,str(ROOT))
from environment import Environment,Config,hash_tree,cost_difference
from lora_native import NativeLearner
from lora import frozen_hash
from structured_policy import ActorCritic,Rule

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def details(state):
 opt=state['optimizer'];moments=opt['state'];return {'weights':hash_tree(state['weights']),'optimizer':hash_tree(opt),'step':hash_tree({k:v['step'] for k,v in moments.items()}),'exp_avg':hash_tree({k:v['exp_avg'] for k,v in moments.items()}),'exp_avg_sq':hash_tree({k:v['exp_avg_sq'] for k,v in moments.items()}),'rng':hash_tree({k:state[k] for k in ['torch_rng','cuda_rng','numpy_rng','python_rng']}),'relative':hash_tree(state['relative'])}
def differences(a,b):
 out={}
 for name,aa,bb in [('weights',a['weights'],b['weights']),('exp_avg',[v['exp_avg'] for v in a['optimizer']['state'].values()],[v['exp_avg'] for v in b['optimizer']['state'].values()]),('exp_avg_sq',[v['exp_avg_sq'] for v in a['optimizer']['state'].values()],[v['exp_avg_sq'] for v in b['optimizer']['state'].values()])]:
  out[name]={'equal':all(torch.equal(x,y) for x,y in zip(aa,bb)),'max_abs':max(float((x-y).abs().max()) for x,y in zip(aa,bb)),'l2':sum(float((x-y).double().square().sum()) for x,y in zip(aa,bb))**.5}
 return out

def main():
 started=time.time();out=ROOT/'debug_restore';out.mkdir(exist_ok=True);inputfile=ROOT/'emergency.pt';state=torch.load(inputfile,map_location='cpu',weights_only=False);input_sha=sha(inputfile);saved_hash=hash_tree(state);learner=NativeLearner(str(Path(os.environ['V13_ALLOCATION_ROOT'])/'model'));base_before=frozen_hash(learner.model);data=torch.load(Path(os.environ['V13_ALLOCATION_ROOT'])/'dataset.pt',weights_only=False);actor=ActorCritic();actor.load_state_dict(torch.load(ROOT/'controllers/ppo_selected.pt',map_location='cpu',weights_only=False)['policy']);actor.eval();cfg=Config(**state['config']);original_step=learner.step;original_loss=learner.loss;capture={}
 def recording_loss(examples,role='monitor',routes=False,pad_to=None):
  result=original_loss(examples,role,routes,pad_to)
  if role in ['train','replay']:capture.setdefault(role,[]).append({'batch_sha256':hash_tree(examples),'losses':result[0].detach().cpu().tolist()})
  return result
 learner.loss=recording_loss
 rows=[];after_states=[]
 for i,kind in enumerate(['fixed','fixed','ppo','ppo']):
  if time.time()-started>480:break
  capture={};before_input=hash_tree(state);e=Environment.from_state(learner,data,cfg,state['environment'],state['learner']);learner.optim.zero_grad();before=learner.snapshot();obs,z=e.observation();before_action_rng=hash_tree({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()});policy=Rule([4]*16,mode='fixed') if kind=='fixed' else actor
  with torch.no_grad():action,latent,lp,entropy,value=policy.act(obs,deterministic=True)
  after_action_rng=hash_tree({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()});assert torch.equal(action,torch.ones(16,64));e._update(action);after=learner.snapshot();learner.check_frozen();after_input=hash_tree(state);path=out/f'after_{i}_{kind}.pt';torch.save(after,path)
  row={'case':i,'policy':kind,'input_immutable_before':before_input==saved_hash,'input_immutable_after':after_input==saved_hash,'input_hash_before':before_input,'input_hash_after':after_input,'before':details(before),'after':details(after),'restored_weights_match_input':hash_tree(before['weights'])==hash_tree(state['learner']['weights']),'restored_optimizer_match_input':hash_tree(before['optimizer'])==hash_tree(state['learner']['optimizer']),'restored_rng_match_input':details(before)['rng']==details(state['learner'])['rng'],'observation_sha256':hash_tree(obs),'action_sha256':hash_tree(action),'policy_did_not_change_rng':before_action_rng==after_action_rng,'batches_and_losses':capture,'monitor_loss_after':e.current_loss,'checkpoint':{'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size},'elapsed_seconds':time.time()-started};rows.append(row);after_states.append(after);save(out/'progress.json',{'cases':rows,'native_cost':learner.cost});print(json.dumps({'case':i,'policy':kind,'monitor_loss_after':e.current_loss,'input_immutable':row['input_immutable_after'],'elapsed':time.time()-started}),flush=True);del before
 comparisons=[{'a':a,'b':b,'difference':differences(after_states[a],after_states[b])} for a,b in [(0,1),(0,2),(2,3)] if b<len(after_states)];base_after=frozen_hash(learner.model);assert base_after==base_before
 save(out/'result.json',{'status':'completed','purpose':'Engineering replay only; not original confirmation or original endpoint SHA recovery','input_checkpoint_sha256':input_sha,'input_root':state['environment']['spec']['id'],'input_task_position':state['environment']['k'],'input_local_step':state['environment']['step'],'controller_sha256':sha(ROOT/'controllers/ppo_selected.pt'),'cases':rows,'comparisons':comparisons,'base_before_new_debug_run':base_before,'base_after_new_debug_run':base_after,'native_cost':learner.cost,'wall_seconds':time.time()-started,'finished_epoch':time.time(),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'torch_version':torch.__version__})
if __name__=='__main__':main()

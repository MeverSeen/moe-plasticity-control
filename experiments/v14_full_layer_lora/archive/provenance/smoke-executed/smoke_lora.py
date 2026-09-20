"""Bounded real-GPU integration checks and end-to-end profiles, not a CL benchmark."""
import gc,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from lora_native import NativeLearner
from lora import set_enabled,frozen_hash

STOP=False
def signal_stop(*args):
 global STOP
 STOP=True
signal.signal(signal.SIGUSR1,signal_stop)
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def flat(learner):return torch.cat([p.detach().flatten() for p in learner.parameters])
def moment_hash(learner):
 h=hashlib.sha256()
 for state in learner.optim.optimizer.state.values():
  for key in ['step','exp_avg','exp_avg_sq']:
   if key in state:h.update(key.encode());h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 root=Path(os.environ['V14_WORK_ROOT']);allocation=Path(os.environ['V13_ALLOCATION_ROOT']);started=time.time();result={'status':'running','protocol':'v14_full_layer_expert_lora_v1','profiles':[]};learner=None
 def report(phase,**kw):
  result.update(phase=phase,**kw);save(root/'smoke_progress.json',result);print(phase,json.dumps(kw),flush=True)
 def guard():
  if STOP or time.time()>min(float(os.environ['V13_DEADLINE_EPOCH'])-180,started+1800):
   if learner is not None:torch.save({'learner':learner.snapshot(),'result':result},root/'emergency.pt')
   report('budget_stop',status='incomplete');raise SystemExit(124)
 t=time.perf_counter();learner=NativeLearner(str(allocation/'model'));load=time.perf_counter()-t;data=torch.load(allocation/'dataset.pt',weights_only=False);manifest=json.loads((allocation/'data_manifest.json').read_text());examples=data[manifest['splits']['train'][0]]['train'][:8];replay=data[manifest['splits']['train'][1]]['train'][:2]
 assert len(learner.groups)==1024 and len(learner.adapters)==3072 and sum(p.numel() for p in learner.parameters)==37748736
 assert all(p.dtype==torch.float32 for p in learner.parameters);assert all(p.requires_grad==('.A' in name or '.B' in name) for name,p in learner.model.named_parameters())
 report('scope',trainable_parameters=37748736,groups=1024,projections=3072,model_load_seconds=load,base_revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5',master_dtype='float32',compute_dtype='bfloat16',initialization_seed=1400)
 guard();t=time.perf_counter();base_hash=frozen_hash(learner.model);report('frozen_hash_before',base_hash=base_hash,hash_seconds=time.perf_counter()-t)
 learner.model.eval();ids,labels,mask=learner.pack(examples[:2],512)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  set_enabled(learner.adapters,False);baseline=learner.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
  set_enabled(learner.adapters,True);enabled=learner.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
  equal=torch.equal(baseline,enabled);diff=float((baseline-enabled).abs().max());assert equal
 del baseline,enabled,ids,labels,mask;gc.collect();report('zero_initialization',initial_logits_bitwise_equal=equal,initial_logits_max_abs_difference=diff)
 guard();learner.step([examples[:2]],[replay[:1]],torch.ones(16,64),pad_to=512);learned=flat(learner).clone();initial=torch.cat([p.flatten() for p in learner.initial]).cuda();nonzero=float((learned-initial).norm());assert nonzero>0;del initial
 with torch.no_grad():before0,_=learner.loss(examples,'monitor',routes=True)
 state=learner.snapshot();learner.step([examples[:2]],[replay[:1]],torch.ones(16,64),pad_to=512);full=flat(learner).clone();mom_full=moment_hash(learner);learner.restore(state);learner.step([examples[:2]],[replay[:1]],torch.zeros(16,64),pad_to=512);zero_exact=torch.equal(flat(learner),learned);assert zero_exact;mom_zero=moment_hash(learner);assert mom_zero==mom_full
 with torch.no_grad():after0,_=learner.loss(examples,'monitor',routes=True)
 assert torch.equal(before0,after0)
 learner.restore(state);multipliers=(torch.arange(1024)%5).float()/4;learner.step([examples[:2]],[replay[:1]],multipliers.reshape(16,64),pad_to=512);actual=flat(learner);expanded=torch.cat([torch.full((sum(p.numel() for p in group),),float(m)) for group,m in zip(learner.groups,multipliers)]).cuda();expected=learned+expanded*(full-learned);error=float((actual-expected).abs().max());assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6);assert moment_hash(learner)==mom_full
 report('displacement_checks',nonzero_update_norm=nonzero,zero_multiplier_exact=zero_exact,zero_keeps_learned_forward=True,zero_moments_match_full_update=True,mixed_multiplier_max_abs_error=error,active_factor_tensors=learner.optim.active_parameters)
 del state,learned,full,actual,expected,expanded;gc.collect();guard()
 for batch in [1,2]:
  learner.step([examples[:batch]],[replay[:1]],torch.ones(16,64),pad_to=512) # one warmup per batch, counted separately
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();cost_before=learner.cost.copy();losses=[]
  for _ in range(3):guard();losses.append(learner.step([examples[:batch]],[replay[:1]],torch.ones(16,64),pad_to=512))
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;cost={k:learner.cost[k]-cost_before[k] for k in cost_before};monitor=[];routes=None
  for collect in [False,True,False,True]:
   torch.cuda.synchronize();q=time.perf_counter()
   with torch.no_grad():ll,stats=learner.loss(examples,'monitor',routes=collect,pad_to=512)
   torch.cuda.synchronize();monitor.append({'routes':collect,'seconds':time.perf_counter()-q,'mean_loss':float(ll.mean())})
   if collect:routes=stats
  plain=[x['seconds'] for x in monitor if not x['routes']];routed=[x['seconds'] for x in monitor if x['routes']];profile={'microbatch':batch,'sequence_length_padded':512,'replay_microbatch':1,'replay_sequence_length_padded':512,'accumulation':1,'updates':3,'training_and_replay_seconds':elapsed,'seconds_per_update':elapsed/3,'losses':losses,'cost':cost,'monitor8_profiles':monitor,'route_collection_incremental_seconds':float(np.mean(routed)-np.mean(plain)),'incremental_timing_scope':'Alternating with/without routes; noisy descriptive difference, not isolated kernel benchmark','route_tokens_each_layer':[routes[i][2] for i in range(16)],'observed_experts_each_layer':[int((routes[i][0]>0).sum()) for i in range(16)],'route_cache_age_updates':learner.cost['updates']-learner.route_profile['last_update'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};result['profiles'].append(profile);report('profile',last_profile=profile)
 guard();expected=flat(learner).cpu();expected_mom=moment_hash(learner);state=learner.snapshot();path=root/'smoke_recovery.pt';q=time.perf_counter();torch.save(state,path);save_seconds=time.perf_counter()-q;checkpoint_bytes=path.stat().st_size;del state;gc.collect()
 with torch.no_grad():learner.parameters[0].add_(.01)
 q=time.perf_counter();state=torch.load(path,map_location='cpu',weights_only=False);learner.restore(state);restore_seconds=time.perf_counter()-q;assert torch.equal(flat(learner).cpu(),expected);assert moment_hash(learner)==expected_mom;assert torch.equal(torch.get_rng_state(),state['torch_rng']);assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng']));moment_dtypes=sorted({str(s['exp_avg'].dtype) for s in learner.optim.optimizer.state.values()});assert moment_dtypes==['torch.float32'];learner.check_frozen();after_hash=frozen_hash(learner.model);assert after_hash==base_hash
 result.update(status='completed',checkpoint={'bytes':checkpoint_bytes,'save_seconds':save_seconds,'load_restore_seconds':restore_seconds,'weights_exact':True,'optimizer_moments_exact':True,'cpu_cuda_rng_exact':True,'moment_dtypes':moment_dtypes},frozen_base_hash_after=after_hash,frozen_base_bitwise_unchanged=True,native_cost=learner.cost,route_profile=learner.route_profile,wall_seconds=time.time()-started,finished_epoch=time.time());save(root/'smoke_result.json',result);report('complete');path.unlink()
if __name__=='__main__':main()

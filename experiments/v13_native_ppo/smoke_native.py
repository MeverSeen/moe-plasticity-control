import gc,json,os,time
from pathlib import Path
import torch
from native import NativeLearner
root=Path(os.environ['V13_ALLOCATION_ROOT']);torch.manual_seed(1300);start=time.perf_counter();learner=NativeLearner(str(root/'model'));load=time.perf_counter()-start
raw=json.loads((root/'smoke_task.json').read_text());examples=[learner.encode(raw['Definition'][0],v['input'],v['output'][0]) for v in raw['Instances'] if len(v['input'])<700][:8];assert all(len(e['ids'])<=512 for e in examples)
result={'model_load_seconds':load,'trainable_parameters':sum(p.numel() for p in learner.parameters),'layers_one_based':[12,16],'experts':128,'master_weight_dtype':str(learner.parameters[0].dtype),'compute_dtype':'torch.bfloat16','profiles':[]}
for batch in [1,2]:
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();losses=[]
 for _ in range(2):losses.append(learner.step([examples[:batch]],[],torch.full((2,64),.5),pad_to=512))
 torch.cuda.synchronize();elapsed=time.perf_counter()-t
 with torch.no_grad():
  t=time.perf_counter();ll,route=learner.loss(examples,'monitor',routes=True);torch.cuda.synchronize();monitor=time.perf_counter()-t
 result['profiles'].append({'batch':batch,'sequence_length':512,'updates':2,'seconds':elapsed,'padded_training_tokens_per_second':batch*512*2/elapsed,'update_seconds':elapsed/2,'monitor8_seconds':monitor,'losses':losses,'monitor_loss':ll.mean().item(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'route_tokens':[r[2] for r in route.values()]});print(json.dumps(result['profiles'][-1]),flush=True)
state=learner.snapshot();path=root/'smoke-recovery.pt';t=time.perf_counter();torch.save(state,path);save_seconds=time.perf_counter()-t;size=path.stat().st_size;del state;gc.collect();t=time.perf_counter();state=torch.load(path,map_location='cpu',weights_only=False);learner.restore(state);restore_seconds=time.perf_counter()-t
assert all(torch.equal(p.detach().cpu(),x) for p,x in zip(learner.parameters,state['weights']));learner.check_frozen();dtypes=sorted({str(s['exp_avg'].dtype) for s in learner.optim.optimizer.state.values()});assert dtypes==['torch.float32']
result.update({'optimizer_moment_dtypes':dtypes,'checkpoint_bytes':size,'save_seconds':save_seconds,'load_restore_seconds':restore_seconds,'restore_weights_exact':True,'frozen_versions_unchanged':True,'cost':learner.cost,'elapsed_seconds':time.perf_counter()-start});(root/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n');print('SMOKE_COMPLETE',json.dumps(result),flush=True);path.unlink()

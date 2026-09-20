"""One resident pretrained model; FP32 editable experts, BF16 forward, FP32 Adam."""
import copy,random,time
import numpy as np
import torch
from torch.nn import functional as F
from mechanisms import GatedAdamW

def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return copy.deepcopy(x)

class NativeLearner:
 def __init__(self,model_path,lr=2e-5,threads=8):
  from transformers import AutoTokenizer,OlmoeForCausalLM
  torch.set_num_threads(threads);self.lr=lr;self.layers=[11,15];self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
  self.model=OlmoeForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();self.model.config.use_cache=False;self.model.config.output_router_logits=False
  self.model.requires_grad_(False);self.groups=[]
  for layer in self.layers:
   for expert in self.model.model.layers[layer].mlp.experts:
    group=list(expert.parameters())
    for p in group:p.data=p.data.float();p.requires_grad_(True)
    self.groups.append(group)
  self.parameters=[p for g in self.groups for p in g];assert sum(p.numel() for p in self.parameters)==805306368
  self.frozen_versions={name:p._version for name,p in self.model.named_parameters() if not p.requires_grad};self.initial=[p.detach().cpu().clone() for p in self.parameters];self.optim=GatedAdamW(self.groups,lr);self.route_mask=None;self.route_stats={};self.hooks=[]
  for index,layer in enumerate(self.layers):
   def hook(module,args,result,index=index):
    if self.route_mask is None:return
    logits=result.detach().reshape(-1,64)[self.route_mask];weights=logits.float().softmax(-1);v,ix=weights.topk(8,-1);use=torch.zeros(64,device=weights.device).scatter_add_(0,ix.flatten(),torch.ones_like(v).flatten());mass=torch.zeros_like(use).scatter_add_(0,ix.flatten(),v.flatten());self.route_stats[index]=(use,mass,len(logits))
   self.hooks.append(self.model.model.layers[layer].mlp.gate.register_forward_hook(hook))
  self.cost={k:0 for k in ['updates','train_input_tokens','train_target_tokens','train_padded_tokens','replay_input_tokens','replay_target_tokens','replay_padded_tokens','monitor_input_tokens','monitor_target_tokens','monitor_padded_tokens','eval_input_tokens','eval_target_tokens','eval_padded_tokens','copies','resets']};self.cost['copy_seconds']=0.;self.cost['reset_seconds']=0.
 def encode(self,definition,input_,answer):
  prompt='Instruction: '+definition.strip()+'\nInput: '+input_.strip()+'\nOutput:';a=self.tokenizer.encode(' '+answer.strip(),add_special_tokens=False)+[self.tokenizer.eos_token_id];p=self.tokenizer.encode(prompt,add_special_tokens=False)
  return {'ids':p+a,'labels':[-100]*len(p)+a,'input_tokens':len(p+a),'target_tokens':len(a)}
 def pack(self,examples,pad_to=None):
  length=max(len(e['ids']) for e in examples) if pad_to is None else pad_to;assert length<=512
  ids=torch.full((len(examples),length),self.tokenizer.pad_token_id,dtype=torch.long,device='cuda');labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
  for i,e in enumerate(examples):
   n=len(e['ids']);ids[i,:n]=torch.tensor(e['ids'],device='cuda');labels[i,:n]=torch.tensor(e['labels'],device='cuda');mask[i,:n]=1
  return ids,labels,mask
 def loss(self,examples,role='monitor',routes=False,pad_to=None):
  ids,labels,mask=self.pack(examples,pad_to);self.cost[role+'_input_tokens']+=int(mask.sum());self.cost[role+'_target_tokens']+=int((labels!=-100).sum());self.cost[role+'_padded_tokens']+=ids.numel();self.route_mask=mask.flatten().bool() if routes else None;self.route_stats={}
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
  loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(len(examples),-1);valid=labels[:,1:]!=-100;per_example=loss.sum(1)/valid.sum(1)
  route=None
  if routes:route={k:(v[0].cpu(),v[1].cpu(),v[2]) for k,v in self.route_stats.items()}
  self.route_mask=None;return per_example,route
 def step(self,current,replay,multipliers,pad_to=None):
  self.model.train();self.optim.zero_grad();losses=[]
  for batch in current:
   ll,_=self.loss(batch,'train',pad_to=pad_to);loss=ll.mean()/len(current);loss.backward();losses.append(float(ll.detach().mean()))
  if replay:
   for batch in replay:
    ll,_=self.loss(batch,'replay');(.5*ll.mean()/len(replay)).backward()
  grad=torch.nn.utils.clip_grad_norm_(self.parameters,1.);assert torch.isfinite(grad);self.optim.step(multipliers.flatten());self.cost['updates']+=1;return float(np.mean(losses))
 def reset(self):
  start=time.perf_counter()
  with torch.no_grad():
   for p,x in zip(self.parameters,self.initial):p.copy_(x)
  self.optim=GatedAdamW(self.groups,self.lr)
  self.cost['resets']+=1;self.cost['reset_seconds']+=time.perf_counter()-start
 def snapshot(self):
  start=time.perf_counter();state={'weights':[p.detach().cpu().clone() for p in self.parameters],'optimizer':cpu_tree(self.optim.optimizer.state_dict()),'relative':self.optim.relative.clone(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()};self.cost['copies']+=1;self.cost['copy_seconds']+=time.perf_counter()-start;return state
 def restore(self,state):
  start=time.perf_counter()
  with torch.no_grad():
   for p,x in zip(self.parameters,state['weights']):p.copy_(x)
  self.optim.optimizer.load_state_dict(state['optimizer']);self.optim.relative=state['relative'].clone();torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);self.cost['copies']+=1;self.cost['copy_seconds']+=time.perf_counter()-start
 def check_frozen(self):assert self.frozen_versions=={name:p._version for name,p in self.model.named_parameters() if not p.requires_grad}

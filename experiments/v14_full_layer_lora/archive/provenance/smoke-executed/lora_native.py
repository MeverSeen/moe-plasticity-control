"""All native experts have rank4 LoRA; frozen BF16 base, FP32 factors and Adam."""
import copy,random,time
import numpy as np
import torch
from torch.nn import functional as F
from optimizer import GatedAdamW

def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return copy.deepcopy(x)

class NativeLearner:
 def __init__(self,model_path,lr=2e-4,threads=8,seed=1400):
  from transformers import AutoTokenizer,OlmoeForCausalLM
  from lora import insert_lora
  torch.set_num_threads(threads);self.lr=lr;self.layers=list(range(16));self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
  self.model=OlmoeForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();self.model.config.use_cache=False;self.model.config.output_router_logits=False
  assert (self.model.config.num_hidden_layers,self.model.config.num_experts,self.model.config.num_experts_per_tok)==(16,64,8)
  self.model.requires_grad_(False);torch.manual_seed(seed);self.groups,self.adapters=insert_lora(self.model,rank=4,alpha=4)
  self.parameters=[p for group in self.groups for p in group];assert sum(p.numel() for p in self.parameters)==37748736
  self.frozen_versions={name:p._version for name,p in self.model.named_parameters() if not p.requires_grad};self.initial=[p.detach().cpu().clone() for p in self.parameters];self.optim=GatedAdamW(self.groups,lr);self.route_mask=None;self.route_stats={};self.hooks=[];self.route_profile={'calls':0,'seconds_including_forward':0.,'tokens_per_layer':0,'last_update':None}
  for index,layer in enumerate(self.layers):
   def hook(module,args,result,index=index):
    if self.route_mask is None:return
    logits=result.detach().reshape(-1,64)[self.route_mask];weights=logits.float().softmax(-1);v,ix=weights.topk(8,-1)
    if self.model.config.norm_topk_prob:v=v/v.sum(-1,keepdim=True)
    use=torch.zeros(64,device=weights.device).scatter_add_(0,ix.flatten(),torch.ones_like(v).flatten());mass=torch.zeros_like(use).scatter_add_(0,ix.flatten(),v.flatten());self.route_stats[index]=(use,mass,len(logits))
   self.hooks.append(self.model.model.layers[layer].mlp.gate.register_forward_hook(hook))
  self.cost={k:0 for k in ['updates','train_input_tokens','train_target_tokens','train_padded_tokens','replay_input_tokens','replay_target_tokens','replay_padded_tokens','monitor_input_tokens','monitor_target_tokens','monitor_padded_tokens','eval_input_tokens','eval_target_tokens','eval_padded_tokens','copies','resets']};self.cost.update(copy_seconds=0.,reset_seconds=0.,train_forward_seconds=0.,replay_forward_seconds=0.,monitor_forward_seconds=0.,eval_forward_seconds=0.,train_backward_seconds=0.,replay_backward_seconds=0.,optimizer_seconds=0.)
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
  started=time.perf_counter();ids,labels,mask=self.pack(examples,pad_to);self.cost[role+'_input_tokens']+=int(mask.sum());self.cost[role+'_target_tokens']+=int((labels!=-100).sum());self.cost[role+'_padded_tokens']+=ids.numel();self.route_mask=mask.flatten().bool() if routes else None;self.route_stats={}
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
  loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(len(examples),-1);valid=labels[:,1:]!=-100;per_example=loss.sum(1)/valid.sum(1)
  route=None
  if routes:route={k:(v[0].cpu(),v[1].cpu(),v[2]) for k,v in self.route_stats.items()}
  self.route_mask=None;torch.cuda.synchronize();elapsed=time.perf_counter()-started;self.cost[role+'_forward_seconds']+=elapsed
  if routes:self.route_profile['calls']+=1;self.route_profile['seconds_including_forward']+=elapsed;self.route_profile['tokens_per_layer']+=int(mask.sum());self.route_profile['last_update']=self.cost['updates']
  return per_example,route
 def step(self,current,replay,multipliers,pad_to=None):
  self.model.train();self.optim.zero_grad();losses=[]
  for batch in current:
   ll,_=self.loss(batch,'train',pad_to=pad_to);loss=ll.mean()/len(current);start=time.perf_counter();loss.backward();torch.cuda.synchronize();self.cost['train_backward_seconds']+=time.perf_counter()-start;losses.append(float(ll.detach().mean()))
  if replay:
   for batch in replay:
    ll,_=self.loss(batch,'replay',pad_to=pad_to);start=time.perf_counter();(.5*ll.mean()/len(replay)).backward();torch.cuda.synchronize();self.cost['replay_backward_seconds']+=time.perf_counter()-start
  grad=torch.nn.utils.clip_grad_norm_(self.parameters,1.);assert torch.isfinite(grad);start=time.perf_counter();self.optim.step(multipliers.flatten());torch.cuda.synchronize();self.cost['optimizer_seconds']+=time.perf_counter()-start;self.cost['updates']+=1;return float(np.mean(losses))
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

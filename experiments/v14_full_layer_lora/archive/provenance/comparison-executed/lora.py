"""Native-expert LoRA only: one persistent adapter set, no new router."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
 def __init__(self,base,rank=4,alpha=4):
  super().__init__();assert isinstance(base,nn.Linear) and rank>0;self.base=base;self.base.requires_grad_(False);self.rank=rank;self.alpha=alpha;self.scale=alpha/rank;self.enabled=True
  self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32));self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,x):
  y=self.base(x)
  if not self.enabled:return y
  return y+self.scale*F.linear(F.linear(x,self.A),self.B)

def insert_lora(model,rank=4,alpha=4):
 groups=[];adapters=[]
 for layer in model.model.layers:
  for expert in layer.mlp.experts:
   group=[]
   for name in ['gate_proj','up_proj','down_proj']:
    adapter=LoRALinear(getattr(expert,name),rank,alpha);setattr(expert,name,adapter);adapters.append(adapter);group.extend([adapter.A,adapter.B])
   groups.append(group)
 return groups,adapters

def set_enabled(adapters,enabled):
 for adapter in adapters:adapter.enabled=enabled

def frozen_hash(model):
 import hashlib
 h=hashlib.sha256();count=0
 for name,p in model.named_parameters():
  if p.requires_grad:continue
  h.update(name.encode());h.update(str((p.dtype,tuple(p.shape))).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes());count+=p.numel()
 return {'sha256':h.hexdigest(),'parameters':count}

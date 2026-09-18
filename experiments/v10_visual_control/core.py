"""Independent visual continual-learning environment; archived v1-v8 are untouched."""
import copy
import hashlib
import json
import pickle
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = torch.tensor([0., .25, .5, .75, 1., 1.5])

@dataclass
class Config:
    width: int = 48
    experts: int = 4
    layers: int = 2
    topk: int = 2
    tasks: int = 4
    batch: int = 32
    steps: int = 200
    interval: int = 10
    lr: float = .001
    decay: float = .01
    memory: int = 200
    target: float = 1.25
    routing: str = 'dynamic'
    threads: int = 1

class Data:
    """Training images only until explicitly requesting final test data."""
    def __init__(self, archive, final=False):
        archive = Path(archive)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != 'eb9058c3a382ffc7106e4002c42a8d85':
            raise ValueError('CIFAR-100 official checksum mismatch')
        self.parts = {}
        with tarfile.open(archive) as tf:
            for name in (['train', 'test'] if final else ['train']):
                obj = pickle.load(tf.extractfile('cifar-100-python/' + name), encoding='bytes')
                self.parts[name] = (torch.from_numpy(obj[b'data'].reshape(-1, 3, 32, 32)), np.array(obj[b'fine_labels']))
        self.md5 = digest

    def task(self, classes, role):
        name = 'test' if role == 'test' else 'train'
        x, labels = self.parts[name]
        ids, ys = [], []
        for y, c in enumerate(classes):
            idx = np.where(labels == c)[0]
            idx = np.random.default_rng(10000 + int(c)).permutation(idx)
            # 350 learner, 50 online, 100 development evaluation per class.
            idx = idx if role == 'test' else idx[{'train':slice(0,350),'monitor':slice(350,400),'eval':slice(400,500)}[role]]
            ids.extend(idx.tolist()); ys.extend([y]*len(idx))
        return x[ids], torch.tensor(ys), torch.tensor(ids)

def normalize(x):
    return (x.float()/255. - .5)/.25

class Learner(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        w = cfg.width
        self.front = nn.Sequential(nn.Conv2d(3,12,3,padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(12,24,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)), nn.Flatten(), nn.Linear(96,w))
        self.blocks = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(w,w*2),nn.ReLU(),nn.Linear(w*2,w)) for _ in range(cfg.experts)]) for _ in range(cfg.layers)])
        self.routers = nn.ModuleList([nn.Linear(w,cfg.experts,bias=False) for _ in range(cfg.layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(w) for _ in range(cfg.layers)])
        self.heads = nn.ModuleList([nn.Linear(w,5) for _ in range(cfg.tasks)])
        for p in self.routers.parameters(): p.requires_grad_(False)
        # Fixed routing variant depends on raw image, never on a trainable representation.
        self.register_buffer('fixed_projection', torch.randn(3*4*4,cfg.layers,cfg.experts)/7)

    def forward(self,x,task):
        raw=normalize(x); h=self.front(raw); routes=[]
        fixed=torch.einsum('bd,dle->ble', F.adaptive_avg_pool2d(raw,(4,4)).flatten(1),self.fixed_projection)
        for l,(block,router,norm) in enumerate(zip(self.blocks,self.routers,self.norms)):
            logits=router(h) if self.cfg.routing=='dynamic' else fixed[:,l]
            vals,idx=logits.topk(self.cfg.topk,dim=-1)
            gate=torch.zeros_like(logits).scatter(1,idx,vals.softmax(-1))
            update=torch.zeros_like(h)
            for e,expert in enumerate(block):
                active=gate[:,e]>0
                if active.any(): update[active] = update[active]+expert(h[active])*gate[active,e,None]
            h=norm(h+update); routes.append(gate)
        return self.heads[task](h),torch.stack(routes,1)

class GatedAdam:
    """Gate full AdamW displacement incl. decay. Moments advance for active grads.
    grad=None experts skip both moments and decay, matching PyTorch AdamW.
    Front/norm/current or replayed heads always use multiplier 1; routers frozen.
    """
    def __init__(self,model,cfg):
        self.model=model
        self.optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,weight_decay=cfg.decay)
        self.last=torch.zeros(cfg.layers,cfg.experts)
    def zero_grad(self): self.optimizer.zero_grad(set_to_none=True)
    def step(self, multipliers):
        saved=[[(p,p.detach().clone()) for p in ex.parameters()] for layer in self.model.blocks for ex in layer]
        self.optimizer.step()
        with torch.no_grad():
            for i,params in enumerate(saved):
                l,e=divmod(i,self.model.cfg.experts); ss=0.; n=0
                for p,old in params:
                    delta=p-old; ss+=delta.square().sum().item(); n+=p.numel()
                    p.copy_(old+multipliers[l,e]*delta)
                self.last[l,e]=(ss/max(n,1))**.5

class Environment:
    def __init__(self,data,cfg,seed,split='train',evaluation='eval'):
        self.cfg=cfg; self.seed=seed; self.data=data; self.evaluation=evaluation
        torch.manual_seed(seed)
        self.model=Learner(cfg); self.optim=GatedAdam(self.model,cfg)
        pool={'train':np.arange(60),'dev':np.arange(60,80),'test':np.arange(80,100)}[split]
        self.classes=np.random.default_rng(seed).permutation(pool)[:cfg.tasks*5].reshape(cfg.tasks,5)
        self.tasksets={}; self.memory={}; self.k=0; self.t=0; self.previous=torch.ones(cfg.layers,cfg.experts)
        self.matrix=[]; self.accuracy=[]; self.records=[]; self.baseline=[]; self.last_loss=None
        self.last_old=0.; self.historical=torch.zeros(cfg.layers,cfg.experts)
        self.cost={'updates':0,'train_examples':0,'observation_examples':0,'evaluation_examples':0,'copies':0,'copy_seconds':0.}
        self.start_loss=None
        self.last_seen=torch.full((cfg.layers,cfg.experts),-1.)

    def clone(self):
        start=time.perf_counter()
        memo={id(self.data):self.data}
        for data in self.tasksets.values():
            for tensor in data: memo[id(tensor)]=tensor
        other=copy.deepcopy(self,memo)
        other.cost['copies']+=1; other.cost['copy_seconds']+=time.perf_counter()-start
        return other

    def dataset(self,k,role):
        if k<self.k and role in ('train','monitor'):
            raise ValueError('Past training/monitor data are inaccessible outside retained memory')
        key=(k,role)
        if key not in self.tasksets: self.tasksets[key]=self.data.task(self.classes[k],role)
        return self.tasksets[key]

    @torch.no_grad()
    def measure(self,k,xy,cost='observation_examples'):
        x,y=xy[:2]; losses=[]; correct=0; routes=[]
        for i in range(0,len(x),128):
            out,g=self.model(x[i:i+128],k)
            losses.append(F.cross_entropy(out,y[i:i+128],reduction='none')); correct+=(out.argmax(1)==y[i:i+128]).sum().item(); routes.append(g)
        self.cost[cost]+=len(x)
        return torch.cat(losses), correct/len(x), torch.cat(routes)

    def observation(self):
        loss,_,g=self.measure(self.k,self.dataset(self.k,'monitor'))
        current=loss.mean().item()
        if self.start_loss is None: self.start_loss=current
        old_usage=torch.zeros_like(self.previous); old_mass=old_usage.clone(); old_cover=old_usage.clone(); damage=[]; n=0
        for j,m in self.memory.items():
            ll,_,gg=self.measure(j,m['monitor'])
            damage.append((ll-m['baseline']).mean().clamp_min(0).item())
            old_usage+=(gg>0).sum(0); old_mass+=gg.sum(0); old_cover+=(gg>0).sum(0); n+=len(gg)
        old_usage/=max(n,1); old_mass/=max(n,1)
        self.last_seen[old_cover>0]=self.cost['updates']
        age=torch.where(self.last_seen>=0,(self.cost['updates']-self.last_seen)/self.cfg.steps,torch.ones_like(self.last_seen))
        forgetting=float(np.mean(damage)) if damage else 0.
        improvement=0 if self.last_loss is None else self.last_loss-current
        glob=torch.tensor([current,self.cfg.target,current-self.cfg.target,improvement,self.t/self.cfg.steps,self.k/self.cfg.tasks,forgetting,forgetting-self.last_old,self.start_loss])
        loc=torch.stack([self.historical,old_usage,old_mass,(g>0).float().mean(0),g.mean(0),self.optim.last,self.previous,
            old_cover/max(self.cfg.memory,1),(old_cover==0).float(),age,
            torch.full_like(old_cover,len(g)/250), torch.arange(self.cfg.layers)[:,None].expand_as(old_cover)/max(self.cfg.layers-1,1)],-1)
        self.last_loss=current; self.last_old=forgetting
        return loc,glob

    def update(self,m):
        cfg=self.cfg
        for _ in range(min(cfg.interval,cfg.steps-self.t)):
            # Independent per-task, local-step streams prevent early stopping shifting future tasks.
            rng=np.random.default_rng(self.seed*1000003+self.k*10007+self.t)
            x,y,_=self.dataset(self.k,'train'); ix=rng.integers(len(x),size=cfg.batch)
            self.optim.zero_grad(); out,_=self.model(x[ix],self.k); loss=F.cross_entropy(out,y[ix])
            if self.memory:
                j=sorted(self.memory)[self.t%len(self.memory)]; xx,yy,_=self.memory[j]['replay']; ii=rng.integers(len(xx),size=cfg.batch//2)
                oo,_=self.model(xx[ii],j); loss=loss+.5*F.cross_entropy(oo,yy[ii]); self.cost['train_examples']+=len(ii)
            loss.backward(); self.optim.step(m); self.cost['updates']+=1; self.cost['train_examples']+=len(ix); self.t+=1
        self.previous=m.clone()

    def finish(self,loss):
        row=[]; acc=[]
        # Full matrix, including future tasks, isolated from observations/rewards.
        for j in range(self.cfg.tasks):
            ll,a,_=self.measure(j,self.dataset(j,self.evaluation),'evaluation_examples'); row.append(ll.mean().item()); acc.append(a)
        self.matrix.append(row); self.accuracy.append(acc)
        self.records.append({'task':self.k,'steps':self.t,'initial_loss':self.start_loss,'loss':loss,'success':loss<=self.cfg.target,'initial_success':self.t==0 and loss<=self.cfg.target})
        self.baseline.append(row[self.k])
        # Equal per-task budgets: retain prefixes only; never reacquire discarded old data.
        quota=self.cfg.memory//(2*(self.k+1))
        for j,m in self.memory.items():
            m['replay']=tuple(a[:quota] for a in m['replay']); m['monitor']=tuple(a[:quota] for a in m['monitor']); m['baseline']=m['baseline'][:quota]; m['route_at_completion']=m['route_at_completion'][:quota]
        rng=np.random.default_rng(self.seed+self.k*71)
        new={}
        for role,source in [('replay','train'),('monitor','monitor')]:
            d=self.dataset(self.k,source); idx=rng.permutation(len(d[0]))[:quota]; new[role]=tuple(a[idx].clone() for a in d)
        ll,_,g=self.measure(self.k,new['monitor']); new['baseline']=ll.clone(); new['route_at_completion']=(g>0).float()
        self.memory[self.k]=new
        self.historical=torch.cat([m['route_at_completion'] for m in self.memory.values()]).mean(0)
        for key in list(self.tasksets):
            if key[0]==self.k and key[1] in ('train','monitor'): del self.tasksets[key]
        self.k+=1; self.t=0; self.last_loss=None; self.start_loss=None; self.last_old=0.

    def result(self):
        mat=np.array(self.matrix); base=np.diag(mat); final=mat[-1,:-1]-base[:-1]
        segment=[None]+[float(np.mean(mat[k,:k]-mat[k-1,:k])) for k in range(1,len(mat))]
        return {'seed':self.seed,'classes':self.classes.tolist(),'tasks':self.records,'matrix':self.matrix,'accuracy_matrix':self.accuracy,
            'success':all(r['success'] for r in self.records),'forgetting':float(np.maximum(final,0).mean()),'signed_forgetting':float(final.mean()),'backward_improvement':float(np.maximum(-final,0).mean()),'segment_damage':segment,
            'final_loss':float(mat[-1].mean()),'final_accuracy':float(np.mean(self.accuracy[-1])),'cost':self.cost}

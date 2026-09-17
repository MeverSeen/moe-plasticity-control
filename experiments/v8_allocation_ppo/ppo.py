"""On-policy discrete PPO with undiscounted retention and failure critics."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import control as l
import sequence_env as e

class ActorCritic(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(18,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh())
        self.actor = nn.Linear(32, actions)
        self.value = nn.Linear(32,1)
        self.risk = nn.Linear(32,1)
    def forward(self,x):
        z=self.body(x)
        return self.actor(z),self.value(z).squeeze(-1),self.risk(z).squeeze(-1)

def chooser(model, ckpt):
    def action(env):
        with torch.no_grad():
            logits,_,_=model(l.features(env.observe(),ckpt['mean'],ckpt['std'],ckpt['feedback']))
        return int(logits.argmax())
    return action

def episode(model, ckpt, item, config):
    env=e.SequenceEnv(item,config); rows=[]
    while not env.done:
        obs=env.observe()
        with torch.no_grad():
            logits,v,c=model(l.features(obs,ckpt['mean'],ckpt['std'],ckpt['feedback']))
            dist=Categorical(logits=logits); action=dist.sample()
        before=env.forgetting
        env.act(int(action),'within')
        rows.append(dict(x=obs.tolist(),action=int(action),logp=float(dist.log_prob(action)),
                         value=float(v),risk=float(c),reward=-(env.forgetting-before)/e.F_SCALE))
    failure=float(not all(r['success'] for r in env.completed))
    ret=0.
    for row in reversed(rows):
        ret+=row['reward']; row['return']=ret; row['cost_return']=failure
    if rows: rows[-1]['terminal_cost']=failure
    return rows,dict(root=item[0]['seed'],failure=failure,forgetting=env.forgetting,
                     steps=env.steps,work=env.work,segments=env.completed)

def update(model,opt,rows,ckpt,lagrange):
    x=l.features([r['x'] for r in rows],ckpt['mean'],ckpt['std'],ckpt['feedback'])
    a=torch.tensor([r['action'] for r in rows]); old=torch.tensor([r['logp'] for r in rows])
    ret=torch.tensor([r['return'] for r in rows]); cost=torch.tensor([r['cost_return'] for r in rows])
    adv=torch.tensor([r['return']-r['value']-lagrange*(r['cost_return']-r['risk']) for r in rows])
    adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    before=torch.cat([p.detach().flatten().clone() for p in model.actor.parameters()])
    for _ in range(4):
        logits,v,c=model(x); dist=Categorical(logits=logits)
        ratio=(dist.log_prob(a)-old).exp()
        policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        loss=policy+.5*((v-ret).square().mean()+(c-cost).square().mean())-.01*dist.entropy().mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    after=torch.cat([p.detach().flatten() for p in model.actor.parameters()])
    return dict(loss=float(loss.detach()),actor_parameter_delta=float((after-before).norm()))

def load(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=ActorCritic(ckpt['actions']); model.load_state_dict(ckpt['state']); model.eval()
    return model,ckpt

def fit(out,scorer_path,states,train,dev,config,seed,batches,pilot):
    out.mkdir(parents=True)
    scorer,teacher=l.load(scorer_path)
    torch.manual_seed(seed)
    model=ActorCritic(teacher['actions'])
    ckpt={k:teacher[k] for k in ['mean','std','feedback','seed','actions']}
    candidates=[]; curve=[]; best=None; selected=None; cost=dict(dev_rollouts=0,dev_updates=0,dev_work=0.,train_episodes=0,train_updates=0,train_work=0.)
    def evaluate(label):
        nonlocal best,selected
        snap=dict(ckpt,state=copy.deepcopy(model.state_dict()),checkpoint=label)
        torch.save(snap,out/f'{label}.pt')
        rows=[e.rollout(item,config,chooser(model,snap),'within') for item in dev]
        metrics=l.metrics(rows); candidates.append(dict(checkpoint=label,metrics=metrics,rows=rows))
        cost['dev_rollouts']+=len(rows); cost['dev_updates']+=sum(r['steps'] for r in rows); cost['dev_work']+=sum(r['work'] for r in rows)
        if label!='random0' and (best is None or l.selection_key(metrics)<best):
            best=l.selection_key(metrics); selected=dict(snap,dev=metrics)
    evaluate('random0')
    x=l.features([s['x'] for s in states],ckpt['mean'],ckpt['std'],ckpt['feedback'])
    with torch.no_grad():
        h,f=scorer(x); targets=torch.tensor(l.choose(h.sigmoid().numpy(),f.numpy(),teacher['slack']))
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    for _ in range(50 if pilot else 200):
        logits,_,_=model(x); loss=nn.functional.cross_entropy(logits,targets)
        opt.zero_grad(); loss.backward(); opt.step()
    evaluate('warm0')
    opt=torch.optim.Adam(model.parameters(),lr=.0003)
    rng=np.random.default_rng(seed); lagrange=1.
    with (out/'trajectories.jsonl').open('w') as archive:
        for batch in range(1,batches+1):
            transitions=[]; summaries=[]
            for idx in rng.choice(len(train),8,replace=len(train)<8):
                trajectory,summary=episode(model,ckpt,train[idx],config)
                archive.write(json.dumps(dict(batch=batch,summary=summary,trajectory=trajectory))+'\n')
                transitions.extend(trajectory); summaries.append(summary)
            result=update(model,opt,transitions,ckpt,lagrange) if transitions else dict(actor_parameter_delta=0.,loss=0.)
            failure=float(np.mean([r['failure'] for r in summaries])); lagrange=float(np.clip(lagrange+.5*(failure-.1),0,20))
            cost['train_episodes']+=len(summaries); cost['train_updates']+=sum(r['steps'] for r in summaries); cost['train_work']+=sum(r['work'] for r in summaries)
            curve.append(dict(batch=batch,lagrange=lagrange,failure=failure,**result))
            if pilot or batch%8==0 or batch==batches: evaluate(f'ppo{batch}')
            print(json.dumps(dict(stage='ppo',seed=seed,actions=ckpt['actions'],feedback=ckpt['feedback'],**curve[-1])),flush=True)
    torch.save(selected,out/'selected.pt')
    l.save(out/'selection.json',{k:v for k,v in selected.items() if k not in ['state','mean','std']})
    l.save(out/'candidates.json',candidates); l.save(out/'training.json',curve); l.save(out/'cost.json',dict(cost,distillation_epochs=50 if pilot else 200,ppo_gradient_steps=4*batches))

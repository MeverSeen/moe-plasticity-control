"""Full-state counterfactual branches and supervised, scalar update control.

The learner sees only training/replay and bounded monitoring memory. Independent
evaluation examples provide OFFLINE meta-training labels on training histories;
held-out-history evaluation labels never enter fitting, features or decisions.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import base_env as base

torch.set_num_threads(1)
ACTIONS = base.ACTIONS
FEATURES = base.FEATURE_NAMES
POLICY_MACS = 14*32 + 32*32 + 32*10
F_SCALE = .01
CHECKPOINT_STEPS = [0, 10, 30, 60]
POLICY_SEEDS = [271, 272, 273]
SPLITS = {'calibration': list(range(310000,310016)),
          'train': list(range(320000,320080)),
          'dev': list(range(330000,330024)),
          'test': list(range(340000,340064))}


def save(path, obj):
    base.save(Path(path), obj)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_env(item, config):
    # Identical reserved scorer cost for all arms: an information ablation,
    # NOT a claim of equal actual wall time or deployment efficiency.
    return base.ScalarEnv(item, config, POLICY_MACS)


def clone_env(env):
    """Copy all mutable training state; share only read-only datasets/router."""
    result = copy.copy(env)
    shared = {'world', 'router', 'new', 'new_monitor', 'old', 'monitors', 'c'}
    for key, value in vars(env).items():
        if key in shared:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().clone().requires_grad_(value.requires_grad)
        elif isinstance(value, np.ndarray):
            value = value.copy()
        else:
            value = copy.deepcopy(value)
        setattr(result, key, value)
    return result


def resample_future(env, replicate):
    """Independent future, shared by every action, preserving the entire past."""
    result = clone_env(env)
    rng = base.core.gen(env.world['seed']*100003 + env.steps*97 + replicate*997 + 8001)
    n = env.c['max_steps']-env.steps
    result.new_indices[env.steps:] = torch.randint(len(env.new[0]),
        (n,env.c['batch']),generator=rng)
    result.old_indices[env.steps:] = torch.randint(len(env.old[0]),
        (n,env.c['old_batch']),generator=rng)
    return result


def evaluator_reference(env):
    # Called by the offline evaluator only, never by state/policy/updates.
    return np.array([base.core.mse(env.initial_w,env.world['tasks'][i]['test'],
        env.router,env.c) for i in range(env.task)])


def endpoint(env, reference):
    after = np.array([base.core.mse(env.w,env.world['tasks'][i]['test'],
        env.router,env.c) for i in range(env.task)])
    forgetting = float(np.maximum(after-reference,0).mean())
    return dict(success=bool(env.success),steps=int(env.steps),work=float(env.work),
        budget=float(env.budget),forgetting=forgetting,
        signed_old_change=float((after-reference).mean()),old_eval_loss=float(after.mean()),
        monitor_forgetting=float(env.forgetting),new_monitor_loss=float(env.current_loss),
        new_eval_loss=base.core.mse(env.w,env.world['tasks'][env.task]['test'],env.router,env.c),
        target=float(env.c['target']),reason=env.reason,
        cost=objective(float(env.success),forgetting))


def objective(hit, forgetting):
    # Diagnostic bounded endpoint cost; every failure costs 2, every success <1.
    # Aggregate gains in this scalar NEVER override reporting reachability.
    return float(2*(1-hit)+hit*forgetting/(F_SCALE+forgetting))


def finish(env, continuation):
    while not env.done:
        env.advance(continuation)
    return env


def rollout(item, config, chooser):
    env = new_env(item, config)
    ref = evaluator_reference(env)
    while not env.done:
        env.advance(int(chooser(env)))
    row = endpoint(env,ref)
    row.update(root=int(env.world['seed']),protected_mass=env.protected_mass,
        protected_fraction=float(env.mask.float().mean()),
        decisions=len(env.events),
        mean_multiplier=float(np.mean([x['multiplier'] for x in env.events])) if env.events else 1.,
        switches=sum(a['multiplier'] != b['multiplier'] for a,b in zip(env.events,env.events[1:])),
        initial_new_loss=env.initial_loss,events=env.events)
    return row


def weighted_mean(values, roots):
    values,roots = np.asarray(values),np.asarray(roots)
    return float(np.mean([values[roots==r].mean() for r in np.unique(roots)]))


def rows_key(rows):
    return (-sum(r['success'] for r in rows),
        float(np.mean([r['cost'] for r in rows])),float(np.mean([r['work'] for r in rows])))


def make_bank(seeds, config):
    return base.make_bank(seeds,config,test=True)


def calibrate(out, config, seeds):
    bank = make_bank(seeds,config)
    candidates = []
    for a in range(len(ACTIONS)):
        rows = [rollout(item,config,lambda env,a=a:a) for item in bank]
        candidates.append(dict(action=a,multiplier=ACTIONS[a],rows=rows,key=rows_key(rows)))
    chosen = min(candidates,key=lambda x:x['key'])['action']
    save(out/'continuation_calibration.json',dict(candidates=candidates,selected_action=chosen))
    return chosen


def snapshots(item, config):
    """Predeclared, outcome-independent state sampling from two behavior paths."""
    root = item[0]['seed']
    for path in ['constant_half','random_prefix']:
        env = new_env(item,config)
        rng = np.random.default_rng(root*131+19)
        if path == 'constant_half' and not env.done:
            yield path,clone_env(env)
        while not env.done and env.steps < max(CHECKPOINT_STEPS):
            action = 2 if path == 'constant_half' else int(rng.integers(5))
            env.advance(action)
            if env.steps in CHECKPOINT_STEPS and not env.done:
                yield path,clone_env(env)


def collect(out, split, seeds, config, continuation, replicates=2):
    path = out/f'{split}_branches.jsonl'
    if path.exists():
        raise RuntimeError('Refusing to overwrite branch data: '+str(path))
    started = time.time()
    states,root_records = [],[]
    total_updates,total_work = 0,0.
    with path.open('w') as stream:
        for ri,seed in enumerate(seeds):
            item = make_bank([seed],config)[0]
            initial = new_env(item,config)
            ref = evaluator_reference(initial)
            count = 0
            for behavior,env in snapshots(item,config):
                branches = [[] for _ in ACTIONS]
                for rep in range(replicates):
                    fresh = resample_future(env,rep)
                    for a in range(len(ACTIONS)):
                        b = clone_env(fresh)
                        b.advance(a)
                        local = dict(new_gain=env.current_loss-b.current_loss,
                            old_monitor_change=float(b.old_losses.mean()-env.old_losses.mean()),
                            steps=b.steps-env.steps)
                        finish(b,continuation)
                        result = endpoint(b,ref)
                        result.update(replicate=rep,local=local)
                        branches[a].append(result)
                        total_updates += b.steps-env.steps
                        total_work += b.work-env.work
                record = dict(root=seed,old_tasks=env.task,path=behavior,step=env.steps,
                    state_id=f'{seed}:{behavior}:{env.steps}',
                    x=env.observe(True).tolist(),mask=env.mask.tolist(),
                    protected_mass=env.protected_mass,branches=branches)
                states.append(record)
                stream.write(json.dumps(record)+'\n')
                count += 1
            root_records.append(dict(root=seed,states=count,initially_done=initial.done,
                initial_new_loss=initial.initial_loss,protected_mass=initial.protected_mass,
                mask=initial.mask.tolist(),old_tasks=initial.task))
            stream.flush()
            if (ri+1)%8==0 or ri==len(seeds)-1:
                print(json.dumps(dict(stage='branch',split=split,histories=ri+1,
                    states=len(states),seconds=round(time.time()-started,1))),flush=True)
    save(out/f'{split}_cost.json',dict(seconds=time.time()-started,states=len(states),
        histories=len(seeds),future_replicates=replicates,branch_updates=total_updates,
        branch_work=total_work,roots=root_records,
        note='Offline branching work excludes common old-task pretraining and state-generation work.'))
    return states


def load_states(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def arrays(states):
    x = np.array([s['x'] for s in states],dtype=np.float32)
    roots = np.array([s['root'] for s in states])
    hit = np.array([[[float(r['success']) for r in a] for a in s['branches']]
        for s in states],dtype=np.float32)
    forget = np.array([[[r['forgetting'] for r in a] for a in s['branches']]
        for s in states],dtype=np.float32)
    costs = 2*(1-hit)+hit*forget/(F_SCALE+forget)
    return x,roots,hit,forget,costs


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(14,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.hit = nn.Linear(32,5)
        self.forget = nn.Linear(32,5)

    def forward(self,x):
        h = self.trunk(x)
        return self.hit(h),self.forget(h)


def features(x, mean, std, feedback):
    result = (np.asarray(x,dtype=np.float32)-mean)/std
    if not feedback:
        result[...,12:] = 0
    return torch.tensor(result,dtype=torch.float32)


def choose(pred_hit, pred_forget, slack):
    """Only predictions determine feasibility; never use actual branch labels."""
    eligible = pred_hit >= pred_hit.max(1,keepdims=True)-slack
    return np.where(eligible,pred_forget,np.inf).argmin(1)


def predict_actions(model, x, mean, std, feedback, slack):
    with torch.no_grad():
        ph,pf = model(features(x,mean,std,feedback))
        ph,pf = ph.sigmoid().numpy(),pf.numpy()
    return choose(ph,pf,slack)


def selection_metrics(states, selected):
    _,roots,hit,forget,cost = arrays(states)
    idx = np.arange(len(states))
    a_hit,a_cost = hit[idx,selected].mean(1),cost[idx,selected].mean(1)
    return dict(reach=weighted_mean(a_hit,roots),cost=weighted_mean(a_cost,roots),
        endpoint_forgetting=weighted_mean(forget[idx,selected].mean(1),roots))


def train_scorers(out, train, dev, epochs=300, seeds=POLICY_SEEDS):
    x,roots,hit,forget,_ = arrays(train)
    mean,std = x.mean(0),np.maximum(x.std(0),.05)
    th = torch.tensor(hit.mean(2))
    valid = hit.sum(2)>0
    target_f = (np.log1p(forget/F_SCALE)*hit).sum(2)/np.maximum(hit.sum(2),1)
    tf,tv = torch.tensor(target_f),torch.tensor(valid,dtype=torch.float32)
    # Equal total weight for each training history, not each checkpoint.
    weights = np.array([1/np.sum(roots==r) for r in roots],dtype=np.float32)
    weights /= weights.mean()
    tw = torch.tensor(weights)
    dx = np.array([s['x'] for s in dev],dtype=np.float32)
    manifest = {}
    for seed in seeds:
        for feedback in [False,True]:
            mode = 'feedback' if feedback else 'progress'
            name = f'{mode}_{seed}'
            torch.manual_seed(seed)
            model = Scorer()
            opt = torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
            tx = features(x,mean,std,feedback)
            best,best_data,history = None,None,[]
            for epoch in range(epochs+1):
                if epoch%25==0 or epoch==epochs:
                    model.eval()
                    for slack in [0.,.02,.1,1.]:
                        selected = predict_actions(model,dx,mean,std,feedback,slack)
                        score = selection_metrics(dev,selected)
                        key = (-round(score['reach'],9),score['cost'])
                        if best is None or key<best:
                            best=key
                            best_data=dict(state=copy.deepcopy(model.state_dict()),mean=mean,std=std,
                                feedback=feedback,slack=slack,seed=seed,epoch=epoch,dev=score)
                    history.append(dict(epoch=epoch,best_dev=best_data['dev']))
                if epoch==epochs:
                    break
                model.train()
                ph,pf=model(tx)
                lhit=F.binary_cross_entropy_with_logits(ph,th,reduction='none').mean(1)
                lf=(F.smooth_l1_loss(pf,tf,reduction='none')*tv).sum(1)/tv.sum(1).clamp_min(1)
                pc=pf-(pf*tv).sum(1,keepdim=True)/tv.sum(1,keepdim=True).clamp_min(1)
                tc=tf-(tf*tv).sum(1,keepdim=True)/tv.sum(1,keepdim=True).clamp_min(1)
                lr=(F.smooth_l1_loss(pc,tc,reduction='none')*tv).sum(1)/tv.sum(1).clamp_min(1)
                objective=((lhit+lf+lr)*tw).mean()
                opt.zero_grad();objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
            torch.save(best_data,out/f'{name}.pt')
            manifest[name]={k:best_data[k] for k in ['feedback','slack','seed','epoch','dev']}
            save(out/f'{name}_training.json',history)
            print(json.dumps(dict(stage='fit',policy=name,**manifest[name])),flush=True)
    save(out/'scorers.json',manifest)
    return manifest


def load_scorer(path):
    ckpt=torch.load(path,weights_only=False)
    model=Scorer();model.load_state_dict(ckpt['state']);model.eval()
    return model,ckpt


def scorer_chooser(model,ckpt):
    def decision(env):
        return int(predict_actions(model,np.array([env.observe(True).tolist()]),
            ckpt['mean'],ckpt['std'],ckpt['feedback'],ckpt['slack'])[0])
    return decision


def tune_baselines(out,config,seeds):
    bank=make_bank(seeds,config)
    grids={'static':[dict(action=a) for a in range(5)],
           'hand':[dict(action=a,tolerance=t,release=r)
               for a in [0,1,2,3] for t in [.001,.01,.03] for r in [.0,.25]]}
    selected,candidates={},[]
    for kind,grid in grids.items():
        best=None
        for cfg in grid:
            rows=[rollout(item,config,baseline_chooser(kind,cfg)) for item in bank]
            key=rows_key(rows)
            candidates.append(dict(kind=kind,config=cfg,key=key,
                rows=[{k:v for k,v in r.items() if k!='events'} for r in rows]))
            if best is None or key<best[0]:best=(key,cfg)
        selected[kind]=best[1]
        print(json.dumps(dict(stage='tune_baseline',kind=kind,selected=best[1],key=best[0])),flush=True)
    save(out/'baseline_candidates.json',candidates)
    save(out/'baselines.json',selected)
    return selected


def baseline_chooser(kind,cfg):
    if kind=='static':return lambda env:cfg['action']
    def hand(env):
        if (env.budget-env.work)/env.budget<cfg['release']:return 4
        return cfg['action'] if env.forgetting>cfg['tolerance'] else 4
    return hand


def offline_predictions(out,states,manifest):
    predictions={}
    x=np.array([s['x'] for s in states],dtype=np.float32)
    for name in manifest:
        model,ckpt=load_scorer(out/f'{name}.pt')
        predictions[name]=predict_actions(model,x,ckpt['mean'],ckpt['std'],
            ckpt['feedback'],ckpt['slack']).tolist()
    save(out/'test_predictions.json',dict(state_ids=[s['state_id'] for s in states],
        actions=predictions))


def online_test(out,config,seeds,manifest,baselines):
    policies={name:scorer_chooser(*load_scorer(out/f'{name}.pt')) for name in manifest}
    policies.update(best_static=baseline_chooser('static',baselines['static']),
        hand_feedback=baseline_chooser('hand',baselines['hand']),full_update=lambda env:4)
    rows=[];started=time.time()
    for i,seed in enumerate(seeds):
        item=make_bank([seed],config)[0]
        for name,chooser in policies.items():
            rows.append(dict(method=name,**rollout(item,config,chooser)))
        if (i+1)%8==0 or i==len(seeds)-1:
            save(out/'test_online.json',rows)
            print(json.dumps(dict(stage='closed_loop',histories=i+1,seconds=round(time.time()-started,1))),flush=True)
    save(out/'online_cost.json',dict(seconds=time.time()-started,
        charged_work=sum(r['work'] for r in rows),model_updates=sum(r['steps'] for r in rows)))


def protocol(config,splits,replicates,epochs):
    return dict(config=config,splits=splits,actions=ACTIONS,checkpoint_steps=CHECKPOINT_STEPS,
        paths=['constant_half','random_prefix'],future_replicates=replicates,epochs=epochs,
        feature_names=FEATURES,feedback_indices=[12,13],policy_seeds=POLICY_SEEDS,
        primary='Held-out-history feedback vs progress scoring, paired by history and policy initialization.',
        objective='Reachability first. Also report bounded diagnostic cost J=2*(1-hit)+hit*F/(F+.01).',
        labels='Independent evaluation samples provide offline labels; train/dev/test worlds are disjoint.',
        endpoint='First new-task monitoring MSE <= .06 or common work/300-step budget exhausted.',
        continuation='Calibrate one fixed multiplier on calibration worlds, then lock for all branches.',
        clone='Weights, Adam moments/time, observation history, budget and indexed future data stream.',
        optimizer='Adam proposal on new + .1 replay gradient; group multiplier on displacement. No decay.',
        frozen_state='Raw-input geometric router and initial protection mask are fixed. No learned router.',
        budget='Common forward-example workload proxy; every arm charged identical monitoring/scorer allowance.',
        statistics='History-clustered paired bootstrap; policy replicas averaged within history.',
        limitations=['Synthetic nonlinear MoE; no learned router or shared trainable backbone.',
            'Branch value depends on one fixed continuation and sampled behavior states.',
            'Two future replicas are a noise diagnostic, not a precise population action optimum.',
            'Within-B adaptation after 1--3 old tasks, not controller-driven multi-task history.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--replicates',type=int,default=2)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Use a fresh output directory; no overwriting previous experiments.')
    args.out.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(base.CONFIG)
    splits=copy.deepcopy(SPLITS)
    if args.pilot:
        splits={k:list(range(350000+i*100,350000+i*100+n))
            for i,(k,n) in enumerate([('calibration',4),('train',8),('dev',4),('test',4)])}
    plan=protocol(config,splits,args.replicates,args.epochs)
    plan['pilot']=args.pilot
    save(args.out/'plan.json',plan)
    save(args.out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1))
    started=time.time()
    continuation=calibrate(args.out,config,splits['calibration'])
    print(json.dumps(dict(stage='continuation_locked',action=continuation,multiplier=ACTIONS[continuation])),flush=True)
    train=collect(args.out,'train',splits['train'],config,continuation,args.replicates)
    dev=collect(args.out,'dev',splits['dev'],config,continuation,args.replicates)
    manifest=train_scorers(args.out,train,dev,args.epochs)
    baselines=tune_baselines(args.out,config,splits['dev'])
    # Offline static intervention comparator selected only on dev branch outcomes.
    static_scores=[selection_metrics(dev,np.full(len(dev),a,dtype=int)) for a in range(5)]
    static_branch=min(range(5),key=lambda a:(-static_scores[a]['reach'],static_scores[a]['cost']))
    lock=dict(plan,continuation_action=continuation,static_branch_action=static_branch,
        static_branch_dev=static_scores,baselines=baselines,scorers=manifest,
        source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        policy_hashes={name:file_hash(args.out/f'{name}.pt') for name in manifest})
    save(args.out/'locked_protocol.json',lock)
    # No held-out histories or evaluation labels have been generated before lock.
    test=collect(args.out,'test',splits['test'],config,continuation,args.replicates)
    offline_predictions(args.out,test,manifest)
    online_test(args.out,config,splits['test'],manifest,baselines)
    save(args.out/'total_cost.json',dict(seconds=time.time()-started,
        note='End-to-end elapsed time includes preparation, branching, fitting, tuning and final evaluation.'))
    print(json.dumps(dict(stage='complete',out=str(args.out),seconds=round(time.time()-started,1))),flush=True)


if __name__=='__main__':
    main()

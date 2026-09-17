"""Matched cost-sensitive supervised controllers and closed-loop selection."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import sequence_env as e


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(out, bank, config, frequency, replicates):
    """Same state/label data for both information arms at a given frequency.

    Counterfactual action lasts one allowed decision interval, followed by m=0
    through the sequence. This is a fixed-continuation teacher, not an optimal
    long-horizon oracle. Evaluator partitions supply offline training targets.
    """
    records, cost = [], dict(branches=0, future_updates=0, future_work=0.)
    for k, item in enumerate(bank):
        for behavior in ['middle', 'random']:
            env = e.SequenceEnv(item, config)
            rng = np.random.default_rng(item[0]['seed']+91)
            while not env.done:
                if env.local_steps == 0 or (frequency == 'within' and env.local_steps in [10, 30]):
                    branches = [[] for _ in e.ACTIONS]
                    for a in range(len(e.ACTIONS)):
                        for rep in range(1, replicates+1):
                            branch = e.clone_env(env)
                            branch.set_future_rep(rep)
                            before_steps, before_work = branch.steps, branch.work
                            branch.act(a, frequency)
                            while not branch.done:
                                branch.act(0, frequency)
                            row = e.endpoint(branch, False)
                            branches[a].append({name: row[name] for name in
                                ['success', 'forgetting', 'steps', 'work', 'segment_hits']})
                            cost['branches'] += 1
                            cost['future_updates'] += branch.steps-before_steps
                            cost['future_work'] += branch.work-before_work
                    records.append(dict(root=item[0]['seed'], behavior=behavior,
                        state_id=f'{item[0]["seed"]}:{behavior}:{env.stage}:{env.local_steps}',
                        stage=env.stage, local_steps=env.local_steps, global_steps=env.steps,
                        x=env.observe().tolist(), branches=branches))
                action = 2 if behavior == 'middle' else int(rng.integers(5))
                env.act(action, frequency)
        if (k+1) % 4 == 0 or k+1 == len(bank):
            print(json.dumps(dict(stage='collect', frequency=frequency, roots=k+1, states=len(records))), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out/f'{frequency}_branches.jsonl').open('w') as f:
        for record in records:
            f.write(json.dumps(record)+'\n')
    save(out/f'{frequency}_cost.json', cost)
    return records


class Scorer(nn.Module):
    def __init__(self, actions=5):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(len(e.FEATURES), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.hit, self.forget = nn.Linear(32, actions), nn.Linear(32, actions)

    def forward(self, x):
        z = self.body(x)
        return self.hit(z), self.forget(z)


def features(x, mean, std, feedback):
    x = (np.asarray(x, np.float32)-mean)/std
    if not feedback:
        x = x.copy()
        x[..., -2:] = 0.
    return torch.tensor(x, dtype=torch.float32)


def prepare(states):
    x = np.array([s['x'] for s in states], dtype=np.float32)
    roots = np.array([s['root'] for s in states])
    hit = np.array([[[r['success'] for r in a] for a in s['branches']] for s in states], np.float32)
    forgetting = np.array([[[r['forgetting'] for r in a] for a in s['branches']] for s in states], np.float32)
    weights = np.array([1/(roots==r).sum() for r in roots], np.float32)
    weights /= weights.mean()
    valid = (hit.sum(2) > 0).astype(np.float32)
    target = (hit*forgetting/e.F_SCALE).sum(2)/np.maximum(hit.sum(2), 1)
    return dict(x=x, mean=x.mean(0), std=np.maximum(x.std(0), .05),
                hit=torch.tensor(hit.mean(2)), valid=torch.tensor(valid),
                target=torch.tensor(target), weights=torch.tensor(weights))


def objective(model, tx, data):
    ph, pf = model(tx)
    valid, target = data['valid'], data['target']
    count = valid.sum(1).clamp_min(1)
    pc = pf-(pf*valid).sum(1, keepdim=True)/count[:, None]
    tc = target-(target*valid).sum(1, keepdim=True)/count[:, None]
    hit = F.binary_cross_entropy_with_logits(ph, data['hit'], reduction='none').mean(1)
    raw = (((pf-target).square()+(pc-tc).square())*valid).sum(1)/count
    return ((hit+raw)*data['weights']).mean()


def choose(ph, pf, slack):
    eligible = ph >= ph.max(axis=-1, keepdims=True)-slack
    return np.where(eligible, pf, np.inf).argmin(-1)


def chooser(model, ckpt):
    def action(env):
        with torch.no_grad():
            hit, forgetting = model(features(env.observe(), ckpt['mean'], ckpt['std'], ckpt['feedback']))
        return int(choose(hit.sigmoid().numpy(), forgetting.numpy(), ckpt['slack']))
    return action


def metrics(rows):
    okay = [r for r in rows if r['success']]
    return dict(reach=sum(r['success'] for r in rows)/len(rows),
        forgetting=float(np.mean([r['forgetting'] for r in okay])) if okay else None,
        work=float(np.mean([r['work'] for r in rows])),
        steps=float(np.mean([r['steps'] for r in rows])))


def selection_key(value):
    return (-round(value['reach'], 9), value['forgetting'] if value['forgetting'] is not None else float('inf'), value['work'])


def fit(out, states, dev_bank, config, frequency, feedback, seed, checkpoints, slacks, actions=5):
    out.mkdir(parents=True, exist_ok=False)
    data = prepare(states)
    torch.manual_seed(seed)
    model = Scorer(actions)
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
    tx = features(data['x'], data['mean'], data['std'], feedback)
    candidates, curve, best, chosen = [], [], None, None
    work, updates = 0., 0
    for epoch in range(max(checkpoints)+1):
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                value = float(objective(model, tx, data))
            curve.append(dict(epoch=epoch, training_loss=value))
            snapshot = dict(state=copy.deepcopy(model.state_dict()), mean=data['mean'], std=data['std'],
                            feedback=feedback, seed=seed, epoch=epoch, frequency=frequency, actions=actions)
            torch.save(dict(snapshot, slack=.1), out/f'epoch{epoch}.pt')
            for slack in slacks:
                ckpt = dict(snapshot, slack=slack)
                rows = [e.rollout(item, config, chooser(model, ckpt), frequency) for item in dev_bank]
                measure = metrics(rows)
                candidates.append(dict(epoch=epoch, slack=slack, metrics=measure, rows=rows))
                work += sum(r['work'] for r in rows)
                updates += sum(r['steps'] for r in rows)
                key = selection_key(measure)
                if best is None or key < best:
                    best, chosen = key, dict(ckpt, dev=measure)
            print(json.dumps(dict(stage='fit', frequency=frequency, feedback=feedback,
                                  seed=seed, epoch=epoch, loss=value)), flush=True)
        if epoch == max(checkpoints):
            break
        model.train()
        loss = objective(model, tx, data)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
    torch.save(chosen, out/'selected.pt')
    save(out/'selection.json', {k: v for k, v in chosen.items() if k not in ['state', 'mean', 'std']})
    save(out/'training.json', curve)
    save(out/'candidates.json', candidates)
    save(out/'cost.json', dict(dev_rollouts=len(candidates)*len(dev_bank), work=work, updates=updates))


def load(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = Scorer(ckpt.get('actions', 5)); model.load_state_dict(ckpt['state']); model.eval()
    return model, ckpt


def baseline_plans():
    plans = [dict(name=f'fixed_{a}', kind='fixed', action=a) for a in range(5)]
    for early, late in [(0, 4), (4, 0)]:
        for stage in [1, 2]:
            plans.append(dict(name=f'block_{early}_{late}_{stage}', kind='block', early=early, late=late, threshold=stage))
        for fraction in [.3, .6]:
            plans.append(dict(name=f'time_{early}_{late}_{fraction}', kind='time', early=early, late=late, threshold=fraction))
    for tolerance in [.001, .01, .03]:
        for safe in [0, 2]:
            plans.append(dict(name=f'hand_{safe}_{tolerance}', kind='hand', safe=safe, tolerance=tolerance))
    return plans


def plan_chooser(plan):
    def action(env):
        if plan['kind'] == 'fixed': return plan['action']
        if plan['kind'] == 'block': return plan['early'] if env.stage < plan['threshold'] else plan['late']
        if plan['kind'] == 'time':
            return plan['early'] if env.local_steps/env.c['segment_steps'] < plan['threshold'] else plan['late']
        if plan['kind'] == 'hand':
            if env.local_steps >= .8*env.c['segment_steps']: return 4
            return plan['safe'] if env.forgetting >= plan['tolerance'] else 4
        raise ValueError(plan)
    return action

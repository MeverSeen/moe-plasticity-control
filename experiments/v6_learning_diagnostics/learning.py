"""Matched scorer training with explicit loss and selection ablations."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

import branching as b

LOSSES = ['legacy', 'raw_mse']
SELECTORS = ['branch_J', 'branch_F', 'closed_F']
SLACKS = [0., .02, .1, 1.]
SEEDS = [681, 682, 683]


def prepare(states, loss_kind):
    x, roots, hit, forgetting, _ = b.arrays(states)
    mean, std = x.mean(0), np.maximum(x.std(0), .05)
    valid = hit.sum(2) > 0
    transformed = (np.log1p(forgetting / b.F_SCALE) if loss_kind == 'legacy'
                   else forgetting / b.F_SCALE)
    target = (transformed * hit).sum(2) / np.maximum(hit.sum(2), 1)
    weights = np.array([1 / np.sum(roots == r) for r in roots], dtype=np.float32)
    weights /= weights.mean()
    return dict(x=x, roots=roots, mean=mean, std=std,
                hit=torch.tensor(hit.mean(2)), target=torch.tensor(target),
                valid=torch.tensor(valid, dtype=torch.float32), weights=torch.tensor(weights))


def objective(model, tx, data, kind):
    ph, pf = model(tx)
    valid, target = data['valid'], data['target']
    denom = valid.sum(1).clamp_min(1)
    hit = F.binary_cross_entropy_with_logits(ph, data['hit'], reduction='none').mean(1)
    predicted_centered = pf - (pf * valid).sum(1, keepdim=True) / denom[:, None]
    target_centered = target - (target * valid).sum(1, keepdim=True) / denom[:, None]
    if kind == 'legacy':
        absolute = F.smooth_l1_loss(pf, target, reduction='none')
        relative = F.smooth_l1_loss(predicted_centered, target_centered, reduction='none')
    elif kind == 'raw_mse':
        absolute = (pf - target).square()
        relative = (predicted_centered - target_centered).square()
    else:
        raise ValueError(kind)
    total = hit + ((absolute + relative) * valid).sum(1) / denom
    return (total * data['weights']).mean()


def root_average(values, roots, mask=None):
    result = []
    for root in np.unique(roots):
        chosen = np.asarray(values)[roots == root]
        if mask is not None:
            chosen = chosen[np.asarray(mask)[roots == root]]
        if chosen.size:
            result.append(float(chosen.mean()))
    return float(np.mean(result)) if result else None


def branch_metrics(states, actions):
    _, roots, hit, forgetting, cost = b.arrays(states)
    idx = np.arange(len(states))
    h, f, j = hit[idx, actions], forgetting[idx, actions], cost[idx, actions]
    work = np.array([[r['work'] for r in s['branches'][a]] for s, a in zip(states, actions)])
    return dict(reach=root_average(h, roots), raw_forgetting=root_average(f, roots, h > 0),
                cost=root_average(j, roots), work=root_average(work, roots))


def endpoint_metrics(rows):
    hit = np.array([r['success'] for r in rows])
    return dict(reach=float(hit.mean()),
                raw_forgetting=float(np.mean([r['forgetting'] for r in rows if r['success']])) if hit.any() else None,
                cost=float(np.mean([r['cost'] for r in rows])),
                work=float(np.mean([r['work'] for r in rows])))


def selection_key(metrics, selector):
    value = metrics['cost'] if selector == 'branch_J' else metrics['raw_forgetting']
    return (-round(metrics['reach'], 9), value if value is not None else float('inf'), metrics['work'])


def predictions(model, data, feedback):
    with torch.no_grad():
        hit, forgetting = model(b.features(data['x'], data['mean'], data['std'], feedback))
    return hit.sigmoid().numpy(), forgetting.numpy()


def training_metrics(states, model, data, feedback, slack=.1):
    x, roots, hit, forgetting, cost = b.arrays(states)
    ph, pf = predictions(model, data, feedback)
    actions = b.choose(ph, pf, slack)
    idx = np.arange(len(states))
    hm, fm, jm = hit.mean(2), forgetting.mean(2), cost.mean(2)
    oracle = np.array([min(range(5), key=lambda a: (-hm[i, a], fm[i, a])) for i in idx])
    all_ok = (hit == 1).all((1, 2))
    gap = fm[idx, actions] - fm.min(1)
    mask = all_ok & (np.ptp(fm, axis=1) > 1e-6)
    return dict(**branch_metrics(states, actions), actions=np.bincount(actions, minlength=5).tolist(),
        all_feasible_raw_regret=root_average(gap, roots, all_ok),
        informative_action_accuracy=float((gap[mask] <= 1e-6).mean()) if mask.any() else None,
        in_sample_J_regret=root_average(jm[idx, actions] - jm.min(1), roots),
        oracle_actions=np.bincount(oracle, minlength=5).tolist(),
        hit_brier=float(np.mean((ph - hm)**2)), all_feasible_states=int(all_ok.sum()))


def fit(out, train, dev, dev_bank, config, kind, feedback, seed, checkpoints,
        selectors=SELECTORS):
    """Every selector sees the identical checkpoint/slack candidate pool."""
    out.mkdir(parents=True, exist_ok=False)
    data = prepare(train, kind)
    dev_data = dict(data, x=np.array([s['x'] for s in dev], dtype=np.float32))
    torch.manual_seed(seed)
    model = b.Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
    tx = b.features(data['x'], data['mean'], data['std'], feedback)
    best, selected, candidates, training_log = {}, {}, [], []
    queried_work = 0.
    for epoch in range(max(checkpoints) + 1):
        if epoch in checkpoints:
            model.eval()
            ph, pf = predictions(model, dev_data, feedback)
            snapshot = dict(state=copy.deepcopy(model.state_dict()), mean=data['mean'], std=data['std'],
                            feedback=feedback, seed=seed, epoch=epoch, loss_kind=kind)
            with torch.no_grad():
                train_loss = float(objective(model, tx, data, kind))
            training_log.append(dict(epoch=epoch, loss=train_loss,
                diagnostics=training_metrics(train, model, data, feedback)))
            for slack in SLACKS:
                actions = b.choose(ph, pf, slack)
                branch = branch_metrics(dev, actions)
                record = dict(epoch=epoch, slack=slack, branch=branch)
                if 'closed_F' in selectors:
                    ckpt = dict(snapshot, slack=slack)
                    rows = [b.rollout(item, config, b.scorer_chooser(model, ckpt)) for item in dev_bank]
                    record['closed'] = endpoint_metrics(rows)
                    record['closed_rows'] = [{k: v for k, v in r.items() if k != 'events'} for r in rows]
                    queried_work += sum(r['work'] for r in rows)
                candidates.append(record)
                for selector in selectors:
                    metrics = record['closed'] if selector == 'closed_F' else branch
                    key = selection_key(metrics, selector)
                    if selector not in best or key < best[selector]:
                        best[selector] = key
                        selected[selector] = dict(snapshot, slack=slack, selector=selector, dev=metrics)
            print(__import__('json').dumps(dict(stage='fit', kind=kind, feedback=feedback,
                seed=seed, epoch=epoch, training_loss=train_loss)), flush=True)
        if epoch == max(checkpoints):
            break
        model.train()
        loss = objective(model, tx, data, kind)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
    manifest = {}
    for selector, ckpt in selected.items():
        torch.save(ckpt, out / f'{selector}.pt')
        manifest[selector] = {k: ckpt[k] for k in ['seed', 'epoch', 'feedback', 'slack', 'loss_kind', 'selector', 'dev']}
    b.save(out / 'selection.json', manifest)
    b.save(out / 'candidates.json', candidates)
    b.save(out / 'training.json', training_log)
    b.save(out / 'cost.json', dict(closed_dev_rollouts=len(dev_bank) * len(checkpoints) * len(SLACKS)
                                 if 'closed_F' in selectors else 0, closed_dev_work=queried_work))
    return manifest


def positive_states(seed, contexts):
    """Auxiliary supervised control, NOT MoE outcomes or proof of feedback value.

    Each context has identical progress features and five equally likely feedback
    values. Thus progress-only exact action accuracy is 1/5 on complete contexts.
    """
    rng = np.random.default_rng(seed)
    states = []
    for i in range(contexts):
        common = rng.normal(0, .5, 12)
        base = .002 + .0005 * np.tanh(common[0])
        for label in range(5):
            x = [*common, label / 4, float(rng.normal(0, .1))]
            branches = []
            for a in range(5):
                forgetting = base + .008 * ((a - label) / 4)**2
                branches.append([dict(success=True, forgetting=forgetting, work=1.,
                                      cost=b.objective(1, forgetting))])
            states.append(dict(root=seed + i, state_id=f'{seed}:{i}:{label}', x=x,
                               branches=branches, protected_mass=1., label=label))
    return states


def positive_score(states, model, ckpt):
    x = np.array([s['x'] for s in states], dtype=np.float32)
    actions = b.predict_actions(model, x, ckpt['mean'], ckpt['std'], ckpt['feedback'], ckpt['slack'])
    _, _, _, f, _ = b.arrays(states)
    idx = np.arange(len(states))
    regret = f[idx, actions, 0] - f[:, :, 0].min(1)
    central = f[:, 2, 0] - f[:, :, 0].min(1)
    return dict(accuracy=float(np.mean(actions == np.array([s['label'] for s in states]))),
                raw_regret=float(regret.mean()), central_baseline_regret=float(central.mean()),
                relative_regret=float(regret.mean() / central.mean()),
                actions=np.bincount(actions, minlength=5).tolist())


def tiny_fit(out, states, kind, seed, checkpoints):
    data = prepare(states, kind)
    torch.manual_seed(seed)
    model = b.Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-4)
    tx = b.features(data['x'], data['mean'], data['std'], True)
    records = []
    for epoch in range(max(checkpoints) + 1):
        if epoch in checkpoints:
            with torch.no_grad(): value = float(objective(model, tx, data, kind))
            records.append(dict(epoch=epoch, loss=value,
                metrics=training_metrics(states, model, data, True)))
        if epoch == max(checkpoints): break
        loss = objective(model, tx, data, kind)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
    b.save(out, dict(kind=kind, seed=seed, states=[s['state_id'] for s in states], records=records,
        interpretation='Training-set memorization only; full original feature set, architecture, optimizer and weight decay retained. Longer training does not establish generalization.'))
    return records


def plan_candidates():
    candidates = [dict(name=f'fixed_{a}', kind='fixed', action=a) for a in range(5)]
    for kind, thresholds in [('time', [.1, .3, .6]), ('progress', [.25, .5, .75])]:
        for early, late in [(0, 4), (4, 0)]:
            for threshold in thresholds:
                candidates.append(dict(name=f'{kind}_{early}_{late}_{threshold}', kind=kind,
                                       early=early, late=late, threshold=threshold))
    for action in [0, 1, 2, 3]:
        for tolerance in [.001, .01, .03]:
            for release in [0., .25]:
                candidates.append(dict(name=f'hand_{action}_{tolerance}_{release}', kind='hand',
                    action=action, tolerance=tolerance, release=release))
    return candidates


def plan_chooser(plan):
    if plan['kind'] == 'fixed': return lambda env: plan['action']
    if plan['kind'] == 'hand': return b.baseline_chooser('hand', plan)
    switched = False
    def choose(env):
        nonlocal switched
        if plan['kind'] == 'time':
            progress = env.steps / env.c['max_steps']
        else:
            initial = max(math.log(max(env.initial_loss, 1e-8) / env.c['target']), 1e-6)
            progress = 1 - math.log(max(env.current_loss, 1e-8) / env.c['target']) / initial
        switched = switched or progress >= plan['threshold']
        return plan['late'] if switched else plan['early']
    return choose

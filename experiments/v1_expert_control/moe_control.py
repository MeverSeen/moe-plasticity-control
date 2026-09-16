"""CPU-scale expert-update control experiment. All data are synthetic.

Fixed top-2 routing; trainable nonlinear experts; shared policy across experts.
No final-test labels are used for policy observations, rewards or selection.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
DTYPE = torch.float32

DEFAULT = dict(experts=6, dim=10, hidden=12, tasks=5, top_k=2,
               train_n=384, monitor_n=128, test_n=512,
               replay_memory=64, monitor_memory=64,
               batch=32, old_batch=32, lr=0.015, replay_weight=0.1,
               max_steps=300, interval=5, base_steps=250,
               target=0.04, levels=[0.16, 0.10, 0.06, 0.04],
               lambda_max=0.15, noise=0.05)


def gen(seed):
    return torch.Generator().manual_seed(int(seed))


def init_weights(c, seed):
    e, d, h = c['experts'], c['dim'], c['hidden']
    rng = gen(seed)
    return torch.cat([torch.randn(e, d*h, generator=rng)/math.sqrt(d),
                      torch.zeros(e, h),
                      torch.randn(e, h, generator=rng)/math.sqrt(h),
                      torch.zeros(e, 1)], 1)


def routes(x, router, c):
    scores = x[:, :2] @ router
    values, ids = scores.topk(c['top_k'], dim=1)
    return ids, values.softmax(1)


def predict(w, x, router, c):
    ids, gates = routes(x, router, c)
    selected = w[ids]
    d, h = c['dim'], c['hidden']
    a = selected[:, :, :d*h].reshape(-1, c['top_k'], d, h)
    b = selected[:, :, d*h:d*h+h]
    v = selected[:, :, d*h+h:d*h+2*h]
    hidden = torch.tanh(torch.einsum('bd,bkdh->bkh', x, a) + b)
    result = (hidden*v).sum(-1) + selected[:, :, -1]
    return (result*gates).sum(1)


def loss(w, data, router, c):
    return ((predict(w, data[0], router, c)-data[1])**2).mean()


def mse(w, data, router, c):
    with torch.no_grad():
        return float(loss(w, data, router, c))


def reference_errors(w, data, router, c):
    with torch.no_grad():
        return (predict(w, data[0], router, c)-data[1]).square().detach()


def trim_references(refs, c):
    # Only store one historical loss value for each retained monitor sample.
    n = c['monitor_memory']//len(refs)
    return [r[:n].clone() for r in refs]


def fisher_diagonal(w, x, router, c):
    """Exact empirical diagonal E[(d f(x)/d theta)^2] for this model.

    This is a Gauss-Newton/Fisher proxy under fixed-variance Gaussian outputs,
    not a noisy squared gradient of a nearly fitted regression loss.
    """
    with torch.no_grad():
        ids, gates = routes(x, router, c)
        selected = w[ids]
        d, h = c['dim'], c['hidden']
        a = selected[:, :, :d*h].reshape(-1, c['top_k'], d, h)
        b = selected[:, :, d*h:d*h+h]
        v = selected[:, :, d*h+h:d*h+2*h]
        z = torch.tanh(torch.einsum('bd,bkdh->bkh', x, a)+b)
        dz = (1-z*z)*v*gates.unsqueeze(-1)
        j_w = (x[:, None, :, None]*dz[:, :, None, :]).flatten(2)
        jac = torch.cat([j_w, dz, z*gates.unsqueeze(-1), gates.unsqueeze(-1)], -1)
        imp = torch.zeros_like(w)
        imp.index_add_(0, ids.flatten(), jac.square().flatten(0, 1))
        return imp / len(x)


def make_world(seed, c, include_test=False):
    """A consistent nonlinear teacher, with overlapping input domains.

    Domains differ in input distribution, never conflicting labels for the same x.
    The router depends on raw input and is immutable, so routes match exactly
    across methods. Test draws have their own generator and are materialized only
    by the offline evaluator.
    """
    e = c['experts']
    angle = torch.arange(e)*2*math.pi/e + (seed % 43)*0.137
    router = 1.4*torch.stack([angle.cos(), angle.sin()])
    teacher = init_weights(c, seed*31+9)
    calibration = torch.randn(2048, c['dim'], generator=gen(seed*31+10))
    calibration[:, :2] *= 1.5
    scale = predict(teacher, calibration, router, c).std().clamp_min(0.1)
    teacher[:, -(c['hidden']+1):] /= scale
    rng = gen(seed*31+11)
    # A random domain order, not fixed expert IDs or task indices in policy input.
    order = torch.randperm(e, generator=rng).tolist()[:c['tasks']]
    tasks = []
    for j, domain in enumerate(order):
        theta = float(angle[domain]) + 0.3
        center = torch.tensor([math.cos(theta), math.sin(theta)])*1.6
        partitions = {}
        for name, n, offset in [('train', c['train_n'], 0),
                                ('monitor', c['monitor_n'], 1),
                                ('test', c['test_n'], 2)]:
            if name == 'test' and not include_test:
                continue
            r = gen(seed*100003+j*1009+offset*71+99)
            x = torch.randn(n, c['dim'], generator=r)
            x[:, :2] = x[:, :2]*0.8 + center
            y = predict(teacher, x, router, c) + c['noise']*torch.randn(n, generator=r)
            partitions[name] = (x, y)
        tasks.append(partitions)
    return dict(seed=seed, router=router, tasks=tasks,
                initial=init_weights(c, seed*31+12), order=order)


def subset(data, n):
    return data[0][:n], data[1][:n]


def memory(world, count, c):
    nr = c['replay_memory']//count
    nm = c['monitor_memory']//count
    replay = [subset(world['tasks'][i]['train'], nr) for i in range(count)]
    # New-task monitoring samples become historical monitor memory; disjoint
    # from replay/training samples. They are outer-training data, not test data.
    monitoring = [subset(world['tasks'][i]['monitor'], nm) for i in range(count)]
    return (torch.cat([p[0] for p in replay]), torch.cat([p[1] for p in replay])), monitoring


def pretrain(world, n_past, c):
    w = world['initial'].clone().requires_grad_()
    opt = torch.optim.Adam([w], lr=c['lr'])
    refs = []
    for j in range(n_past):
        data = world['tasks'][j]['train']
        old = memory(world, j, c)[0] if j else None
        rng = gen(world['seed']*17+j*151+88)
        for t in range(c['base_steps']):
            ix = torch.randint(len(data[0]), (c['batch'],), generator=rng)
            l = loss(w, (data[0][ix], data[1][ix]), world['router'], c)
            if old is not None:
                oi = torch.randint(len(old[0]), (c['old_batch'],), generator=rng)
                l = l+c['replay_weight']*loss(w, (old[0][oi], old[1][oi]), world['router'], c)
            opt.zero_grad()
            l.backward()
            opt.step()
        refs.append(reference_errors(w, world['tasks'][j]['monitor'], world['router'], c))
        refs = trim_references(refs, c)
    return w.detach(), refs


def normalized_scores(values):
    """Preserve average update budget, including when no expert is observed."""
    return values/(values.mean()+1e-8)


class AdaptEnv:
    def __init__(self, world, task, initial, references, c):
        self.c, self.world, self.task = c, world, task
        self.router = world['router']
        self.w = initial.detach().clone().requires_grad_()
        self.anchor = initial.detach().clone()
        self.references = references
        self.new = world['tasks'][task]['train']
        self.new_monitor = world['tasks'][task]['monitor']
        self.old, self.old_monitor = memory(world, task, c)
        self.old_reference = [float(references[i][:len(d[0])].mean())
                              for i, d in enumerate(self.old_monitor)]
        imp = fisher_diagonal(self.anchor, self.old[0], self.router, c)
        self.importance = imp/(imp.mean()+1e-8)
        self.imp_score = self.importance.mean(1)
        self.m = torch.zeros_like(self.w)
        self.v = torch.zeros_like(self.w)
        self.step_count = 0
        self.prev_action = torch.zeros(c['experts'], 2)
        self.initial_loss = mse(self.w, self.new_monitor, self.router, c)
        self.current_loss = self.initial_loss
        self.previous_loss = self.initial_loss
        self.forgetting = self.forget_monitor()
        self.initial_forgetting = self.forgetting
        rng = gen(world['seed']*199+task*1331+744)
        self.new_indices = torch.randint(len(self.new[0]), (c['max_steps']+1, c['batch']), generator=rng)
        self.old_indices = torch.randint(len(self.old[0]), (c['max_steps']+1, c['old_batch']), generator=rng)
        self.done = self.current_loss <= c['target']
        self.obs_calls = 0
        self.hits = {v: (0, self.current_loss, self.w.detach().clone())
                     for v in c['levels'] if self.current_loss <= v}

    def forget_monitor(self):
        vals = [mse(self.w, d, self.router, self.c) for d in self.old_monitor]
        return float(np.mean(np.maximum(np.array(vals)-self.old_reference, 0)))

    def gradients(self, idx):
        ni, oi = self.new_indices[idx], self.old_indices[idx]
        new = (self.new[0][ni], self.new[1][ni])
        old = (self.old[0][oi], self.old[1][oi])
        gn = torch.autograd.grad(loss(self.w, new, self.router, self.c), self.w)[0]
        go = torch.autograd.grad(loss(self.w, old, self.router, self.c), self.w)[0]
        return gn, go, new[0], old[0]

    def observation(self):
        self.obs_calls += 1
        gn, go, xn, xo = self.gradients(self.step_count)
        n, o = gn.norm(dim=1), go.norm(dim=1)
        cosine = (gn*go).sum(1)/(n*o+1e-8)
        rn, _ = routes(xn, self.router, self.c)
        ro, _ = routes(self.old[0], self.router, self.c)
        fn = torch.bincount(rn.flatten(), minlength=self.c['experts']).float()/len(xn)
        fo = torch.bincount(ro.flatten(), minlength=self.c['experts']).float()/len(self.old[0])
        drift = (self.w.detach()-self.anchor).square().mean(1).sqrt()
        momentum_align = (self.m*gn).sum(1)/(self.m.norm(dim=1)*n+1e-8)
        local = torch.stack([torch.log1p(n*10), torch.log1p(o*10), cosine, fn, fo,
                             torch.log1p(self.imp_score), drift,
                             momentum_align, (n>1e-7).float(), (o>1e-7).float(),
                             self.prev_action[:, 0], self.prev_action[:, 1]], 1)
        global_f = torch.tensor([math.log(max(self.current_loss, 1e-7)/self.c['target']),
                                 math.log(max(self.initial_loss, 1e-7)/self.c['target']),
                                 self.previous_loss-self.current_loss,
                                 self.forgetting, self.step_count/self.c['max_steps']]).clamp(-10, 10)
        return local.detach(), global_f

    def potential(self):
        # Shaping has zero terminal potential, gamma=1. The undiscounted sum
        # is exactly terminal base reward minus initial potential.
        if self.done:
            return 0.0
        gap = max(math.log(max(self.current_loss, 1e-8)/self.c['target']), 0)
        return -0.7*gap-2.0*min(self.forgetting, 2.0)

    def advance(self, controls):
        alpha, strength = controls[:, 0], controls[:, 1]
        phi0 = self.potential()
        self.prev_action = torch.stack([(alpha-1.025)/0.975,
                                       strength/self.c['lambda_max']*2-1], 1).detach()
        c = self.c
        old_loss = self.current_loss
        for _ in range(min(c['interval'], c['max_steps']-self.step_count)):
            gn, go, _, _ = self.gradients(self.step_count)
            reg = strength[:, None]*self.importance*(self.w.detach()-self.anchor)
            g = gn+c['replay_weight']*go+reg
            # Adam state updates are shared in implementation across all arms.
            self.step_count += 1
            self.m.mul_(0.9).add_(g, alpha=0.1)
            self.v.mul_(0.99).addcmul_(g, g, value=0.01)
            mh = self.m/(1-0.9**self.step_count)
            vh = self.v/(1-0.99**self.step_count)
            with torch.no_grad():
                self.w.add_(-c['lr']*alpha[:, None]*mh/(vh.sqrt()+1e-8))
            # Observe threshold crossings every optimizer step even though the
            # controller changes its action only once per control interval.
            self.current_loss = mse(self.w, self.new_monitor, self.router, c)
            for threshold in c['levels']:
                if threshold not in self.hits and self.current_loss <= threshold:
                    self.hits[threshold] = (self.step_count, self.current_loss, self.w.detach().clone())
            if self.current_loss <= c['target']:
                break
        self.previous_loss = old_loss
        self.forgetting = self.forget_monitor()
        success = self.current_loss <= c['target']
        self.done = success or self.step_count >= c['max_steps']
        base = 0.0
        if self.done:
            base = (-5*min(self.forgetting, 2.0) if success else
                    -12-min(max(self.current_loss/c['target']-1, 0), 8))
        reward = base+self.potential()-phi0
        return float(reward), self.done, dict(success=success, base_reward=base)


def static_controls(method, local, config, c):
    lr_scale, protection, slope = config
    if method == 'uniform':
        score = torch.ones(c['experts'])
    elif method == 'activation':
        score = normalized_scores(local[:, 4].clamp_min(1e-5))
    elif method == 'importance':
        score = normalized_scores(torch.expm1(local[:, 5]).clamp_min(1e-5))
    else:
        raise ValueError(method)
    multiplier = normalized_scores(1/(1+slope*score))
    alpha = (lr_scale*multiplier).clamp(0.05, 2)
    strength = (protection*score).clamp(0, c['lambda_max'])
    return torch.stack([alpha, strength], 1)


class Controller(nn.Module):
    """Permutation-equivariant expert policy; no expert ID or task ID input."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(32*2+5, 32), nn.Tanh(), nn.Linear(32, 2))
        self.critic = nn.Sequential(nn.Linear(32+5, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_std = nn.Parameter(torch.tensor([-0.5, -0.5]))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.actor[-1].bias.data[1] = -1.0

    def forward(self, local, global_f):
        z = self.encoder(local)
        pool = z.mean(-2)
        global_z = torch.cat([pool, global_f], -1)
        ctx = global_z.unsqueeze(-2).expand(*z.shape[:-1], global_z.shape[-1])
        mean = self.actor(torch.cat([z, ctx], -1))
        value = self.critic(global_z).squeeze(-1)
        dist = torch.distributions.Normal(mean, self.log_std.exp().clamp(0.08, 1.0))
        return dist, value

    def controls(self, latent, c):
        y = latent.tanh()
        return torch.stack([0.05+1.95*(y[..., 0]+1)/2,
                            c['lambda_max']*(y[..., 1]+1)/2], -1)


def run_episode(env, method, config=None, policy=None, train=False):
    trajectory, events = [], []
    snapshots = [(0, env.current_loss, env.w.detach().clone())]
    while not env.done:
        local, glob = env.observation()
        if method == 'learned':
            with torch.no_grad():
                dist, value = policy(local, glob)
                latent = dist.sample() if train else dist.mean
                logp = dist.log_prob(latent).sum()
                controls = policy.controls(latent, env.c)
        else:
            controls = static_controls(method, local, config, env.c)
        reward, done, info = env.advance(controls)
        if train:
            trajectory.append(dict(local=local, global_f=glob, latent=latent,
                                   logp=logp, value=value, reward=reward))
        events.append(dict(step=env.step_count, loss=env.current_loss, forgetting=env.forgetting,
                           alpha=controls[:, 0].tolist(), protection=controls[:, 1].tolist()))
        snapshots.append((env.step_count, env.current_loss, env.w.detach().clone()))
    success = env.current_loss <= env.c['target']
    score = (-5*min(env.forgetting, 2) if success else
             -12-min(max(env.current_loss/env.c['target']-1, 0), 8))
    return dict(trajectory=trajectory, events=events, snapshots=snapshots,
                success=success, score=score, steps=env.step_count,
                loss=env.current_loss, forgetting=env.forgetting, hits=env.hits)


def make_bank(seeds, c):
    bank = []
    for seed in seeds:
        world = make_world(seed, c)
        history = 1+seed%3
        w, refs = pretrain(world, history, c)
        bank.append((world, history, w, refs))
    return bank


def bank_eval(bank, c, method, config=None, policy=None):
    results = []
    for world, task, w, refs in bank:
        env = AdaptEnv(world, task, w, refs, c)
        r = run_episode(env, method, config, policy)
        results.append({k: r[k] for k in ['success', 'score', 'steps', 'loss', 'forgetting']})
    return dict(success=float(np.mean([r['success'] for r in results])),
                score=float(np.mean([r['score'] for r in results])),
                forgetting=float(np.mean([r['forgetting'] for r in results])), results=results)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def pilot(out, c):
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    bank = make_bank(range(10, 26), c)
    result = bank_eval(bank, c, 'uniform', (1.0, 0.02, 0.0))
    write_json(out/'pilot.json', result)
    print(json.dumps(dict(stage='pilot', elapsed=time.time()-start, **result)), flush=True)


def tune(out, c, n_dev=8):
    start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    bank = make_bank(range(1000, 1000+n_dev), c)
    configs = [(a, p, s) for a in [0.5, 1.0, 1.5]
               for p in [0.0, 0.01, 0.03] for s in [0.0, 2.0]]
    chosen, rows = {}, []
    for method in ['uniform', 'activation', 'importance']:
        # Equal number of unique candidates; slope is redundant for uniform,
        # so substitute finer learning-rate values for that arm.
        grid = configs if method != 'uniform' else [(a,p,0.0) for a in [.3,.5,.75,1,1.5,2]
                                                   for p in [0.0,.01,.03]]
        for cfg in grid:
            r = bank_eval(bank, c, method, cfg)
            rows.append(dict(method=method, config=cfg, **r))
        candidates = [r for r in rows if r['method']==method]
        # Reachability first, then forgetting, before any final-test access.
        best = min(candidates, key=lambda r:(-r['success'], r['forgetting']))
        chosen[method] = best['config']
        print(json.dumps(dict(stage='tune', method=method, config=best['config'],
                              success=best['success'], forgetting=best['forgetting'])), flush=True)
        write_json(out/'tuning_results.json', rows)
        write_json(out/'selected_baselines.json', chosen)
    write_json(out/'tuning_cost.json',dict(seconds=time.time()-start, candidates_per_method=18,
                                         validation_worlds=n_dev))


def train(out, c, iterations=40, episodes=8, n_train=32, n_dev=8):
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    bank = make_bank(range(100,100+n_train), c)
    validation = make_bank(range(1000,1000+n_dev), c)
    policy = Controller()
    torch.save(dict(policy=policy.state_dict(), config=c, iteration=0), out/'controller_initial.pt')
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history, best_key = [], None
    for iteration in range(iterations+1):
        if iteration % 5 == 0 or iteration == iterations:
            val = bank_eval(validation, c, 'learned', policy=policy)
            row = dict(iteration=iteration, elapsed=time.time()-start,
                       success=val['success'], forgetting=val['forgetting'], score=val['score'])
            history.append(row)
            print(json.dumps(dict(stage='train_validation', **row)), flush=True)
            key = (-val['success'], val['forgetting'])
            if best_key is None or key < best_key:
                best_key = key
                torch.save(dict(policy=policy.state_dict(), config=c, iteration=iteration,
                                validation=row), out/'controller.pt')
            write_json(out/'training_history.json', history)
        if iteration == iterations:
            break
        transitions, episode_scores = [], []
        for _ in range(episodes):
            world, task, w, refs = bank[np.random.randint(len(bank))]
            env = AdaptEnv(world, task, w, refs, c)
            rollout = run_episode(env, 'learned', policy=policy, train=True)
            episode_scores.append(rollout['score'])
            seq = rollout['trajectory']
            # GAE with gamma=1: do not reward postponing a negative terminal cost.
            advantage = 0.0
            for t in reversed(range(len(seq))):
                nxt = float(seq[t+1]['value']) if t+1<len(seq) else 0.0
                delta = seq[t]['reward']+nxt-float(seq[t]['value'])
                advantage = delta+0.95*advantage
                seq[t]['advantage'] = advantage
                seq[t]['return'] = advantage+float(seq[t]['value'])
            transitions.extend(seq)
        if not transitions:
            continue
        local = torch.stack([r['local'] for r in transitions])
        glob = torch.stack([r['global_f'] for r in transitions])
        latent = torch.stack([r['latent'] for r in transitions])
        old_logp = torch.stack([r['logp'] for r in transitions])
        adv = torch.tensor([r['advantage'] for r in transitions])
        ret = torch.tensor([r['return'] for r in transitions])
        adv = (adv-adv.mean())/(adv.std()+1e-6)
        for epoch in range(4):
            for idx in torch.randperm(len(transitions)).split(64):
                dist, value = policy(local[idx], glob[idx])
                logp = dist.log_prob(latent[idx]).sum((-1,-2))
                ratio = (logp-old_logp[idx]).exp()
                actor = -torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]).mean()
                critic = 0.5*(value-ret[idx]).square().mean()
                entropy = dist.entropy().mean()
                objective = actor+0.5*critic-0.005*entropy
                opt.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), .5)
                opt.step()
        if iteration%5 == 4:
            print(json.dumps(dict(stage='ppo', iteration=iteration+1,
                                  sampled_score=float(np.mean(episode_scores)),
                                  transitions=len(transitions))), flush=True)
    torch.save(dict(policy=policy.state_dict(), config=c), out/'controller_last.pt')
    write_json(out/'training_cost.json', dict(seconds=time.time()-start, iterations=iterations,
               episodes_per_iteration=episodes, training_worlds=n_train, validation_worlds=n_dev,
               policy_parameters=sum(p.numel() for p in policy.parameters())))


def evaluate(out, c, n_test=20):
    if (out/'locked_protocol.json').exists():
        raise RuntimeError('This run is sealed. Do not overwrite a held-out evaluation.')
    selected = json.loads((out/'selected_baselines.json').read_text())
    checkpoint = torch.load(out/'controller.pt', weights_only=False)
    assert checkpoint['config'] == c, 'Training and evaluation configurations differ.'
    policy = Controller()
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    rows, trajectories, terminal = [], [], []
    # Seal all choices before exposing any held-out test sequence.
    protocol = dict(config=c, pilot_seeds=list(range(10,26)),
                    train_seeds=list(range(100,132)), dev_seeds=list(range(1000,1008)),
                    test_seeds=list(range(10000,10000+n_test)),
                    selected_baselines=selected, controller_iteration=checkpoint['iteration'],
                    policy_sha256=hashlib.sha256((out/'controller.pt').read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    diagnostic_arm='untrained controller, not used for model selection',
                    endpoint='first monitored new-task MSE <= common threshold',
                    primary='positive old-test MSE increase, paired by sequence at matched threshold',
                    primary_threshold=c['target'],
                    inference='10000 paired bootstrap draws clustered by world/sequence, RNG seed 916',
                    sensitivity='both losses within 5% below threshold; also exclude zero-update endpoints',
                    success_rule='No efficacy claim from conditional forgetting alone if reachability is worse.',
                    caveat='fixed routing synthetic regression only; no LLM efficacy claim')
    write_json(out/'locked_protocol.json', protocol)
    start = time.time()
    for seed in protocol['test_seeds']:
        world = make_world(seed, c, include_test=True)
        initial, initial_refs = pretrain(world, 1, c)
        untrained = Controller()
        untrained.load_state_dict(torch.load(out/'controller_initial.pt',weights_only=False)['policy'])
        first_test_reference = mse(initial,world['tasks'][0]['test'],world['router'],c)
        for method in ['uniform','activation','importance','learned','untrained']:
            w, refs = initial.clone(), [initial_refs[0].clone()]
            test_references = [first_test_reference]
            method_start = time.time()
            steps_total = 0
            for task in range(1,c['tasks']):
                env = AdaptEnv(world, task, w, refs, c)
                run = run_episode(env, 'learned' if method=='untrained' else method,
                                  selected.get(method), untrained if method=='untrained' else policy)
                old_test_refs = np.array(test_references)
                for threshold in c['levels']:
                    hit = run['hits'].get(threshold)
                    if hit is None:
                        rows.append(dict(seed=seed,method=method,task=task,threshold=threshold,
                                         reached=False,steps=c['max_steps'],new_monitor_loss=None,
                                         overshoot=None,old_forgetting=None,old_absolute_loss=None,
                                         new_test_loss=None,old_signed_change=None,
                                         initial_new_loss=env.initial_loss))
                    else:
                        step,new_loss,wh = hit
                        old_losses = np.array([mse(wh,world['tasks'][i]['test'],world['router'],c)
                                               for i in range(task)])
                        delta = old_losses-old_test_refs
                        rows.append(dict(seed=seed,method=method,task=task,threshold=threshold,
                                         reached=True,steps=step,new_monitor_loss=new_loss,
                                         overshoot=threshold-new_loss,
                                         old_forgetting=float(np.maximum(delta,0).mean()),
                                         old_signed_change=float(delta.mean()),
                                         old_absolute_loss=float(old_losses.mean()),
                                         initial_new_loss=env.initial_loss,
                                         new_test_loss=mse(wh,world['tasks'][task]['test'],world['router'],c)))
                trajectories.append(dict(seed=seed,method=method,task=task,
                                         initial_loss=env.initial_loss,events=run['events']))
                steps_total += run['steps']
                w = env.w.detach().clone()
                refs.append(reference_errors(w,world['tasks'][task]['monitor'],world['router'],c))
                refs = trim_references(refs,c)
                test_references.append(mse(w,world['tasks'][task]['test'],world['router'],c))
            final_losses = [mse(w,t['test'],world['router'],c) for t in world['tasks']]
            terminal.append(dict(seed=seed,method=method,steps=steps_total,
                                 seconds=time.time()-method_start,final_loss=float(np.mean(final_losses)),
                                 task_losses=final_losses))
        print(json.dumps(dict(stage='evaluate',seed=seed,elapsed=time.time()-start)),flush=True)
        write_json(out/'test_endpoints.json',rows)
        write_json(out/'test_trajectories.json',trajectories)
        write_json(out/'test_terminal.json',terminal)
    with (out/'test_endpoints.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['pilot','tune','train','evaluate'])
    p.add_argument('--out',default='work/run_v1')
    p.add_argument('--config')
    p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--n-test',type=int,default=20)
    a=p.parse_args()
    c=copy.deepcopy(DEFAULT)
    if a.config:c.update(json.loads(Path(a.config).read_text()))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='pilot':pilot(out,c)
    elif a.stage=='tune':tune(out,c)
    elif a.stage=='train':train(out,c,a.iterations,a.episodes)
    elif a.stage=='evaluate':evaluate(out,c,a.n_test)


if __name__=='__main__':main()

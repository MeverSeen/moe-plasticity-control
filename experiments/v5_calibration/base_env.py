"""A scalar feedback controller for a fixed-capacity synthetic sparse MoE.

No test labels enter observations, rewards, grouping, or model selection.
The protection mask is measured once at the start of B and stays fixed.
The learned progress/feedback policies have identical architectures and budgets.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import model_core as core

torch.set_num_threads(1)
ACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONFIG = dict(core.DEFAULT, target=0.08, levels=[0.08], max_steps=300,
              interval=5, replay_weight=0.1, mask_old_threshold=1/3,
              mask_new_threshold=0.05, reward_scale=0.01,
              failure_penalty=10.0, forgetting_cap=5.0)
TRAIN_SEEDS = list(range(230000, 230064))
DEV_SEEDS = list(range(240000, 240024))
TEST_SEEDS = list(range(250000, 250080))
POLICY_SEEDS = [9171, 9172, 9173]
FEATURE_NAMES = [
    'initial_protected_B_gate_mass', 'initial_old_usage_protected',
    'initial_old_usage_other', 'initial_protected_fraction',
    'initial_B_usage_entropy', 'initial_old_log_loss', 'initial_new_log_gap',
    'new_log_gap', 'recent_new_log_improvement', 'remaining_work_fraction',
    'step_fraction', 'previous_multiplier',
    'current_monitor_forgetting_scaled', 'recent_old_loss_change_scaled']
FEEDBACK_SLICE = slice(12, 14)


def save(path, value):
    core.write_json(path, value)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bank(seeds, c, test=False):
    result = []
    for seed in seeds:
        world = core.make_world(seed, c, include_test=test)
        task = 1 + seed % 3
        w, refs = core.pretrain(world, task, c)
        result.append((world, task, w, refs))
    return result


class ScalarEnv:
    """One new-task episode after 1--3 old tasks; repeated decisions within B.

    Work units: one forward example = 1; forward+backward example = 3.
    A policy MAC is charged relative to the model's per-example affine MACs.
    This is a declared workload proxy, NOT exact hardware FLOPs or wall time.
    Initial grouping/measurement and all online monitoring are charged.
    Every arm uses the same monitoring calls (information-ablation setting).
    """
    def __init__(self, item, c, policy_macs=0):
        world, task, w, refs = item
        self.c, self.world, self.task = c, world, task
        self.router = world['router']
        self.w = w.detach().clone().requires_grad_()
        self.initial_w = w.detach().clone()
        self.new = world['tasks'][task]['train']
        self.new_monitor = world['tasks'][task]['monitor']
        self.old, self.monitors = core.memory(world, task, c)
        self.old_refs = np.array([core.mse(w, d, self.router, c) for d in self.monitors])
        self.old_losses = self.old_refs.copy()
        self.prev_old_mean = float(self.old_refs.mean())
        self.old_n = sum(len(d[0]) for d in self.monitors)
        self.current_loss = core.mse(w, self.new_monitor, self.router, c)
        self.initial_loss = self.current_loss
        self.previous_loss = self.current_loss
        ro, _ = core.routes(self.old[0], self.router, c)
        rb, gb = core.routes(self.new_monitor[0], self.router, c)
        self.ra = torch.bincount(ro.flatten(), minlength=c['experts']).float()/len(ro)
        self.rb = torch.bincount(rb.flatten(), minlength=c['experts']).float()/len(rb)
        self.mask = ((self.ra >= c['mask_old_threshold']) &
                     (self.rb >= c['mask_new_threshold']))
        gate_mass = torch.zeros(c['experts'])
        gate_mass.index_add_(0, rb.flatten(), gb.flatten()/len(rb))
        self.protected_mass = float(gate_mass[self.mask].sum())
        def mean_or_zero(x):
            return float(x.mean()) if x.numel() else 0.0
        q = self.rb/self.rb.sum().clamp_min(1e-8)
        self.initial_features = [self.protected_mass,
            mean_or_zero(self.ra[self.mask]), mean_or_zero(self.ra[~self.mask]),
            float(self.mask.float().mean()),
            float(-(q*q.clamp_min(1e-8).log()).sum()/math.log(c['experts'])),
            math.log1p(float(self.old_refs.mean())/c['target']),
            math.log(max(self.initial_loss, 1e-8)/c['target'])]
        self.adam_m = torch.zeros_like(w)
        self.adam_v = torch.zeros_like(w)
        self.steps = 0
        self.last_multiplier = 1.0
        self.forgetting = 0.0
        self.events = []
        # Same exogenous batches for every arm on the same root history.
        rng = core.gen(world['seed']*199+task*1331+744)
        self.new_indices = torch.randint(len(self.new[0]), (c['max_steps'],c['batch']), generator=rng)
        self.old_indices = torch.randint(len(self.old[0]), (c['max_steps'],c['old_batch']), generator=rng)
        affine_macs = c['top_k']*(c['dim']*c['hidden']+c['hidden'])+2*c['experts']
        self.policy_charge = float(policy_macs)/affine_macs
        # Initial measurements plus conservative charge for routing-only probes.
        self.work = float(len(self.new_monitor[0])+self.old_n+len(self.old[0])+len(self.new_monitor[0]))
        self.initial_work = self.work
        # Common envelope: full 300 model updates, at most 64 old monitors each interval.
        self.step_charge = 3*(c['batch']+c['old_batch'])+len(self.new_monitor[0])
        common_initial = 2*c['monitor_n']+c['monitor_memory']+c['replay_memory']
        common_decisions = math.ceil(c['max_steps']/c['interval'])
        self.budget = float(common_initial+c['max_steps']*self.step_charge+
                            common_decisions*c['monitor_memory'])
        self.success = self.current_loss <= c['target']
        self.done = self.success
        self.reason = 'target' if self.success else None

    def observe(self, feedback=True):
        c = self.c
        values = self.initial_features+[
            math.log(max(self.current_loss, 1e-8)/c['target']),
            math.log(max(self.previous_loss,1e-8)/max(self.current_loss,1e-8)),
            max(0.0, (self.budget-self.work)/self.budget),
            self.steps/c['max_steps'], self.last_multiplier,
            self.forgetting/c['reward_scale'],
            (float(self.old_losses.mean())-self.prev_old_mean)/c['reward_scale']]
        x = torch.tensor(values, dtype=torch.float32).clamp(-10,10)
        if not feedback:
            x[FEEDBACK_SLICE] = 0
        return x

    def potential(self):
        if self.done:
            return 0.0
        gap = max(math.log(max(self.current_loss,1e-8)/self.c['target']),0.0)
        return -gap-min(self.forgetting/self.c['reward_scale'],self.c['forgetting_cap'])

    def terminal_reward(self):
        if self.success:
            return -min(self.forgetting/self.c['reward_scale'], self.c['forgetting_cap'])
        gap = max(math.log(max(self.current_loss,1e-8)/self.c['target']),0.0)
        return -self.c['failure_penalty']-min(gap,5.0)

    def advance(self, action):
        if self.done:
            raise RuntimeError('Cannot advance a terminal environment')
        c = self.c
        multiplier = ACTIONS[int(action)]
        alpha = torch.where(self.mask, multiplier, 1.0)
        before_potential = self.potential()
        self.previous_loss = self.current_loss
        self.prev_old_mean = float(self.old_losses.mean())
        self.last_multiplier = multiplier
        self.work += self.policy_charge
        before_steps = self.steps
        for _ in range(c['interval']):
            # Reserve the old-monitor measurement, which happens at interval end.
            if (self.steps >= c['max_steps'] or
                    self.work+self.step_charge+self.old_n > self.budget+1e-8):
                self.reason = 'budget'
                break
            ni, oi = self.new_indices[self.steps], self.old_indices[self.steps]
            ln = core.loss(self.w,(self.new[0][ni],self.new[1][ni]),self.router,c)
            lo = core.loss(self.w,(self.old[0][oi],self.old[1][oi]),self.router,c)
            g = torch.autograd.grad(ln+c['replay_weight']*lo,self.w)[0]
            self.steps += 1
            self.adam_m.mul_(.9).add_(g,alpha=.1)
            self.adam_v.mul_(.99).addcmul_(g,g,value=.01)
            mh = self.adam_m/(1-.9**self.steps)
            vh = self.adam_v/(1-.99**self.steps)
            with torch.no_grad():
                # Scale actual Adam displacement, not the raw gradient.
                self.w.add_(-c['lr']*alpha[:,None]*mh/(vh.sqrt()+1e-8))
            self.current_loss = core.mse(self.w,self.new_monitor,self.router,c)
            self.work += self.step_charge
            if self.current_loss <= c['target']:
                self.success, self.reason = True, 'target'
                break
        # There is normally a reserved measurement. If no step fits, retain the
        # last observed loss without charging an impossible additional probe.
        if self.steps > before_steps:
            self.old_losses = np.array([core.mse(self.w,d,self.router,c) for d in self.monitors])
            self.work += self.old_n
            self.forgetting = float(np.maximum(self.old_losses-self.old_refs,0).mean())
        self.done = (self.success or self.steps >= c['max_steps'] or
                     self.work+self.step_charge+self.old_n+self.policy_charge > self.budget+1e-8)
        if self.done and not self.success:
            self.reason = 'budget'
        base = self.terminal_reward() if self.done else 0.0
        reward = base+self.potential()-before_potential
        self.events.append(dict(step=self.steps, multiplier=multiplier,
            new_monitor_loss=self.current_loss, old_monitor_forgetting=self.forgetting,
            old_monitor_mean=float(self.old_losses.mean()), work=self.work))
        return reward


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.Tanh(),
                                   nn.Linear(32,32),nn.Tanh())
        self.actor = nn.Linear(32,len(ACTIONS))
        self.critic = nn.Linear(32,1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self,x):
        z = self.trunk(x)
        return torch.distributions.Categorical(logits=self.actor(z)),self.critic(z).squeeze(-1)

    @staticmethod
    def macs():
        # Includes critic evaluation even at deployment because forward computes it.
        return len(FEATURE_NAMES)*32+32*32+32*(len(ACTIONS)+1)


def rule_action(env, kind, cfg):
    if kind == 'fixed':
        return cfg['action']
    if kind == 'schedule':
        initial_gap = max(math.log(max(env.initial_loss,1e-8)/env.c['target']),1e-6)
        gap = max(math.log(max(env.current_loss,1e-8)/env.c['target']),0.0)
        progress = 1-gap/initial_gap
        return cfg['late'] if progress >= cfg['switch'] else cfg['early']
    if kind == 'rule':
        remaining = (env.budget-env.work)/env.budget
        if remaining < cfg['release_budget']:
            return len(ACTIONS)-1
        return cfg['protected_action'] if env.forgetting > cfg['tolerance'] else len(ACTIONS)-1
    raise ValueError(kind)


def rollout(item,c,kind,cfg=None,policy=None,feedback=True,train=False,trace=False):
    start = time.perf_counter()
    env = ScalarEnv(item,c,Policy.macs() if policy is not None else 0)
    initial_potential = env.potential()
    trajectory = []
    while not env.done:
        x = env.observe(feedback)
        if policy is None:
            action = rule_action(env,kind,cfg)
        else:
            with torch.no_grad():
                dist,value = policy(x)
                action = int(dist.sample()) if train else int(dist.logits.argmax())
                logp = float(dist.log_prob(torch.tensor(action)))
        reward = env.advance(action)
        if train:
            trajectory.append(dict(x=x,action=action,value=float(value),logp=logp,reward=reward))
    result = dict(seed=env.world['seed'],old_tasks=env.task,success=env.success,
        steps=env.steps,work=env.work,budget=env.budget,decisions=len(env.events),
        new_monitor_loss=env.current_loss,initial_new_loss=env.initial_loss,
        monitor_forgetting=env.forgetting,terminal_reward=env.terminal_reward(),
        reason=env.reason,protected_mass=env.protected_mass,mask=env.mask.tolist(),
        initial_old_usage=env.ra.tolist(),initial_new_usage=env.rb.tolist(),
        mean_multiplier=float(np.mean([e['multiplier'] for e in env.events])) if env.events else 1.0,
        action_changes=sum(a['multiplier'] != b['multiplier'] for a,b in zip(env.events,env.events[1:])),
        seconds=time.perf_counter()-start)
    if trace:
        result['events'] = env.events
    if 'test' in env.world['tasks'][0]:
        before = np.array([core.mse(env.initial_w,env.world['tasks'][i]['test'],env.router,c)
                           for i in range(env.task)])
        after = np.array([core.mse(env.w,env.world['tasks'][i]['test'],env.router,c)
                          for i in range(env.task)])
        result.update(test_forgetting=float(np.maximum(after-before,0).mean()),
            test_old_signed_change=float((after-before).mean()),test_old_loss=float(after.mean()),
            test_old_initial_loss=float(before.mean()),
            new_test_loss=core.mse(env.w,env.world['tasks'][env.task]['test'],env.router,c))
    return result,trajectory,env


def eval_bank(bank,c,kind,cfg=None,policy=None,feedback=True,trace=False):
    return [rollout(item,c,kind,cfg,policy,feedback,trace=trace)[0] for item in bank]


def selection_key(rows):
    ok = [r for r in rows if r['success']]
    return (-len(ok),float(np.mean([r['monitor_forgetting'] for r in ok])) if ok else 1e9,
            float(np.mean([r['work'] for r in rows])))


def brief(rows):
    ok = [r for r in rows if r['success']]
    return dict(reached=len(ok),total=len(rows),
        forgetting_success=float(np.mean([r['monitor_forgetting'] for r in ok])) if ok else None,
        mean_steps=float(np.mean([r['steps'] for r in rows])),
        mean_work=float(np.mean([r['work'] for r in rows])))


def tune(out,bank,c):
    start = time.time()
    grids = {
        'fixed':[dict(action=a) for a in range(len(ACTIONS))],
        'schedule':[dict(early=a,late=b,switch=s) for a in range(len(ACTIONS))
                    for b in range(len(ACTIONS)) for s in [.33,.5,.67]],
        'rule':[dict(protected_action=a,tolerance=t,release_budget=b)
                for a in range(4) for t in [.001,.003,.01,.03] for b in [0,.25,.5]]}
    all_results,selected = [],{}
    for kind,grid in grids.items():
        best = None
        for cfg in grid:
            rows = eval_bank(bank,c,kind,cfg)
            all_results.append(dict(kind=kind,config=cfg,summary=brief(rows),rows=rows))
            key = selection_key(rows)
            if best is None or key < best[0]:
                best = key,cfg
        selected[kind] = best[1]
        save(out/'baseline_candidates.json',all_results)
        save(out/'selected_baselines.json',selected)
        print(json.dumps(dict(stage='tune',kind=kind,config=best[1],key=best[0],
                              elapsed=time.time()-start)),flush=True)
    save(out/'tuning_cost.json',dict(seconds=time.time()-start,
        candidates={k:len(v) for k,v in grids.items()},
        model_work=sum(r['work'] for a in all_results for r in a['rows'])))
    return selected


def train_policy(out,bank,dev,c,seed,feedback,iterations=60,episodes=12):
    mode = 'feedback' if feedback else 'progress'
    name = f'{mode}_{seed}'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(),lr=7e-4)
    best_key,history = None,[]
    start = time.time()
    total_work = 0
    training_episodes = 0
    total_transitions = 0
    for iteration in range(iterations+1):
        if iteration%10 == 0 or iteration == iterations:
            validation = eval_bank(dev,c,'policy',policy=policy,feedback=feedback)
            total_work += sum(r['work'] for r in validation)
            key = selection_key(validation)
            record = dict(iteration=iteration,**brief(validation),elapsed=time.time()-start)
            history.append(record)
            if best_key is None or key < best_key:
                best_key = key
                torch.save(dict(state=policy.state_dict(),config=c,seed=seed,feedback=feedback,
                    iteration=iteration,validation=record),out/f'{name}.pt')
            save(out/f'{name}_history.json',history)
            print(json.dumps(dict(stage='validation',policy=name,**record)),flush=True)
        if iteration == iterations:
            break
        batch = []
        for index in rng.integers(len(bank),size=episodes):
            row,trajectory,_ = rollout(bank[index],c,'policy',policy=policy,
                                       feedback=feedback,train=True)
            total_work += row['work']
            training_episodes += 1
            advantage = 0.0
            for j in reversed(range(len(trajectory))):
                nxt = trajectory[j+1]['value'] if j+1 < len(trajectory) else 0.0
                delta = trajectory[j]['reward']+nxt-trajectory[j]['value']
                advantage = delta+.95*advantage
                trajectory[j]['advantage'] = advantage
                trajectory[j]['return'] = advantage+trajectory[j]['value']
            batch.extend(trajectory)
        if not batch:
            continue
        total_transitions += len(batch)
        x = torch.stack([r['x'] for r in batch])
        actions = torch.tensor([r['action'] for r in batch])
        old_logp = torch.tensor([r['logp'] for r in batch])
        adv = torch.tensor([r['advantage'] for r in batch])
        returns = torch.tensor([r['return'] for r in batch])
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
        for _ in range(4):
            for ids in torch.randperm(len(batch)).split(128):
                dist,value = policy(x[ids])
                ratio = (dist.log_prob(actions[ids])-old_logp[ids]).exp()
                actor = -torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                objective = actor+.25*(value-returns[ids]).square().mean()-.01*dist.entropy().mean()
                optimizer.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),1.0)
                optimizer.step()
    save(out/f'{name}_cost.json',dict(seconds=time.time()-start,model_work=total_work,
        episodes=training_episodes,transitions=total_transitions,iterations=iterations,
        policy_parameters=sum(p.numel() for p in policy.parameters())))


def load_policy(path):
    checkpoint = torch.load(path,weights_only=False)
    policy = Policy()
    policy.load_state_dict(checkpoint['state'])
    policy.eval()
    return policy,checkpoint


def evaluate(out,c,selected,seeds=TEST_SEEDS,policy_seeds=POLICY_SEEDS):
    if (out/'locked_protocol.json').exists():
        raise RuntimeError('Test protocol already sealed; use a new output directory.')
    models = {}
    for mode in ['progress','feedback']:
        for s in policy_seeds:
            name = f'{mode}_{s}'
            policy,checkpoint = load_policy(out/f'{name}.pt')
            assert checkpoint['config'] == c
            models[name] = policy
    protocol = dict(config=c,actions=ACTIONS,features=FEATURE_NAMES,
        train_seeds=TRAIN_SEEDS,dev_seeds=DEV_SEEDS,test_seeds=list(seeds),
        policy_seeds=policy_seeds,selected_baselines=selected,
        primary_comparison='feedback PPO minus progress-only PPO, paired training seed and history',
        secondary_comparisons=['fixed','schedule','rule','full_update'],
        success_rule='Report reachability first; no positive claim from selected successful cases alone.',
        forgetting='Mean positive old-test MSE increase relative to start of B, per old task.',
        endpoint='First new monitor MSE <= target; actual endpoint loss and independent test loss reported.',
        mask='Frozen at B start; old usage >= K/E and B usage >= .05; no Fisher.',
        workload='Forward-example equivalents; backward charged 2; policy affine MACs charged. Not hardware FLOPs.',
        inference='Paired history bootstrap; average 3 training replicas within history. Also two-way bootstrap.',
        task_scope='Repeated control during one B adaptation after 1--3 standard-trained old tasks.',
        pilot_seeds=list(range(220000,220012)),
        calibration='Pilot full-update reach: .06 = 9/12, .08 = 12/12, .10 = 12/12; choose .08 before training/test.',
        code_hashes={p.name:checksum(p) for p in Path(__file__).parent.glob('*.py')},
        policy_hashes={name:checksum(out/f'{name}.pt') for name in models})
    # This file is written BEFORE generating ANY held-out world or test labels.
    save(out/'locked_protocol.json',protocol)
    all_rows = []
    start = time.time()
    for index,seed in enumerate(seeds):
        item = make_bank([seed],c,test=True)[0]
        for kind in ['fixed','schedule','rule','full_update']:
            actual = 'fixed' if kind == 'full_update' else kind
            cfg = dict(action=4) if kind == 'full_update' else selected[kind]
            row,_,_ = rollout(item,c,actual,cfg,trace=True)
            all_rows.append(dict(method=kind,**row))
        for name,policy in models.items():
            row,_,_ = rollout(item,c,'policy',policy=policy,
                              feedback=name.startswith('feedback'),trace=True)
            all_rows.append(dict(method=name,**row))
        if index%10 == 9 or index == len(seeds)-1:
            save(out/'test_results.json',all_rows)
            print(json.dumps(dict(stage='test',histories=index+1,elapsed=time.time()-start)),flush=True)
    save(out/'evaluation_cost.json',dict(seconds=time.time()-start,histories=len(seeds)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['pilot','run'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=60)
    parser.add_argument('--episodes',type=int,default=12)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Refusing to overwrite results; use a fresh output directory.')
    args.out.mkdir(parents=True,exist_ok=True)
    c = copy.deepcopy(CONFIG)
    save(args.out/'config.json',c)
    if args.stage == 'pilot':
        bank = make_bank(range(220000,220012),c)
        rows = {str(m):eval_bank(bank,c,'fixed',dict(action=i)) for i,m in enumerate(ACTIONS)}
        save(args.out/'pilot.json',rows)
        print(json.dumps({k:brief(v) for k,v in rows.items()},indent=2),flush=True)
        return
    save(args.out/'plan.json',dict(config=c,train=TRAIN_SEEDS,dev=DEV_SEEDS,test=TEST_SEEDS,
        policy_seeds=POLICY_SEEDS,iterations=args.iterations,episodes=args.episodes,
        primary='feedback vs progress-only PPO; 3 matched policy initializations'))
    dev = make_bank(DEV_SEEDS,c)
    selected = tune(args.out,dev,c)
    bank = make_bank(TRAIN_SEEDS,c)
    for seed in POLICY_SEEDS:
        for feedback in [False,True]:
            train_policy(args.out,bank,dev,c,seed,feedback,args.iterations,args.episodes)
    evaluate(args.out,c,selected)


if __name__ == '__main__':
    main()

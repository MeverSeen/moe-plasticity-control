"""All-history adaptation with bounded memory and genuinely fixed sample routing.

Only current-task data and retained memory are used by the learner. Evaluation
is a separate, post-rollout operation over saved endpoint weights. The policy
receives a tensor, never a learner or a dataset handle.
"""
import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import FixedMemory
from model import MultiLayerMoE, ScaledAdam

DEFAULT = dict(tasks=5, experts=8, width=32, expert_hidden=64, top_k=2,
               train_per_class=48, monitor_per_class=16, eval_per_class=32,
               replay_capacity=128, monitor_capacity=128, batch=32,
               max_steps=80, interval=5, lr=.003, replay_weight=.5,
               target_loss=1.35)
FEATURES = ['old_usage', 'old_gate_mass', 'new_usage', 'new_gate_mass',
            'log_old_coverage', 'has_old_observation', 'last_multiplier',
            'new_loss_over_target', 'recent_new_progress', 'remaining_budget',
            'task_fraction', 'cumulative_memory_forgetting', 'recent_old_change']
LEVELS = torch.tensor([0., .5, 1.])


@torch.no_grad()
def measure(model, batch, chunk=128):
    losses, correct = [], []
    for start in range(0, len(batch), chunk):
        logits = model(batch.x[start:start+chunk], batch.task[start:start+chunk])
        losses.append(F.cross_entropy(logits, batch.y[start:start+chunk], reduction='none'))
        correct.append(logits.argmax(1).eq(batch.y[start:start+chunk]))
    return torch.cat(losses), float(torch.cat(correct).float().mean())


class RetentionEnv:
    def __init__(self, history, root, config, capture=False):
        self.history, self.root, self.c = history, int(root), dict(config)
        with torch.random.fork_rng():
            torch.manual_seed(root)
            self.model = MultiLayerMoE(config['tasks'], experts=config['experts'],
                width=config['width'], expert_hidden=config['expert_hidden'],
                top_k=config['top_k'], seed=root)
        self.optimizer = ScaledAdam(self.model, lr=config['lr'])
        self.memory = FixedMemory(config['replay_capacity'], config['monitor_capacity'])
        self.stage = 0
        self.completed, self.events, self.snapshots = [], [], []
        self.capture = capture
        self.cost = dict(learner_updates=0, train_examples=0, monitor_examples=0,
                         audit_examples=0, controller_calls=0, route_examples=0)
        self.last = torch.ones(2, config['experts'])
        self.old_change, self.forgetting = 0., 0.
        self._start()

    @property
    def done(self):
        return self.stage >= len(self.history)

    def clone(self):
        # The immutable experiment harness can share source data; model, Adam,
        # bounded buffers, references, counters and RNG state must be copied.
        return copy.deepcopy(self, {id(self.history): self.history})

    def _start(self):
        self.current = self.history[self.stage]
        self.local_steps = 0
        self.rng = torch.Generator().manual_seed(self.root*100003 + self.stage*1009 + 81)
        self.current_loss = self.initial_loss = self.previous_loss = self._new_loss()
        self.new_usage, self.new_mass = self.model.route_stats(self.current['monitor'].x)
        old, _ = self.memory.monitor()
        if old is None:
            self.old_usage, self.old_mass = torch.zeros_like(self.new_usage), torch.zeros_like(self.new_mass)
            self.old_n = 0
        else:
            self.old_usage, self.old_mass = self.model.route_stats(old.x)
            self.old_n = len(old)
        self.cost['route_examples'] += len(self.current['monitor']) + self.old_n
        self.old_change = 0.
        self._old_loss()

    def _new_loss(self):
        batch = self.current['monitor']
        values, _ = measure(self.model, batch)
        self.cost['monitor_examples'] += len(batch)
        return float(values.mean())

    def _old_loss(self):
        batch, refs = self.memory.monitor()
        if batch is None:
            self.old_mean, self.forgetting = 0., 0.
            return
        values, _ = measure(self.model, batch)
        self.cost['monitor_examples'] += len(batch)
        means, changes = [], []
        for task in batch.task.unique():
            mask = batch.task == task
            means.append(values[mask].mean())
            # Each task uses its own fixed learn-time reference on the samples
            # still retained. Never reset references at a later task boundary.
            changes.append((values[mask].mean()-refs[mask].mean()).clamp_min(0))
        self.old_mean = float(torch.stack(means).mean())
        self.forgetting = float(torch.stack(changes).mean())

    def observe(self):
        shape = self.old_usage.shape
        scalar = lambda value: torch.full(shape, float(value))
        features = [self.old_usage, self.old_mass, self.new_usage, self.new_mass,
            torch.log1p(self.old_usage*self.old_n), (self.old_usage > 0).float(),
            self.last, scalar(self.current_loss/self.c['target_loss']),
            scalar(self.previous_loss-self.current_loss),
            scalar(1-self.local_steps/self.c['max_steps']),
            scalar(self.stage/max(len(self.history)-1, 1)),
            scalar(self.forgetting), scalar(self.old_change)]
        return torch.stack(features, -1).clamp(-10, 10)

    def advance(self, multipliers):
        if self.done:
            raise RuntimeError('terminal environment')
        multipliers = multipliers.detach().clone()
        if multipliers.shape != self.last.shape or not torch.isin(multipliers, LEVELS).all():
            raise ValueError('actions must be [2, experts] with values 0, .5, 1')
        self.cost['controller_calls'] += 1
        self.previous_loss = self.current_loss
        previous_old = self.old_mean
        new, old = self.current['train'], self.memory.replay()
        for _ in range(min(self.c['interval'], self.c['max_steps']-self.local_steps)):
            ni = torch.randint(len(new), (self.c['batch'],), generator=self.rng)
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(new.x[ni], new.task[ni]), new.y[ni])
            self.cost['train_examples'] += len(ni)
            if old is not None:
                oi = torch.randint(len(old), (self.c['batch'],), generator=self.rng)
                loss = loss + self.c['replay_weight'] * F.cross_entropy(
                    self.model(old.x[oi], old.task[oi]), old.y[oi])
                self.cost['train_examples'] += len(oi)
            loss.backward()
            self.optimizer.step(multipliers)
            self.local_steps += 1
            self.cost['learner_updates'] += 1
        self.current_loss = self._new_loss()
        self._old_loss()
        self.old_change = self.old_mean-previous_old
        self.last = multipliers
        if self.capture:
            self.events.append(dict(task=self.stage, steps=self.local_steps,
                new_loss=self.current_loss, memory_forgetting=self.forgetting,
                multipliers=multipliers.tolist()))

    def finish_task(self):
        hit = self.current_loss <= self.c['target_loss']
        if not hit and self.local_steps < self.c['max_steps']:
            raise RuntimeError('task must reach target or exhaust its budget')
        self.completed.append(dict(task=self.stage, initial_loss=self.initial_loss,
            terminal_loss=self.current_loss, success=hit, steps=self.local_steps,
            memory_forgetting=self.forgetting))
        # Preserve this task's per-sample baseline at its own endpoint.
        refs, _ = measure(self.model, self.current['monitor'])
        self.cost['monitor_examples'] += len(refs)
        self.memory.observe_task(self.stage, self.current['train'], self.current['monitor'], refs)
        if self.capture:
            self.snapshots.append(copy.deepcopy(self.model.state_dict()))
        self.stage += 1
        if not self.done:
            self._start()

    def run_task(self, chooser, trace=None):
        while self.current_loss > self.c['target_loss'] and self.local_steps < self.c['max_steps']:
            obs = self.observe()
            action, record = chooser(obs)
            if trace is not None and record is not None:
                trace.append(record)
            self.advance(action)
        self.finish_task()


def run_remaining(env, chooser, trace=None):
    start = time.perf_counter()
    while not env.done:
        # Every method shares the same first-task policy, matching the complete
        # checkpoint prefix used by controller training. Control starts at task2.
        if env.stage == 0:
            env.run_task(lambda obs: (torch.ones_like(obs[..., 0]), None))
        else:
            env.run_task(chooser, trace)
    env.cost['seconds'] = time.perf_counter()-start
    return env


def evaluate_endpoints(env, eval_batches, split, method, policy_seed=None):
    """Offline full-history audit; never consumed by observations or rewards."""
    if not env.done or len(env.snapshots) != len(eval_batches):
        raise ValueError('need a completed captured rollout and one audit batch per task')
    k = len(eval_batches)
    losses, accuracies = [[None]*k for _ in range(k)], [[None]*k for _ in range(k)]
    evaluator = copy.deepcopy(env.model)
    count = 0
    start = time.perf_counter()
    for t, state in enumerate(env.snapshots):
        evaluator.load_state_dict(state)
        for j in range(t+1):
            values, accuracy = measure(evaluator, eval_batches[j])
            losses[t][j], accuracies[t][j] = float(values.mean()), accuracy
            count += len(values)
    cost = dict(env.cost, audit_examples=count, audit_seconds=time.perf_counter()-start)
    return dict(split=split, root=env.root, method=method, policy_seed=policy_seed,
        tasks=env.completed, loss_matrix=losses, accuracy_matrix=accuracies,
        cost=cost, target_loss=env.c['target_loss'], events=env.events,
        classes=[t['classes'] for t in env.history],
        final_memory_counts=dict(replay=len(env.memory.replay()), monitor=len(env.memory.monitor()[0])))

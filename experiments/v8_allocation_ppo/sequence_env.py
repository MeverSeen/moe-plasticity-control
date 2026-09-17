"""Continuous B1 -> B2 -> B3 adaptation, retaining weights and Adam moments.

Original-A replay/monitor memory and forgetting references are fixed throughout.
Prior B data are NOT added to memory; their retention is an offline diagnostic.
"""
import copy
import math

import numpy as np
import torch

import model_core as core

CONFIG = dict(core.DEFAULT, tasks=6, segments=3, segment_steps=100,
              target=.08, interval=5, replay_weight=.1, mask_old_threshold=1/3)
ACTIONS = [(p, u) for u in [1., .5, 1.5] for p in [0., .25, .5, .75, 1.]]
FEATURES = ['protected_B_gate_mass', 'old_usage_protected', 'old_usage_other',
            'protected_fraction', 'B_route_entropy', 'global_initial_A_loss',
            'segment_initial_B_loss', 'current_B_loss', 'recent_B_progress',
            'remaining_segment_steps', 'segment_progress', 'last_multiplier',
            'segment_index', 'global_updates', 'previous_segment_B_loss',
            'previous_segment_hit', 'cumulative_A_forgetting', 'recent_A_loss_change']
F_SCALE = .01
POLICY_MACS = len(FEATURES)*32 + 32*32 + 32*30


def make_item(seed, c, evaluation=True):
    world = core.make_world(seed, c, include_test=evaluation)
    old_count = 1 + seed % 3
    assert old_count + c['segments'] <= len(world['tasks'])
    w, _ = core.pretrain(world, old_count, c)
    return world, old_count, w


class SequenceEnv:
    def __init__(self, item, c):
        self.world, self.old_count, initial = item
        self.c, self.router = c, self.world['router']
        self.w = initial.detach().clone().requires_grad_()
        self.initial_w = initial.detach().clone()
        self.adam_m, self.adam_v = torch.zeros_like(self.w), torch.zeros_like(self.w)
        self.steps, self.stage, self.local_steps = 0, 0, 0
        self.old, self.monitors = core.memory(self.world, self.old_count, c)
        self.old_refs = self.old_losses()
        self.old_current = self.old_refs.copy()
        self.old_delta, self.forgetting = 0., 0.
        self.last_multiplier = 1.
        self.previous_segment_loss, self.previous_segment_hit = c['target'], True
        ids, _ = core.routes(self.old[0], self.router, c)
        self.ra = torch.bincount(ids.flatten(), minlength=c['experts']).float()/len(self.old[0])
        # Fixed original-A protection group; no future B or current-route redesign.
        self.mask = self.ra >= c['mask_old_threshold']
        self.completed, self.events, self.decisions = [], [], []
        self.done = False
        affine = c['top_k']*(c['dim']*c['hidden']+c['hidden'])+2*c['experts']
        self.policy_charge = POLICY_MACS / affine
        self.step_charge = 3*(c['batch']+c['old_batch'])+c['monitor_n']
        self.old_n = sum(len(d[0]) for d in self.monitors)
        self.work = float(self.old_n + len(self.old[0]))
        # Reserve the same maximum observation/controller work for all four arms.
        self.budget = (c['monitor_memory']+c['replay_memory'] + c['segments'] *
            (2*c['monitor_n'] + c['segment_steps']*self.step_charge +
             math.ceil(c['segment_steps']/c['interval'])*(c['monitor_memory']+self.policy_charge)))
        self._start_segment()
        self._skip_initially_solved()

    def old_losses(self):
        return np.array([core.mse(self.w, d, self.router, self.c) for d in self.monitors])

    def _start_segment(self):
        self.task = self.old_count + self.stage
        self.new = self.world['tasks'][self.task]['train']
        self.new_monitor = self.world['tasks'][self.task]['monitor']
        self.local_steps = 0
        self.current_loss = core.mse(self.w, self.new_monitor, self.router, self.c)
        self.initial_loss = self.previous_loss = self.current_loss
        ids, gates = core.routes(self.new_monitor[0], self.router, self.c)
        rb = torch.bincount(ids.flatten(), minlength=self.c['experts']).float()/len(self.new_monitor[0])
        p = rb / rb.sum()
        entropy = float(-(p*p.clamp_min(1e-8).log()).sum()/math.log(self.c['experts']))
        self.context = [float((gates*self.mask[ids]).sum(1).mean()),
            float(self.ra[self.mask].mean()) if self.mask.any() else 0.,
            float(self.ra[~self.mask].mean()) if (~self.mask).any() else 0.,
            float(self.mask.float().mean()), entropy,
            math.log1p(float(self.old_refs.mean())/self.c['target']),
            math.log(max(self.initial_loss, 1e-8)/self.c['target'])]
        self.new_indices, self.old_indices = self._indices(self.stage, 0)
        self.future_rep = 0
        self.work += 2*len(self.new_monitor[0])

    def _indices(self, stage, replicate):
        # Stage-keyed streams: actions never shift which exogenous stream a B receives.
        rng = core.gen(self.world['seed']*199 + (self.old_count+stage)*1331 + 744 + replicate*1000003)
        n = self.c['segment_steps']
        return (torch.randint(len(self.world['tasks'][self.old_count+stage]['train'][0]),
                              (n, self.c['batch']), generator=rng),
                torch.randint(len(self.old[0]), (n, self.c['old_batch']), generator=rng))

    def set_future_rep(self, replicate):
        """Change only future minibatches, identically across candidate actions."""
        self.future_rep = replicate
        new, old = self._indices(self.stage, replicate)
        self.new_indices[self.local_steps:] = new[self.local_steps:]
        self.old_indices[self.local_steps:] = old[self.local_steps:]

    def _finish_segment(self):
        record = dict(stage=self.stage, success=self.current_loss <= self.c['target'],
            initial_B_loss=self.initial_loss, B_loss=self.current_loss, steps=self.local_steps,
            global_steps=self.steps, cumulative_monitor_forgetting=self.forgetting,
            monitor_A_losses=self.old_current.tolist(), work=self.work)
        self.completed.append(record)
        self.previous_segment_loss, self.previous_segment_hit = self.current_loss, record['success']
        rep = self.future_rep
        if self.stage+1 == self.c['segments']:
            self.done = True
        else:
            self.stage += 1
            self._start_segment()
            self.set_future_rep(rep)

    def _skip_initially_solved(self):
        while not self.done and self.current_loss <= self.c['target']:
            self._finish_segment()

    def observe(self, feedback=True):
        x = self.context + [math.log(max(self.current_loss, 1e-8)/self.c['target']),
            math.log(max(self.previous_loss, 1e-8)/max(self.current_loss, 1e-8)),
            1-self.local_steps/self.c['segment_steps'], self.local_steps/self.c['segment_steps'],
            self.last_multiplier, self.stage/max(self.c['segments']-1, 1),
            self.steps/(self.c['segments']*self.c['segment_steps']),
            math.log(max(self.previous_segment_loss, 1e-8)/self.c['target']),
            float(self.previous_segment_hit), self.forgetting/F_SCALE, self.old_delta/F_SCALE]
        x = np.clip(np.asarray(x, dtype=np.float32), -10, 10)
        if not feedback:
            x[-2:] = 0
        return x

    def advance_window(self, action):
        if self.done:
            raise RuntimeError('Cannot advance a terminal sequence')
        if action not in range(len(ACTIONS)):
            raise ValueError(action)
        stage = self.stage
        multiplier, other = ACTIONS[action]
        alpha = torch.where(self.mask, multiplier, other)[:, None]
        before_old = float(self.old_current.mean())
        self.previous_loss, self.last_multiplier = self.current_loss, action/4.
        # Block policies pay the same per-window controller proxy, even when held.
        self.work += self.policy_charge
        for _ in range(min(self.c['interval'], self.c['segment_steps']-self.local_steps)):
            ni, oi = self.new_indices[self.local_steps], self.old_indices[self.local_steps]
            new = self.new[0][ni], self.new[1][ni]
            old = self.old[0][oi], self.old[1][oi]
            loss = core.loss(self.w, new, self.router, self.c)
            loss = loss+self.c['replay_weight']*core.loss(self.w, old, self.router, self.c)
            grad = torch.autograd.grad(loss, self.w)[0]
            self.steps += 1
            self.local_steps += 1
            with torch.no_grad():
                self.adam_m.mul_(.9).add_(grad, alpha=.1)
                self.adam_v.mul_(.99).addcmul_(grad, grad, value=.01)
                mh = self.adam_m/(1-.9**self.steps)
                vh = self.adam_v/(1-.99**self.steps)
                self.w.add_(-self.c['lr']*alpha*mh/(vh.sqrt()+1e-8))
            self.current_loss = core.mse(self.w, self.new_monitor, self.router, self.c)
            self.work += self.step_charge
            if self.current_loss <= self.c['target']:
                break
        self.old_current = self.old_losses()
        self.old_delta = float(self.old_current.mean())-before_old
        self.forgetting = float(np.maximum(self.old_current-self.old_refs, 0).mean())
        self.work += self.old_n
        self.events.append(dict(stage=stage, local_steps=self.local_steps, global_steps=self.steps,
            action=action, multiplier=multiplier, other_multiplier=other, B_loss=self.current_loss,
            monitor_forgetting=self.forgetting, work=self.work))
        if self.current_loss <= self.c['target'] or self.local_steps >= self.c['segment_steps']:
            self._finish_segment()
            self._skip_initially_solved()
        assert self.work <= self.budget + 1e-5

    def act(self, action, frequency):
        if frequency not in ['block', 'within']:
            raise ValueError(frequency)
        self.decisions.append(dict(stage=self.stage, local_steps=self.local_steps,
            global_steps=self.steps, action=action, observation=self.observe().tolist()))
        stage = self.stage
        self.advance_window(action)
        while frequency == 'block' and not self.done and self.stage == stage:
            self.advance_window(action)


def clone_env(env):
    result = copy.copy(env)
    shared = {'world', 'c', 'router', 'new', 'new_monitor', 'old', 'monitors'}
    for name, value in vars(env).items():
        if name in shared:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()
            if name == 'w': value.requires_grad_()
        else:
            value = copy.deepcopy(value)
        setattr(result, name, value)
    return result


def endpoint(env, include_events=True):
    assert env.done
    old_data = [env.world['tasks'][i]['test'] for i in range(env.old_count)]
    refs = np.array([core.mse(env.initial_w, d, env.router, env.c) for d in old_data])
    old = np.array([core.mse(env.w, d, env.router, env.c) for d in old_data])
    new = [core.mse(env.w, env.world['tasks'][env.old_count+k]['test'], env.router, env.c)
           for k in range(env.c['segments'])]
    final_B_monitor = [core.mse(env.w, env.world['tasks'][env.old_count+k]['monitor'], env.router, env.c)
                       for k in range(env.c['segments'])]
    return dict(root=env.world['seed'], success=all(r['success'] for r in env.completed),
        segment_hits=[r['success'] for r in env.completed],
        forgetting=float(np.maximum(old-refs, 0).mean()), old_eval_loss=float(old.mean()),
        old_eval_references=refs.tolist(), steps=env.steps, work=env.work, budget=env.budget,
        mask=env.mask.tolist(), segments=env.completed, final_B_test=new, final_B_monitor=final_B_monitor,
        final_all_B_monitor_hit=all(v <= env.c['target'] for v in final_B_monitor),
        initially_solved=sum(r['steps']==0 for r in env.completed),
        decisions=env.decisions if include_events else [], events=env.events if include_events else [])


def rollout(item, c, chooser, frequency, include_events=False):
    env = SequenceEnv(item, c)
    while not env.done:
        env.act(int(chooser(env)), frequency)
    return endpoint(env, include_events)

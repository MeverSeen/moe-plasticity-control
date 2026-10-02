"""Matched-action baselines and a shared controller with layer/global pooling."""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from environment import FEATURES, LEVELS, RetentionEnv, run_remaining


def rule_action(obs, method):
    old, new = obs[..., 1], obs[..., 3]
    if method.startswith('global_'):
        return torch.full_like(old, float(method.split('_')[1]))
    # These are explicit, inexpensive pilot baselines, not claims of optimal
    # tuned rules. All receive exactly the learned policy's observation tensor.
    if method == 'group_rule':
        return torch.where(old >= 1/old.shape[-1], .5, 1.)
    if method == 'local_rule':
        ratio = new/(old+.05)
        return torch.where(ratio >= 1., 1., torch.where(ratio >= .5, .5, 0.))
    if method == 'coordinated_rule':
        score = new/(old+.05)
        action = torch.full_like(old, .5)
        winners = score.topk(max(1, old.shape[-1]//2), -1).indices
        action.scatter_(1, winners, 1.)
        return action
    raise ValueError(method)


def rule_chooser(method):
    return lambda obs: (rule_action(obs, method), None)


class SharedController(nn.Module):
    def __init__(self, seed=0, coordinated=True):
        super().__init__()
        self.coordinated = coordinated
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.local = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
            self.output = nn.Linear(96 if coordinated else 32, len(LEVELS))
            nn.init.normal_(self.output.weight, std=.01)
            with torch.no_grad():
                self.output.bias.copy_(torch.tensor([-.5, 0., .5]))

    def forward(self, obs):
        local = self.local(obs)
        if self.coordinated:
            layer = local.mean(-2, keepdim=True).expand_as(local)
            global_state = local.mean((-3, -2), keepdim=True).expand_as(local)
            local = torch.cat([local, layer, global_state], -1)
        return self.output(local)

    def chooser(self, sample=False, generator=None):
        @torch.no_grad()
        def choose(obs):
            logits = self(obs)
            if sample:
                action = torch.multinomial(logits.softmax(-1).reshape(-1, 3), 1,
                    generator=generator).reshape(logits.shape[:-1])
            else:
                action = logits.argmax(-1)
            distribution = Categorical(logits=logits)
            record = dict(obs=obs.detach().clone(), action=action,
                          old_logp=distribution.log_prob(action).sum().detach())
            return LEVELS[action], record
        return choose


def train_controller(histories, config, seed, rounds=8, group_size=4, coordinated=True):
    """One on-policy clipped GRPO update per group of real learner trajectories.

    All group members clone the same complete learner+Adam+memory checkpoint
    after the first task. Future batch streams are paired by task and step.
    Actions have a factorized *joint* probability over all layer/expert slots.
    This small pilot uses no critic, reference-policy KL or stale branch labels.
    """
    policy = SharedController(seed, coordinated)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    candidates = [(0, copy.deepcopy(policy.state_dict()))]
    log, cost = [], dict(learner_updates=0, train_examples=0, monitor_examples=0,
                        route_examples=0, controller_calls=0, branches=0,
                        prefix_updates=0, clone_seconds=0., controller_update_seconds=0.)
    import time
    start = time.perf_counter()
    for iteration in range(rounds):
        root, history = histories[iteration % len(histories)]
        prefix = RetentionEnv(history, root, config)
        prefix.run_task(rule_chooser('global_1'))
        for key in ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples', 'controller_calls']:
            cost[key] += prefix.cost[key]
        cost['prefix_updates'] += prefix.cost['learner_updates']
        trajectories, rewards = [], []
        for member in range(group_size):
            copy_start = time.perf_counter()
            env = prefix.clone()
            cost['clone_seconds'] += time.perf_counter()-copy_start
            rng = torch.Generator().manual_seed(seed*1000003+iteration*101+member)
            trace = []
            run_remaining(env, policy.chooser(sample=True, generator=rng), trace)
            # Final old-memory F excludes the just-learned final task. Failed
            # tasks remain in the reward, and finite-buffer F is only a proxy.
            failures = np.mean([not t['success'] for t in env.completed[1:]])
            terminal_loss = np.mean([t['terminal_loss'] for t in env.completed[1:]])
            reward = -5*env.forgetting - 2*failures - .1*terminal_loss
            rewards.append(float(reward))
            trajectories.append(trace)
            for key in ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples', 'controller_calls']:
                cost[key] += env.cost[key]-prefix.cost[key]
            cost['branches'] += 1
        reward_tensor = torch.tensor(rewards)
        std = reward_tensor.std(unbiased=False)
        advantage = (reward_tensor-reward_tensor.mean())/std.clamp_min(1e-4)
        update_start = time.perf_counter()
        terms, entropies = [], []
        for trace, adv in zip(trajectories, advantage):
            if not trace:
                continue
            observations = torch.stack([x['obs'] for x in trace])
            actions = torch.stack([x['action'] for x in trace])
            old_logp = torch.stack([x['old_logp'] for x in trace])
            dist = Categorical(logits=policy(observations))
            joint_logp = dist.log_prob(actions).sum((-2, -1))
            ratio = (joint_logp-old_logp).exp()
            terms.append(-torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean())
            entropies.append(dist.entropy().mean())
        loss_value = None
        if terms:
            loss = torch.stack(terms).mean() - .005*torch.stack(entropies).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            loss_value = float(loss.detach())
        cost['controller_update_seconds'] += time.perf_counter()-update_start
        log.append(dict(round=iteration+1, root=root, rewards=rewards,
            reward_std=float(std), nonempty_trajectories=sum(bool(t) for t in trajectories),
            decision_counts=[len(t) for t in trajectories], loss=loss_value))
        if iteration+1 in {max(1, rounds//2), rounds}:
            candidates.append((iteration+1, copy.deepcopy(policy.state_dict())))
        print(f'controller={seed} coordinated={coordinated} round={iteration+1}/{rounds} reward={reward_tensor.mean():.4f} std={std:.4f}', flush=True)
    cost['seconds'] = time.perf_counter()-start
    return policy, candidates, log, cost

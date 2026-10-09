"""Read-only diagnostics for v8's on-policy GRPO controller update.

This module deliberately reuses the frozen v8 learner and policy.  Its update
expression is kept identical to ``control.train_controller``; diagnostics are
computed with ``autograd.grad`` before the ordinary backward pass and never
enter the reward, action choice, or optimizer.
"""

from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.distributions import Categorical

V8 = Path(__file__).resolve().parents[1] / "v8_multilayer_retention"
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from control import SharedController, rule_chooser  # noqa: E402
from environment import RetentionEnv, run_remaining  # noqa: E402


def _cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _rng_state():
    return (copy.deepcopy(random.getstate()), copy.deepcopy(np.random.get_state()),
            torch.random.get_rng_state().clone())


def _restore_rng(state):
    random.setstate(copy.deepcopy(state[0]))
    np.random.set_state(copy.deepcopy(state[1]))
    torch.random.set_rng_state(state[2])


def _norm_and_cosine(left, right):
    left = [value.reshape(-1) for value in left if value is not None]
    right = [value.reshape(-1) for value in right if value is not None]
    if not left or not right:
        return 0.0, 0.0, None
    a, b = torch.cat(left), torch.cat(right)
    an, bn = float(a.norm()), float(b.norm())
    cosine = None if not an or not bn else float(torch.dot(a, b) / (a.norm() * b.norm()))
    return an, bn, cosine


def _distribution_summary(policy, observations):
    """Summarize positions from traces without introducing sampling or gradients."""
    if not observations:
        return dict(positions=0, mean_probabilities=[0.0, 0.0, 0.0],
                    action_frequencies=[0.0, 0.0, 0.0], entropy=0.0,
                    all_1_joint_probability=0.0, all_1_logit_margin=0.0)
    with torch.no_grad():
        raw_logits = [policy(item) for item in observations]
        logits = torch.cat([item.reshape(-1, 3) for item in raw_logits])
        probabilities = logits.softmax(-1)
        argmax = probabilities.argmax(-1)
        frequencies = torch.bincount(argmax, minlength=3).float() / len(argmax)
        # One joint probability per controller decision, not a product across
        # all decisions/trajectories (which would underflow and answer a
        # different question).  Each decision factorizes over its 2×experts
        # action slots exactly as v8's joint log probability does.
        joint_all_1 = torch.cat([item.softmax(-1)[..., 2].flatten(-2).prod(dim=-1).reshape(-1)
                                 for item in raw_logits])
        margin = logits[:, 2] - logits[:, :2].max(-1).values
        return dict(positions=int(len(argmax)), mean_probabilities=probabilities.mean(0).tolist(),
                    action_frequencies=frequencies.tolist(), entropy=float(Categorical(logits=logits).entropy().mean()),
                    all_1_joint_probability_mean=float(joint_all_1.mean()), all_1_logit_margin=float(margin.mean()))


def _sampled_action_frequencies(trajectories):
    actions = [entry['action'].reshape(-1) for trace in trajectories for entry in trace]
    if not actions:
        return [0.0, 0.0, 0.0]
    values = torch.cat(actions)
    return (torch.bincount(values, minlength=3).float() / len(values)).tolist()


def _probe_summary(policy, probes, initial=None):
    rows = []
    per_probe = []
    with torch.no_grad():
        for probe in probes:
            raw_logits = policy(probe)
            raw_probabilities = raw_logits.softmax(-1)
            raw_argmax = raw_probabilities.argmax(-1)
            logits = raw_logits.reshape(-1, 3)
            probabilities = raw_probabilities.reshape(-1, 3)
            argmax = raw_argmax.reshape(-1)
            rows.append((probabilities, argmax))
            # Keep every expert/layer slot: aggregate summaries alone can hide
            # probability drift that does not alter the deterministic action.
            per_probe.append(dict(logits=raw_logits.detach().cpu().tolist(),
                                  probabilities=raw_probabilities.detach().cpu().tolist(),
                                  argmax=raw_argmax.detach().cpu().tolist()))
    if not rows:
        return dict(probes=0, positions=0, mean_probabilities=[0.0, 0.0, 0.0],
                    argmax_frequencies=[0.0, 0.0, 0.0], max1_minus_second=0.0,
                    kl_from_initial=0.0, argmax_changed_fraction=0.0)
    probabilities = torch.cat([row[0] for row in rows])
    argmax = torch.cat([row[1] for row in rows])
    top2 = probabilities.topk(2, dim=-1).values
    result = dict(probes=len(probes), positions=int(len(argmax)),
                  mean_probabilities=probabilities.mean(0).tolist(),
                  argmax_frequencies=(torch.bincount(argmax, minlength=3).float() / len(argmax)).tolist(),
                  max1_minus_second=float((top2[:, 0] - top2[:, 1]).mean()),
                  kl_from_initial=0.0, argmax_changed_fraction=0.0,
                  per_probe=per_probe)
    if initial is not None:
        initial_probs, initial_argmax = initial
        result['kl_from_initial'] = float((initial_probs * (initial_probs.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())).sum(-1).mean())
        result['argmax_changed_fraction'] = float((argmax != initial_argmax).float().mean())
    return result, (probabilities.detach().clone(), argmax.detach().clone())


def _callback(callback, epoch, policy, optimizer):
    if callback is None:
        return
    rng = _rng_state()
    state, optimizer_state, training = _cpu_state(policy), copy.deepcopy(optimizer.state_dict()), policy.training
    try:
        callback(epoch, policy)
    finally:
        policy.load_state_dict(state)
        optimizer.load_state_dict(optimizer_state)
        policy.train(training)
        _restore_rng(rng)


def _reward_parts(env):
    suffix = env.completed[1:]
    if not suffix:  # Supports intentionally empty traces without NaN rewards.
        failures = terminal = 0.0
    else:
        failures = float(np.mean([not task['success'] for task in suffix]))
        terminal = float(np.mean([task['terminal_loss'] for task in suffix]))
    forgetting = float(env.forgetting)
    parts = dict(forgetting=-5 * forgetting, failures=-2 * failures, terminal_loss=-.1 * terminal)
    return parts, float(sum(parts.values()))


def _objective(policy, trajectories, advantage):
    terms, entropies, ratios = [], [], []
    trace_observations = []
    for trace, adv in zip(trajectories, advantage):
        if not trace:
            continue
        observations = torch.stack([entry['obs'] for entry in trace])
        actions = torch.stack([entry['action'] for entry in trace])
        old_logp = torch.stack([entry['old_logp'] for entry in trace])
        distribution = Categorical(logits=policy(observations))
        joint_logp = distribution.log_prob(actions).sum((-2, -1))
        ratio = (joint_logp-old_logp).exp()
        terms.append(-torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean())
        entropies.append(distribution.entropy().mean())
        ratios.append(ratio)
        trace_observations.append(observations)
    if not terms:
        return None, None, None, [], trace_observations
    policy_loss = torch.stack(terms).mean()
    entropy_loss = -.005*torch.stack(entropies).mean()
    return policy_loss, entropy_loss, torch.cat(ratios), terms, trace_observations


def train_diagnostic(histories, config, seed, coordinated, rounds, checkpoints,
                     probe_observations, on_checkpoint=None):
    """Run v8's single-update GRPO loop with read-only, checkpointed diagnostics.

    ``checkpoints`` is an explicit epoch collection (including optional epoch 0);
    it does not select a model.  ``probe_observations`` are fixed tensors used
    only under ``no_grad`` and never enter learner rollouts or optimization.
    """
    if not histories:
        raise ValueError('histories must be non-empty')
    if rounds < 0:
        raise ValueError('rounds must be non-negative')
    checkpoints = set(int(epoch) for epoch in checkpoints)
    if any(epoch < 0 for epoch in checkpoints):
        raise ValueError('checkpoints must be non-negative')
    probes = [probe.detach().clone() for probe in probe_observations]
    policy = SharedController(seed, coordinated)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    candidates, logs = [], []
    cost = dict(learner_updates=0, train_examples=0, monitor_examples=0,
                route_examples=0, controller_calls=0, branches=0,
                prefix_updates=0, clone_seconds=0., controller_update_seconds=0.,
                diagnostic_seconds=0.)
    initial_probe_payload = _probe_summary(policy, probes)
    initial_probe = initial_probe_payload[1] if probes else None

    def checkpoint(epoch):
        if epoch not in checkpoints:
            return
        started = time.perf_counter()
        candidates.append((epoch, _cpu_state(policy)))
        _callback(on_checkpoint, epoch, policy, optimizer)
        cost['diagnostic_seconds'] += time.perf_counter() - started

    checkpoint(0)
    start = time.perf_counter()
    for iteration in range(rounds):
        root, history = histories[iteration % len(histories)]
        prefix = RetentionEnv(history, root, config)
        prefix.run_task(rule_chooser('global_1'))
        for key in ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples', 'controller_calls']:
            cost[key] += prefix.cost[key]
        cost['prefix_updates'] += prefix.cost['learner_updates']
        trajectories, rewards, reward_parts = [], [], []
        for member in range(4):
            copy_start = time.perf_counter()
            env = prefix.clone()
            cost['clone_seconds'] += time.perf_counter()-copy_start
            rng = torch.Generator().manual_seed(seed*1000003+iteration*101+member)
            trace = []
            run_remaining(env, policy.chooser(sample=True, generator=rng), trace)
            parts, reward = _reward_parts(env)
            rewards.append(reward)
            reward_parts.append(parts)
            trajectories.append(trace)
            for key in ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples', 'controller_calls']:
                cost[key] += env.cost[key]-prefix.cost[key]
            cost['branches'] += 1
        reward_tensor = torch.tensor(rewards)
        std = reward_tensor.std(unbiased=False)
        advantage = (reward_tensor-reward_tensor.mean())/std.clamp_min(1e-4)
        update_start = time.perf_counter()
        policy_loss, entropy_loss, ratios, terms, trace_observations = _objective(policy, trajectories, advantage)
        diagnostics = {}
        loss_value = None
        if policy_loss is not None:
            params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
            gradient_diagnostic_start = time.perf_counter()
            pg_grad = torch.autograd.grad(policy_loss, params, retain_graph=True, allow_unused=True)
            entropy_grad = torch.autograd.grad(entropy_loss, params, retain_graph=True, allow_unused=True)
            pg_norm, entropy_norm, cosine = _norm_and_cosine(pg_grad, entropy_grad)
            cost['diagnostic_seconds'] += time.perf_counter()-gradient_diagnostic_start
            loss = policy_loss + entropy_loss
            before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters() if parameter.requires_grad}
            optimizer.zero_grad()
            loss.backward()
            total_pre = float(torch.sqrt(sum((parameter.grad.detach().float().square().sum() for parameter in params if parameter.grad is not None), torch.tensor(0.))))
            reported_pre = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            total_post = float(torch.sqrt(sum((parameter.grad.detach().float().square().sum() for parameter in params if parameter.grad is not None), torch.tensor(0.))))
            optimizer.step()
            change = float(torch.sqrt(sum(((parameter.detach()-before[name]).float().square().sum() for name, parameter in policy.named_parameters() if parameter.requires_grad), torch.tensor(0.))))
            bias_change = float((policy.output.bias.detach()-before['output.bias']).float().norm())
            diagnostics.update(policy_gradient_grad_norm=pg_norm, entropy_grad_norm=entropy_norm,
                               policy_entropy_grad_cosine=cosine, total_grad_norm_pre_clip=total_pre,
                               clip_reported_pre_norm=reported_pre, total_grad_norm_post_clip=total_post,
                               parameter_change_l2=change, parameter_delta_l2=change,
                               output_bias_delta_l2=bias_change, policy_loss=float(policy_loss.detach()),
                               entropy_loss=float(entropy_loss.detach()))
            loss_value = float(loss.detach())
        else:
            diagnostics.update(policy_gradient_grad_norm=0., entropy_grad_norm=0., policy_entropy_grad_cosine=None,
                               total_grad_norm_pre_clip=0., clip_reported_pre_norm=0., total_grad_norm_post_clip=0.,
                               parameter_change_l2=0., parameter_delta_l2=0., output_bias_delta_l2=0.,
                               policy_loss=None, entropy_loss=None)
        cost['controller_update_seconds'] += time.perf_counter()-update_start
        diagnostic_start = time.perf_counter()
        trace_summary = _distribution_summary(policy, trace_observations)
        trace_summary['sampled_action_frequencies'] = _sampled_action_frequencies(trajectories)
        if ratios is None:
            ratio_summary = dict(count=0, minimum=None, maximum=None, clip_fraction=0.)
        else:
            ratio_values = ratios.detach()
            ratio_summary = dict(count=int(ratio_values.numel()), minimum=float(ratio_values.min()), maximum=float(ratio_values.max()),
                                 clip_fraction=float(((ratio_values < .8) | (ratio_values > 1.2)).float().mean()))
        current_probe_payload = _probe_summary(policy, probes, initial_probe) if probes else _probe_summary(policy, probes)
        probe_summary = current_probe_payload[0] if probes else current_probe_payload
        cost['diagnostic_seconds'] += time.perf_counter()-diagnostic_start
        logs.append(dict(round=iteration+1, root=root, rewards=rewards, reward_parts=reward_parts,
                         reward_std=float(std), standardized_advantages=advantage.tolist(),
                         nonempty_trajectories=sum(bool(trace) for trace in trajectories),
                         decision_counts=[len(trace) for trace in trajectories], loss=loss_value,
                         action_policy=trace_summary, importance_ratio=ratio_summary,
                         probe=probe_summary, **diagnostics))
        checkpoint(iteration+1)
    cost['seconds'] = time.perf_counter()-start
    return policy, candidates, logs, cost

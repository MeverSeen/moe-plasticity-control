"""CPU H=1 toy: forgetting / total motion / protected-overlap motion.

Frozen v2 learner/data reused; every arm observes the same per-update monitor
stream. No mutation of the running language-model experiment or old archives.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'v2_feedback_control'
sys.path.insert(0, str(V2))
import experiment as old

torch.set_num_threads(1)
ARMS = [(1, r) for r in ('forgetting', 'movement', 'overlap')]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(item, config, hold, reward_kind, movement_scale, policy=None,
            action=2, train=False, keep_trace=False, rule=False):
    started = time.perf_counter()
    # Use one-step native updates/monitoring in BOTH H arms. The control hold is
    # implemented outside the learner, so its budget never changes with H.
    env = old.ScalarEnv(item, dict(config, interval=1), policy_macs=0)
    trajectory, detailed = [], []
    path_energy = 0.
    overlap_energy, other_energy = 0., 0.
    expert_energy = torch.zeros(config['experts'], dtype=torch.float64)
    controller_seconds = 0.
    decisions = []
    while not env.done:
        obs = env.observe(True)
        start = time.perf_counter()
        if policy is not None:
            with torch.no_grad():
                distribution, value = policy(obs)
                selected = int(distribution.sample()) if train else int(distribution.logits.argmax())
                logp = float(distribution.log_prob(torch.tensor(selected)))
        elif rule:
            selected = 4 if env.steps >= int(.8 * config['max_steps']) or env.forgetting <= .003 else 0
        else:
            selected = action
        controller_seconds += time.perf_counter() - start
        decisions.append(selected)
        before_f, before_d, before_step = env.forgetting, path_energy, env.steps
        before_overlap = overlap_energy
        for _ in range(hold):
            if env.done: break
            before = env.w.detach().clone()
            env.advance(selected)
            step_sq = (env.w.detach().double() - before.double()).square()
            delta = step_sq.sum()
            path_energy += float(delta)
            per_expert = step_sq.sum(dim=1)
            expert_energy += per_expert
            overlap_delta = float(step_sq[env.mask].sum())
            other_delta = float(step_sq[~env.mask].sum())
            overlap_energy += overlap_delta
            other_energy += other_delta
            if keep_trace:
                detailed.append({**env.events[-1], 'actual_update_sq': float(delta),
                                 'cumulative_update_sq': path_energy,
                                 'expert_update_sq': per_expert.tolist(),
                                 'overlap_update_sq': overlap_delta,
                                 'other_update_sq': other_delta,
                                 'cumulative_overlap_update_sq': overlap_energy,
                                 'cumulative_other_update_sq': other_energy,
                                 'net_displacement_sq': float((env.w.detach().double() - env.initial_w.double()).square().sum())})
        duration = env.steps - before_step
        assert duration > 0
        if reward_kind == 'forgetting':
            reward = -(env.forgetting - before_f) / config['reward_scale']
        elif reward_kind == 'movement':
            reward = -(path_energy - before_d) / movement_scale
        elif reward_kind == 'overlap':
            reward = -(overlap_energy - before_overlap) / movement_scale
        else:
            raise ValueError(reward_kind)
        if env.done and not env.success: reward -= 10.
        if policy is not None:
            trajectory.append(dict(x=obs, action=selected, logp=logp, value=float(value),
                                   reward=reward, duration=duration))
    cost = {'forgetting': env.forgetting / config['reward_scale'],
            'movement': path_energy / movement_scale,
            'overlap': overlap_energy / movement_scale}[reward_kind]
    terminal_reward = -cost - 10. * (not env.success)
    if trajectory:
        assert abs(sum(t['reward'] for t in trajectory) - terminal_reward) < 1e-8
    policy_macs = old.Policy.macs() * len(decisions) if policy is not None else 0
    row = dict(root=item[0]['seed'], success=bool(env.success), steps=env.steps,
               decisions=len(decisions), monitor_F=env.forgetting, new_monitor_loss=env.current_loss,
               initial_new_loss=env.initial_loss, path_energy=path_energy,
               overlap_path_energy=overlap_energy, other_path_energy=other_energy,
               expert_path_energy=expert_energy.tolist(), protected_mask=env.mask.tolist(),
               overlap_net_displacement_sq=float((env.w.detach().double() - env.initial_w.double())[env.mask].square().sum()),
               other_net_displacement_sq=float((env.w.detach().double() - env.initial_w.double())[~env.mask].square().sum()),
               net_displacement_sq=float((env.w.detach().double() - env.initial_w.double()).square().sum()),
               terminal_reward=terminal_reward, learner_monitor_work=env.work,
               policy_macs=policy_macs, controller_seconds=controller_seconds,
               seconds=time.perf_counter() - started,
               mean_multiplier=float(np.mean([old.ACTIONS[x] for x in decisions])) if decisions else None,
               switches=sum(a != b for a, b in zip(decisions, decisions[1:])), actions=decisions,
               protected_experts=int(env.mask.sum()), protected_new_gate_mass=env.protected_mass)
    if keep_trace: row['events'] = detailed
    if 'test' in env.world['tasks'][0]:
        initial = np.array([old.core.mse(env.initial_w, env.world['tasks'][i]['test'], env.router, config)
                            for i in range(env.task)])
        final = np.array([old.core.mse(env.w, env.world['tasks'][i]['test'], env.router, config)
                          for i in range(env.task)])
        row.update(test_F=float(np.maximum(final - initial, 0).mean()),
                   test_old_loss=float(final.mean()), test_old_initial=float(initial.mean()),
                   new_test_loss=old.core.mse(env.w, env.world['tasks'][env.task]['test'], env.router, config))
    return row, trajectory, env.w.detach().clone()


def select_key(rows):
    success = [r for r in rows if r['success']]
    return [-len(success), float(np.mean([r['monitor_F'] for r in success])) if success else 1e9,
            float(np.mean([r['steps'] for r in rows]))]


def train_policy(out, bank, dev, config, hold, reward, scale, seed, rounds, episodes):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = old.Policy()
    initial = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=7e-4)
    name = f'h{hold}_{reward}_{seed}'
    best_key, selected_round = None, None
    learning, rows, transitions, updates = [], [], 0, 0
    begin = time.time()
    with gzip.open(out / f'{name}_training.jsonl.gz', 'wt') as archive:
        for iteration in range(rounds + 1):
            if iteration in (0, rounds // 2, rounds):
                validation = [rollout(x, config, hold, reward, scale, policy)[0] for x in dev]
                key = select_key(validation)
                if best_key is None or key < best_key:
                    best_key, selected_round = key, iteration
                    torch.save({'policy': policy.state_dict(), 'round': iteration, 'seed': seed,
                                'hold': hold, 'reward': reward}, out / f'{name}_selected.pt')
                record = {'round': iteration, 'key': key, 'rows': validation}
                save(out / f'{name}_dev_{iteration}.json', record)
            if iteration == rounds: break
            batch = []
            for index in rng.integers(len(bank), size=episodes):
                result, trajectory, _ = rollout(bank[index], config, hold, reward, scale, policy,
                                               train=True, keep_trace=True)
                rows.append(result)
                advantage = 0.
                for j in reversed(range(len(trajectory))):
                    next_value = trajectory[j + 1]['value'] if j + 1 < len(trajectory) else 0.
                    delta = trajectory[j]['reward'] + next_value - trajectory[j]['value']
                    # Same eligibility decay per learner update, not per decision.
                    advantage = delta + (.95 ** trajectory[j]['duration']) * advantage
                    trajectory[j]['advantage'] = advantage
                    trajectory[j]['return'] = advantage + trajectory[j]['value']
                archive.write(json.dumps({'iteration': iteration, 'result': result,
                    'trajectory': [{**t, 'x': t['x'].tolist()} for t in trajectory]}, allow_nan=False) + '\n')
                batch.extend(trajectory)
            if not batch:
                learning.append({'round': iteration + 1, 'decisions': 0}); continue
            transitions += len(batch)
            x = torch.stack([r['x'] for r in batch]); a = torch.tensor([r['action'] for r in batch])
            logp = torch.tensor([r['logp'] for r in batch])
            adv = torch.tensor([r['advantage'] for r in batch]); ret = torch.tensor([r['return'] for r in batch])
            raw_std = float(adv.std(unbiased=False))
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
            diagnostics = []
            # Full-batch epochs keep optimizer-step count equal across H arms.
            for _ in range(4):
                dist, value = policy(x)
                ratio = (dist.log_prob(a) - logp).exp()
                actor = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
                loss = actor + .25 * (value - ret).square().mean() - .01 * dist.entropy().mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step(); updates += 1
                diagnostics.append({'actor_loss': float(actor.detach()), 'value_mse': float((value.detach() - ret).square().mean()),
                                    'entropy': float(dist.entropy().mean().detach()),
                                    'clip_fraction': float(((ratio.detach() - 1).abs() > .2).float().mean())})
            learning.append({'round': iteration + 1, 'decisions': len(batch), 'advantage_std': raw_std,
                             'optimizer': diagnostics})
    torch.save({'policy': policy.state_dict(), 'round': rounds, 'seed': seed,
                'hold': hold, 'reward': reward}, out / f'{name}_last.pt')
    save(out / f'{name}_learning.json', learning)
    summary = {'name': name, 'hold': hold, 'reward': reward, 'seed': seed,
               'selected_round': selected_round, 'selected_key': best_key, 'episodes': len(rows),
               'learner_updates': sum(r['steps'] for r in rows), 'decisions': transitions,
               'ppo_optimizer_updates': updates, 'seconds': time.time() - begin,
               'parameter_change_l2': sum(float((policy.state_dict()[k] - v).square().sum()) for k, v in initial.items()) ** .5,
               'selected_sha256': sha(out / f'{name}_selected.pt'), 'last_sha256': sha(out / f'{name}_last.pt')}
    save(out / f'{name}_summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main(output):
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise RuntimeError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    previous = HERE.parent/'v16_step_reward_toy'/'archive'
    prior = json.loads((previous/'protocol.json').read_text())
    config = prior['environment']
    seeds = prior['policy_seeds']
    start = time.time()
    protocol = {**prior, 'arms': ARMS, 'hold': 1,
        'diagnostic_test_seeds': prior['test_seeds'],
        'test_seeds': list(range(1730000, 1730048)),
        'reward': 'gamma=1; -delta F/.01, -ALL actual Adam delta squared/s_D, or -protected-overlap actual Adam delta squared/s_D; failure penalty 10; no clipping or other shaping',
        'movement_scale': 'Identical scalar from v16 training-only fixed .5 calibration for BOTH movement rewards. Only the summation mask changes; no group-size or protected-cost renormalization.',
        'overlap_definition': 'Frozen at episode start: old usage >=1/3 AND new usage >=.05. This is the existing protected group, not every nonzero routing overlap. No change to input/action/learner.',
        'comparison': 'All H=1; same train/dev roots, policy seeds, batches, budgets, optimizer steps and checkpoint selection as v16. Fresh confirmatory test roots generated after model lock; old test roots diagnostic only.',
        'source_sha256': {str(p): sha(p) for p in [Path(__file__), V2/'experiment.py', V2/'model_core.py']},
        'v16_protocol_sha256': sha(previous/'protocol.json'),
        'v16_scale_sha256': sha(previous/'scale_lock.json')}
    save(out/'protocol.json', protocol)
    source = out/'source'; source.mkdir()
    for p in [Path(__file__), V2/'experiment.py', V2/'model_core.py']:
        shutil.copy2(p, source/p.name)
    train = old.make_bank(protocol['train_seeds'], config)
    dev = old.make_bank(protocol['dev_seeds'], config)
    scale = json.loads((previous/'scale_lock.json').read_text())['scale']
    calibration = [rollout(x, config, 1, 'movement', scale, action=2)[0] for x in train]
    remeasured = float(np.median([r['path_energy'] for r in calibration if r['path_energy']>0]))
    assert abs(remeasured - scale) < 1e-12
    save(out/'scale_lock.json', {'scale': scale, 'rows': calibration, 'same_scale_for_both_movement_rewards': True})
    print(json.dumps({'stage':'scale_verified', 'scale':scale, 'seconds':time.time()-start}), flush=True)
    dev_fixed = {str(a): [rollout(x, config, 1, 'forgetting', scale, action=a)[0] for x in dev]
                 for a in range(5)}
    fixed = int(min(dev_fixed, key=lambda a: select_key(dev_fixed[a])))
    assert fixed == json.loads((previous/'baseline_selection.json').read_text())['selected_action']
    save(out/'baseline_selection.json', {'selected_action':fixed, 'candidates':dev_fixed})
    fits = [train_policy(out, train, dev, config, h, reward, scale, seed,
                         protocol['rounds'], protocol['episodes_per_round'])
            for h, reward in ARMS for seed in seeds]
    save(out/'test_lock.json', {'protocol_sha256':sha(out/'protocol.json'),
         'scale_sha256':sha(out/'scale_lock.json'), 'fixed':fixed, 'fits':fits, 'epoch':time.time()})
    counts = {}
    for cohort, root_seeds in [('confirmation',protocol['test_seeds']),
                              ('diagnostic',protocol['diagnostic_test_seeds'])]:
        print(f'Generating locked {cohort} roots', flush=True)
        bank = old.make_bank(root_seeds, config, test=True)
        results = {}
        for a in range(5):
            results[f'fixed_{old.ACTIONS[a]}'] = [rollout(x,config,1,'forgetting',scale,action=a,keep_trace=True)[0] for x in bank]
        for fit in fits:
            for kind in ('selected','last'):
                policy = old.Policy()
                checkpoint = torch.load(out/f'{fit["name"]}_{kind}.pt',weights_only=False)
                policy.load_state_dict(checkpoint['policy']);policy.eval()
                name = f'{fit["name"]}_{kind}'
                results[name] = [rollout(x,config,1,fit['reward'],scale,policy,keep_trace=True)[0] for x in bank]
                print(json.dumps({'cohort':cohort,'test':name,'success':sum(r['success'] for r in results[name]),
                    'F':float(np.mean([r['test_F'] for r in results[name]])),
                    'overlap_D':float(np.mean([r['overlap_path_energy'] for r in results[name]]))}),flush=True)
        with gzip.open(out/f'{cohort}_results.json.gz','wt') as file:
            json.dump(results,file,allow_nan=False)
        counts[cohort] = {'methods':len(results),'roots':len(bank),'rollouts':sum(map(len,results.values())),
                         'learner_updates':sum(r['steps'] for rows in results.values() for r in rows)}
    save(out/'completion.json', {'status':'completed','seconds':time.time()-start,'fits':fits,'evaluation':counts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--out',required=True)
    main(parser.parse_args().out)

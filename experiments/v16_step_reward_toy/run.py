"""CPU toy 2x2: control hold 1/5 x forgetting/actual Adam path energy.

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
ARMS = [(h, r) for h in (1, 5) for r in ('forgetting', 'movement')]


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
        for _ in range(hold):
            if env.done: break
            before = env.w.detach().clone()
            env.advance(selected)
            delta = (env.w.detach().double() - before.double()).square().sum()
            path_energy += float(delta)
            if keep_trace:
                detailed.append({**env.events[-1], 'actual_update_sq': float(delta),
                                 'cumulative_update_sq': path_energy,
                                 'net_displacement_sq': float((env.w.detach().double() - env.initial_w.double()).square().sum())})
        duration = env.steps - before_step
        assert duration > 0
        reward = (-(env.forgetting - before_f) / config['reward_scale'] if reward_kind == 'forgetting'
                  else -(path_energy - before_d) / movement_scale)
        if env.done and not env.success: reward -= 10.
        if policy is not None:
            trajectory.append(dict(x=obs, action=selected, logp=logp, value=float(value),
                                   reward=reward, duration=duration))
    terminal_reward = -(env.forgetting / config['reward_scale'] if reward_kind == 'forgetting'
                        else path_energy / movement_scale) - 10. * (not env.success)
    if trajectory:
        assert abs(sum(t['reward'] for t in trajectory) - terminal_reward) < 1e-8
    policy_macs = old.Policy.macs() * len(decisions) if policy is not None else 0
    row = dict(root=item[0]['seed'], success=bool(env.success), steps=env.steps,
               decisions=len(decisions), monitor_F=env.forgetting, new_monitor_loss=env.current_loss,
               initial_new_loss=env.initial_loss, path_energy=path_energy,
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


def main(output, small=False):
    out = Path(output)
    if out.exists() and any(out.iterdir()): raise RuntimeError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(old.CONFIG, interval=1)
    ntrain, ndev, ntest = (6, 4, 8) if small else (32, 16, 48)
    rounds, episodes = (4, 3) if small else (32, 8)
    seeds = [16171] if small else [16171, 16172, 16173]
    seed_offset = 100000 if small else 0
    start = time.time()
    protocol = {'environment': config, 'arms': ARMS, 'policy_seeds': seeds,
                'train_seeds': list(range(1610000 + seed_offset, 1610000 + seed_offset + ntrain)),
                'dev_seeds': list(range(1620000 + seed_offset, 1620000 + seed_offset + ndev)),
                'test_seeds': list(range(1630000 + seed_offset, 1630000 + seed_offset + ntest)),
                'rounds': rounds, 'episodes_per_round': episodes, 'checkpoints': [0, rounds // 2, rounds],
                'reward': 'gamma=1; -delta F/.01 or -actual Adam delta squared/movement_scale, minus 10 at failure; no other shaping or reward clipping',
                'movement_scale': 'Median positive path energy of fixed .5 on training roots only, locked before policy learning.',
                'comparison': 'Same root order, initialization, batches, 300 inner-update cap and per-update monitoring. H changes policy hold only. Same four full-batch PPO epochs/round; GAE .95**actual_duration.',
                'selection': 'Identical dev lexicographic success count, successful monitor F, updates; selected and final checkpoints both reported.',
                'limits': 'Toy pilot on fresh synthetic seeds; reward penalty is a proxy, not a hard feasibility guarantee. Reuses v2 uniform initialization whose greedy tie chooses m=0. No claim of equal total wall time across H.',
                'source_sha256': {str(p): sha(p) for p in [Path(__file__), V2/'experiment.py', V2/'model_core.py']}}
    save(out / 'protocol.json', protocol)
    sources = out / 'source'; sources.mkdir()
    for p in [Path(__file__), V2/'experiment.py', V2/'model_core.py']:
        shutil.copy2(p, sources / p.name)
    print('Building training and development roots', flush=True)
    train = old.make_bank(protocol['train_seeds'], config)
    dev = old.make_bank(protocol['dev_seeds'], config)
    calibration = [rollout(x, config, 5, 'movement', 1., action=2)[0] for x in train]
    positive = [r['path_energy'] for r in calibration if r['path_energy'] > 0]
    if not positive: raise RuntimeError('No positive training motion; cannot calibrate scale')
    scale = float(np.median(positive))
    save(out / 'scale_lock.json', {'scale': scale, 'rows': calibration, 'training_only': True})
    print(json.dumps({'stage': 'scale_locked', 'movement_scale': scale, 'seconds': time.time()-start}), flush=True)
    dev_fixed = {str(a): [rollout(x, config, 5, 'forgetting', scale, action=a)[0] for x in dev]
                 for a in range(5)}
    selected_fixed = min(dev_fixed, key=lambda a: select_key(dev_fixed[a]))
    save(out / 'baseline_selection.json', {'selected_action': int(selected_fixed), 'candidates': dev_fixed})
    fits = []
    for hold, reward in ARMS:
        for seed in seeds:
            fits.append(train_policy(out, train, dev, config, hold, reward, scale, seed, rounds, episodes))
    save(out / 'test_lock.json', {'protocol_sha256': sha(out/'protocol.json'), 'scale_sha256': sha(out/'scale_lock.json'),
                                 'fixed': int(selected_fixed), 'fits': fits, 'epoch': time.time()})
    print('Generating held-out test roots after model selection lock', flush=True)
    test = old.make_bank(protocol['test_seeds'], config, test=True)
    results = {}
    for a in range(5):
        results[f'fixed_{old.ACTIONS[a]}'] = [rollout(x, config, 5, 'forgetting', scale, action=a, keep_trace=True)[0] for x in test]
    for hold in (1, 5):
        results[f'rule_h{hold}'] = [rollout(x, config, hold, 'forgetting', scale, rule=True, keep_trace=True)[0] for x in test]
    for fit in fits:
        for kind in ('selected', 'last'):
            policy = old.Policy()
            checkpoint = torch.load(out/f'{fit["name"]}_{kind}.pt', weights_only=False)
            policy.load_state_dict(checkpoint['policy']); policy.eval()
            name = f'{fit["name"]}_{kind}'
            results[name] = [rollout(x, config, fit['hold'], fit['reward'], scale, policy, keep_trace=True)[0] for x in test]
            print(json.dumps({'test': name, 'success': sum(r['success'] for r in results[name]),
                              'mean_F': float(np.mean([r['test_F'] for r in results[name]]))}), flush=True)
    with gzip.open(out/'test_results.json.gz', 'wt') as file:
        json.dump(results, file, allow_nan=False)
    save(out/'completion.json', {'status': 'completed', 'seconds': time.time()-start,
                                'fits': fits, 'test_methods': len(results), 'test_roots': len(test),
                                'test_rollouts': sum(len(r) for r in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    parser.add_argument('--small', action='store_true'); args = parser.parse_args()
    main(args.out, args.small)

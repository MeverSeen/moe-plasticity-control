"""Run the fixed, CPU-only v11 development diagnostic into a new directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V8 = HERE.parent / 'v8_multilayer_retention'
sys.path.insert(0, str(V8))
from control import SharedController, rule_chooser
from data import CIFAR100_MD5, class_partition, load_dataset
from environment import RetentionEnv, evaluate_endpoints, run_remaining
from run_retention import history_for
from diagnostic_control import train_diagnostic, _reward_parts


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_hashes():
    # Lock code that can affect rollouts before loading data. Post-run reporting
    # and tests are separately covered by the final archive manifest.
    paths = [HERE/'protocol.json', HERE/'run_diagnostics.py',
             HERE/'diagnostic_control.py', *V8.glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def action_seed(root, draw, task, step):
    """Stable external randomness independent of policy/checkpoint and path length."""
    payload = f'v11:{root}:{draw}:{task}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)


def collect_probes(histories, config, limit):
    observations, provenance, costs = [], [], []
    for root, history in histories:
        env = RetentionEnv(history, root, config)
        counts = {}
        def choose(obs):
            count = counts.get(env.stage, 0)
            if count < limit:
                frozen = obs.detach().clone()
                observations.append(frozen)
                provenance.append(dict(root=root, task=env.stage, step=env.local_steps,
                    sha256=hashlib.sha256(frozen.numpy().tobytes()).hexdigest()))
            counts[env.stage] = count+1
            return torch.ones(2, config['experts']), None
        run_remaining(env, choose)
        costs.append(dict(root=root, **env.cost))
    if not observations:
        raise ValueError('no controlled observations in fixed training-only probe panel')
    return observations, provenance, costs


def compare_original(logs, arm, seed, atol=2e-6):
    if seed != 881:
        return dict(applicable=False, reason='Only seed 881 was archived by the v8 pilot.')
    path = V8/'results/pilot/models'/f'learned_{arm}_{seed}_training.json'
    original = json.loads(path.read_text())['log']
    errors, maxima = [], {}
    for new, old in zip(logs, original):
        for field in ['round', 'root', 'nonempty_trajectories', 'decision_counts']:
            if new[field] != old[field]:
                errors.append(f'round {old["round"]}: {field}')
        for field in ['rewards', 'reward_std', 'loss']:
            if old[field] is None or new[field] is None:
                delta = 0. if old[field] == new[field] else float('inf')
            else:
                delta = float(np.max(np.abs(np.asarray(new[field])-np.asarray(old[field]))))
            maxima[field] = max(maxima.get(field, 0.), delta)
            if delta > atol:
                errors.append(f'round {old["round"]}: {field}, delta={delta}')
    if len(logs) < len(original):
        errors.append('not enough rounds for original pilot replay')
    result = dict(applicable=True, passed=not errors, compared_rounds=len(original),
                  absolute_tolerance=atol, maximum_absolute_deltas=maxima, errors=errors,
                  source=path.relative_to(ROOT).as_posix(),
                  limitation='The original archive has no per-action/logprob trace; those cannot be replay-compared.')
    return result


def evaluate(history, root, config, chooser_factory, metadata):
    env = RetentionEnv(history, root, config, capture=True)
    run_remaining(env, chooser_factory(env))
    row = evaluate_endpoints(env, [task['eval'] for task in history], 'dev',
                             metadata['arm'], metadata['policy_seed'])
    parts, reward = _reward_parts(env)
    row.update(metadata, monitor_reward=reward, monitor_reward_parts=parts)
    return row


def total_cost(training, evaluation, probes):
    keys = ['learner_updates', 'train_examples', 'monitor_examples', 'route_examples',
            'controller_calls', 'audit_examples']
    def total(rows):
        return {key: sum(row.get(key, 0) for row in rows) for key in keys}
    groups = dict(controller_training=total(training),
                  development_evaluation=total([row['cost'] for row in evaluation]),
                  fixed_probe_collection=total(probes))
    return dict(**groups, total={key: sum(group[key] for group in groups.values()) for key in keys},
                note='Physical learner work. Shared training prefixes are charged once per GRPO group; each dev rollout is run and charged independently. audit_examples is the inherited endpoint-evaluation counter and contains train-split dev images only. Not FLOPs matched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/datasets')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error('output exists; use a fresh directory')
    if (ROOT/'experiments') in out.parents:
        parser.error('do not run inside the frozen experiments archive')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = json.loads((HERE/'protocol.json').read_text())
    plan.update(source_hashes=source_hashes(), class_partition=class_partition(),
                dataset=dict(name='CIFAR-100', archive_md5=CIFAR100_MD5,
                             loaded_splits=['train'], official_test_loaded=False))
    out.mkdir(parents=True)
    (out/'training').mkdir()
    (out/'checkpoints').mkdir()
    started = time.perf_counter()
    save(out/'locked_protocol.json', plan)
    save(out/'environment.json', dict(python=sys.version, torch=torch.__version__,
         numpy=np.__version__, platform=platform.platform(), threads=1, device='cpu'))
    images, labels = load_dataset(args.cache, 'train', download=False)
    config, partition = plan['config'], plan['class_partition']
    train = [(root, history_for(images, labels, partition['train'], root, config))
             for root in plan['training_roots']]
    dev = [(root, history_for(images, labels, partition['dev'], root, config))
           for root in plan['development_roots']]
    observations, provenance, probe_costs = collect_probes(train, config,
        plan['probe']['first_decisions_per_controlled_task'])
    torch.save(observations, out/'probe_observations.pt')
    save(out/'probes.json', dict(provenance=provenance, cost=probe_costs))
    print(f'Frozen {len(observations)} training-only probe observations', flush=True)
    training_costs, regressions, checkpoints = [], [], []
    for arm in plan['arms']:
        for seed in plan['policy_seeds']:
            print(f'Train {arm} seed={seed}: {plan["rounds"]} rounds', flush=True)
            def notify(epoch, policy):
                print(f'  {arm}/{seed} checkpoint={epoch}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
            _, candidates, logs, cost = train_diagnostic(train, config, seed,
                arm == 'coordinated', plan['rounds'], plan['checkpoints'], observations, notify)
            save(out/'training'/f'{arm}-{seed}.json', dict(log=logs, cost=cost))
            regression = dict(arm=arm, seed=seed, **compare_original(logs, arm, seed))
            regressions.append(regression)
            save(out/'original_replay_check.json', regressions)
            if regression.get('applicable') and not regression['passed']:
                raise RuntimeError(f'Original v8 replay regression failed: {regression}')
            training_costs.append(cost)
            for epoch, state in candidates:
                path = out/'checkpoints'/f'{arm}-{seed}-epoch{epoch}.pt'
                torch.save(dict(state=state, arm=arm, seed=seed, epoch=epoch), path)
                checkpoints.append((arm, seed, epoch, state))
    rows = []
    for rule in plan['rules']:
        for root, history in dev:
            metadata = dict(rowkind='baseline', arm=rule, policy_seed=None, epoch=None,
                            mode='greedy', draw=None)
            rows.append(evaluate(history, root, config, lambda env: rule_chooser(rule), metadata))
        save(out/'evaluations.json', rows)
        print(f'Baseline {rule} done, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    for arm, seed, epoch, state in checkpoints:
        policy = SharedController(seed, arm == 'coordinated')
        policy.load_state_dict(state)
        policy.eval()
        for root, history in dev:
            for mode, draw in [('greedy', None), *[('sample', d) for d in plan['sampling_draws']]]:
                def make_chooser(env):
                    if mode == 'greedy':
                        return policy.chooser(sample=False)
                    def choose(obs):
                        rng = torch.Generator().manual_seed(action_seed(root, draw, env.stage, env.local_steps))
                        return policy.chooser(sample=True, generator=rng)(obs)
                    return choose
                metadata = dict(rowkind='learned', arm=arm, policy_seed=seed, epoch=epoch,
                                mode=mode, draw=draw)
                rows.append(evaluate(history, root, config, make_chooser, metadata))
        save(out/'evaluations.json', rows)
        print(f'Evaluated {arm}/{seed} epoch={epoch}; rows={len(rows)}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    cost = total_cost(training_costs, rows, probe_costs)
    cost['wall_seconds'] = time.perf_counter()-started
    cost['controller_training_seconds'] = sum(c['seconds'] for c in training_costs)
    cost['diagnostic_seconds_within_training'] = sum(c['diagnostic_seconds'] for c in training_costs)
    save(out/'total_cost.json', cost)
    save(out/'completion.json', dict(complete=True, evaluation_rows=len(rows),
         checkpoint_files=len(checkpoints), source_hashes_unchanged=source_hashes() == plan['source_hashes'],
         development_only=True))
    print(f'Completed {len(rows)} development rows; physical updates={cost["total"]["learner_updates"]}; seconds={cost["wall_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()

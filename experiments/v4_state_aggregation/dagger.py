"""Paired cost-sensitive state aggregation with an unchanged branch teacher.

This is DAgger-style collection, not a claim that DAgger's guarantees apply:
the teacher supplies noisy action values under a fixed continuation policy.
Train/dev evaluator labels are offline supervision; test labels are scoring only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import branching as b

SEEDS = [381, 382, 383]
SPLITS = {
    'calibration': list(range(410000, 410016)),
    'train': list(range(420000, 420064)),
    'dev': list(range(430000, 430024)),
    'test': list(range(440000, 440080)),
}
PILOT_SPLITS = {
    key: list(range(450000 + 100 * i, 450000 + 100 * i + n))
    for i, (key, n) in enumerate([('calibration', 4), ('train', 8), ('dev', 6), ('test', 8)])
}


def fingerprint(env):
    """Include optimizer, observation history and future stream, not only weights."""
    h = hashlib.sha256()
    h.update(str((env.world['seed'], env.steps, env.work, env.budget)).encode())
    for value in [env.w, env.adam_m, env.adam_v, env.observe(True),
                  env.new_indices, env.old_indices]:
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def trajectory_snapshots(item, config, chooser, limit):
    env = b.new_env(item, config)
    states = []
    while not env.done:
        states.append(b.clone_env(env))
        env.advance(int(chooser(env)))
    if limit is not None and len(states) > limit:
        # Time-spread samples; no filtering by branch rewards or action disagreement.
        indices = np.linspace(0, len(states) - 1, limit, dtype=int)
        states = [states[i] for i in indices]
    return states, dict(updates=env.steps, work=env.work)


def learned_snapshots(item, config, policies, limit):
    unique = {}
    cost = dict(updates=0, work=0., trajectories=0)
    for name, chooser in policies.items():
        states, used = trajectory_snapshots(item, config, chooser, limit)
        cost['updates'] += used['updates']
        cost['work'] += used['work']
        cost['trajectories'] += 1
        for env in states:
            key = fingerprint(env)
            if key not in unique:
                unique[key] = (env, [])
            unique[key][1].append(name)
    return [(env, sources) for env, sources in unique.values()], cost


def behavior_snapshots(item, config, count, round_id):
    """Independent nonlearned paths, with exactly the same number of label queries."""
    if count == 0:
        return [], dict(updates=0, work=0., trajectories=0, duplicate_queries=0)
    root = item[0]['seed']
    pool = {}
    cost = dict(updates=0, work=0., trajectories=0, duplicate_queries=0)
    for path in range(8):
        rng = np.random.default_rng(root * 191 + round_id * 101 + path * 31)
        chooser = (lambda env, a=path: a) if path < 5 else (lambda env: int(rng.integers(5)))
        states, used = trajectory_snapshots(item, config, chooser, None)
        cost['updates'] += used['updates']
        cost['work'] += used['work']
        cost['trajectories'] += 1
        for env in states:
            pool.setdefault(fingerprint(env), (env, [f'nonlearned_{path}']))
    candidates = list(pool.values())
    if not candidates:
        raise RuntimeError('Matched query count requested for a terminal history')
    rng = np.random.default_rng(root * 61 + round_id * 3001)
    indices = rng.permutation(len(candidates)).tolist()
    if len(indices) < count:
        extra = count - len(indices)
        indices += rng.integers(len(candidates), size=extra).tolist()
        cost['duplicate_queries'] = extra
    return [candidates[i] for i in indices[:count]], cost


def query(env, continuation, replicates, state_id, sources, mode='fixed'):
    """One current intervention, then fixed/constant/adaptive continuation."""
    ref = b.evaluator_reference(env)
    branches = [[] for _ in b.ACTIONS]
    cost = dict(branch_updates=0, branch_work=0.)
    for rep in range(replicates):
        future = b.resample_future(env, rep)
        for action in range(len(b.ACTIONS)):
            branch = b.clone_env(future)
            branch.advance(action)
            local = dict(new_gain=env.current_loss - branch.current_loss,
                         old_monitor_change=float(branch.old_losses.mean() - env.old_losses.mean()),
                         steps=branch.steps - env.steps)
            if mode == 'constant':
                b.finish(branch, action)
            elif mode == 'fixed':
                b.finish(branch, continuation)
            elif mode == 'adaptive':
                while not branch.done:
                    branch.advance(int(continuation(branch)))
            else:
                raise ValueError(mode)
            row = b.endpoint(branch, ref)
            row.update(replicate=rep, local=local)
            branches[action].append(row)
            cost['branch_updates'] += branch.steps - env.steps
            cost['branch_work'] += branch.work - env.work
    record = dict(root=int(env.world['seed']), old_tasks=env.task, step=env.steps,
                  state_id=state_id, fingerprint=fingerprint(env), sources=sources,
                  x=env.observe(True).tolist(), mask=env.mask.tolist(),
                  protected_mass=env.protected_mass, branches=branches,
                  continuation_mode=mode)
    return record, cost


def load_policies(folder, manifest, prefix=''):
    return {prefix + name: b.scorer_chooser(*b.load_scorer(folder / f'{name}.pt'))
            for name in manifest}


def fit(folder, train, dev, epochs):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.time()
    manifest = b.train_scorers(folder, train, dev, epochs, seeds=SEEDS)
    b.save(folder / 'fit_cost.json', dict(seconds=time.time() - start, states=len(train),
        histories=len({s['root'] for s in train}), epochs_per_model=epochs, models=len(manifest),
        note='Same final fitting algorithm, epochs, architecture and matched initializations.'))
    return manifest


def collect_aggregation(out, seeds, config, policies, continuation, round_id, replicates):
    folder = out / f'collection_r{round_id}'
    folder.mkdir(parents=True, exist_ok=False)
    data = {'aggregate': [], 'matched': []}
    costs = {kind: dict(branch_updates=0, branch_work=0., path_updates=0, path_work=0.,
                      queries=0, duplicate_queries=0) for kind in data}
    roots = []
    start = time.time()
    streams = {kind: (folder / f'{kind}_branches.jsonl').open('w') for kind in data}
    try:
        for index, seed in enumerate(seeds):
            item = b.make_bank([seed], config)[0]
            visited, used_on = learned_snapshots(item, config, policies, limit=3)
            extra, used_off = behavior_snapshots(item, config, len(visited), round_id)
            roots.append(dict(root=seed, aggregate=len(visited), matched=len(extra),
                              matched_duplicate_queries=used_off['duplicate_queries']))
            for kind, entries, used in [('aggregate', visited, used_on), ('matched', extra, used_off)]:
                costs[kind]['path_updates'] += used['updates']
                costs[kind]['path_work'] += used['work']
                costs[kind]['duplicate_queries'] += used.get('duplicate_queries', 0)
                for j, (env, sources) in enumerate(entries):
                    record, expense = query(env, continuation, replicates,
                        f'train:r{round_id}:{seed}:{kind}:{j}', sources)
                    data[kind].append(record)
                    streams[kind].write(json.dumps(record) + '\n')
                    costs[kind]['queries'] += 1
                    for key in expense:
                        costs[kind][key] += expense[key]
                streams[kind].flush()
            if (index + 1) % 8 == 0 or index + 1 == len(seeds):
                print(json.dumps(dict(stage='aggregate_queries', round=round_id,
                    histories=index + 1, states=len(data['aggregate']),
                    seconds=round(time.time() - start, 1))), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    b.save(folder / 'collection_cost.json', dict(costs=costs, roots=roots,
        seconds=time.time() - start,
        note='Query counts matched per history, not FLOPs; pretraining excluded from work counters.'))
    return data


def policy_predictions(out, states, registry, name):
    x = np.array([s['x'] for s in states], dtype=np.float32)
    actions = {}
    for method, path in registry.items():
        model, ckpt = b.load_scorer(out / path)
        actions[method] = b.predict_actions(model, x, ckpt['mean'], ckpt['std'],
                                          ckpt['feedback'], ckpt['slack']).tolist()
    b.save(out / f'{name}_predictions.json', dict(state_ids=[s['state_id'] for s in states], actions=actions))


def online_eval(out, seeds, config, registry, baselines):
    policies = {name: b.scorer_chooser(*b.load_scorer(out / path)) for name, path in registry.items()}
    policies.update(best_static=b.baseline_chooser('static', baselines['static']),
                    hand_feedback=b.baseline_chooser('hand', baselines['hand']), full_update=lambda env: 4)
    rows = []
    start = time.time()
    for index, seed in enumerate(seeds):
        item = b.make_bank([seed], config)[0]
        for name, chooser in policies.items():
            rows.append(dict(method=name, **b.rollout(item, config, chooser)))
        if (index + 1) % 8 == 0 or index + 1 == len(seeds):
            b.save(out / 'test_online.json', rows)
            print(json.dumps(dict(stage='online', histories=index + 1,
                seconds=round(time.time() - start, 1))), flush=True)
    b.save(out / 'online_cost.json', dict(seconds=time.time() - start,
        model_updates=sum(r['steps'] for r in rows), charged_work=sum(r['work'] for r in rows)))


def test_visited_and_continuation(out, seeds, config, registry, continuation, rounds, replicates):
    """Frozen policies visit new test worlds. Branch outcomes are diagnostic only."""
    selected = {name: b.scorer_chooser(*b.load_scorer(out / path)) for name, path in registry.items()
                if name.startswith('bc_') or name.startswith(f'aggregate_r{rounds}_')}
    # Predeclared first policy initialization, never picked using test outcomes.
    adaptive_name = f'aggregate_r{rounds}_feedback_{SEEDS[0]}'
    adaptive = selected[adaptive_name]
    states, diagnostic = [], []
    costs = dict(branch_updates=0, branch_work=0., path_updates=0, path_work=0.)
    start = time.time()
    with (out / 'test_visited_branches.jsonl').open('w') as stream, \
            (out / 'test_continuations.jsonl').open('w') as diag_stream:
        for index, seed in enumerate(seeds):
            item = b.make_bank([seed], config)[0]
            entries, used = learned_snapshots(item, config, selected, limit=2)
            costs['path_updates'] += used['updates']
            costs['path_work'] += used['work']
            # One noninitial visited state per history; otherwise initial state.
            diagnostic_index = next((j for j, (env, _) in enumerate(entries) if env.steps > 0), 0)
            for j, (env, sources) in enumerate(entries):
                state_id = f'test:{seed}:visited:{j}'
                record, used = query(env, continuation, replicates, state_id, sources)
                for key in used:
                    costs[key] += used[key]
                states.append(record)
                stream.write(json.dumps(record) + '\n')
                if j == diagnostic_index:
                    families = {'fixed': record}
                    for mode, chooser in [('constant', continuation), ('adaptive', adaptive)]:
                        row, used = query(env, chooser, replicates, state_id, sources, mode)
                        families[mode] = row
                        for key in used:
                            costs[key] += used[key]
                    entry = dict(root=seed, state_id=state_id, adaptive_policy=adaptive_name,
                                 families=families)
                    diagnostic.append(entry)
                    diag_stream.write(json.dumps(entry) + '\n')
            stream.flush()
            diag_stream.flush()
            if (index + 1) % 8 == 0 or index + 1 == len(seeds):
                print(json.dumps(dict(stage='test_visited', histories=index + 1,
                    states=len(states), seconds=round(time.time() - start, 1))), flush=True)
    b.save(out / 'test_visited_cost.json', dict(costs=costs, states=len(states),
        diagnostic_states=len(diagnostic), adaptive_policy=adaptive_name, seconds=time.time() - start,
        note='Privileged offline test diagnostics; no teacher decisions enter deployable rollouts.'))
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1 or args.epochs < 1:
        parser.error('rounds and epochs must be positive')
    if args.out.exists():
        parser.error('Use a fresh output directory')
    args.out.mkdir(parents=True)
    start = time.time()
    config = copy.deepcopy(b.base.CONFIG)
    splits = copy.deepcopy(PILOT_SPLITS if args.pilot else SPLITS)
    plan = dict(config=config, splits=splits, pilot=args.pilot, rounds=args.rounds,
        epochs=args.epochs, replicates=2, policy_seeds=SEEDS, actions=b.ACTIONS,
        features=b.FEATURES, feedback_indices=[12, 13],
        primary='Final-round aggregate feedback vs progress on all held-out histories, reach first then paired forgetting.',
        secondary=['Aggregate vs BC within each observation mode',
                   'Aggregate vs query-count-matched nonlearned collection',
                   'Decision cost on behavior and learner-visited test states',
                   'Continuation mismatch under paired independent futures'],
        aggregation='Union of all progress/feedback replicas; same training states for both inputs; 3 time-spread snapshots per rollout, deduplicate complete states within root and round.',
        teacher='5 current actions for H=5 then one calibrated constant continuation; unchanged across aggregation rounds.',
        matched='Same label-query counts per root and round, sampled from 5 fixed and 3 random behavior paths. Not equal branch FLOPs.',
        selection='Unchanged v3 branch-dev checkpoint/slack selection; final round predeclared, no test selection.',
        labels='Independent evaluator samples supply train/dev offline supervision; test labels only score frozen policies and privileged diagnostics.',
        endpoint=f"First new-monitor MSE <= {config['target']} or common work/300-step budget.",
        correction='v3 actual config target=.08; parts of its prose incorrectly said .06. v4 retains .08 explicitly.',
        continuation_diagnostic='One outcome-independent visited state/history; Qconstant, H+fixed, H+frozen adaptive. Choose with future 0, evaluate with future 1 under the same adaptive continuation.',
        adaptive_policy=f'aggregate_r{args.rounds}_feedback_{SEEDS[0]}',
        limitations=['Synthetic fixed router, not learned router or a language model.',
                     'No claim of globally optimal teacher or RL necessity.',
                     'Three fitted seeds do not fully characterize training uncertainty.',
                     'Online workload proxy is not actual FLOPs; offline search cost separate.'])
    b.save(args.out / 'plan.json', plan)
    b.save(args.out / 'environment.json', dict(python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, device='cpu', threads=1))
    initial = args.out / 'initial'
    initial.mkdir()
    continuation = b.calibrate(initial, config, splits['calibration'])
    train = b.collect(initial, 'train', splits['train'], config, continuation, 2)
    dev = b.collect(initial, 'dev', splits['dev'], config, continuation, 2)
    registry = {}

    def register(prefix, folder, manifest):
        for name in manifest:
            registry[f'{prefix}_{name}'] = str((folder / f'{name}.pt').relative_to(args.out))

    manifest = fit(args.out / 'bc', train, dev, args.epochs)
    register('bc', args.out / 'bc', manifest)
    behavior = load_policies(args.out / 'bc', manifest)
    data = {'aggregate': list(train), 'matched': list(train)}
    for round_id in range(1, args.rounds + 1):
        additions = collect_aggregation(args.out, splits['train'], config, behavior,
                                        continuation, round_id, 2)
        for kind in ['aggregate', 'matched']:
            data[kind].extend(additions[kind])
            folder = args.out / f'{kind}_r{round_id}'
            new_manifest = fit(folder, data[kind], dev, args.epochs)
            register(f'{kind}_r{round_id}', folder, new_manifest)
            if kind == 'aggregate':
                behavior = load_policies(folder, new_manifest)
    baselines = b.tune_baselines(initial, config, splits['dev'])
    static_scores = [b.selection_metrics(dev, np.full(len(dev), a, dtype=int)) for a in range(5)]
    static_action = min(range(5), key=lambda a: (-static_scores[a]['reach'], static_scores[a]['cost']))
    lock = dict(plan, registry=registry, continuation_action=continuation, baselines=baselines,
        static_branch_action=static_action,
        source_hashes={p.name: b.file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        policy_hashes={path: b.file_hash(args.out / path) for path in registry.values()})
    b.save(args.out / 'locked_protocol.json', lock)
    print(json.dumps(dict(stage='locked_before_test', policies=len(registry), target=config['target'])), flush=True)
    # First creation of final test worlds is strictly below the lock.
    test = b.collect(args.out, 'test_behavior', splits['test'], config, continuation, 2)
    policy_predictions(args.out, test, registry, 'test_behavior')
    online_eval(args.out, splits['test'], config, registry, baselines)
    visited = test_visited_and_continuation(args.out, splits['test'], config, registry,
                                           continuation, args.rounds, 2)
    policy_predictions(args.out, visited, registry, 'test_visited')
    b.save(args.out / 'total_cost.json', dict(seconds=time.time() - start,
        note='End-to-end time includes old-task pretraining, querying, fitting, tuning and evaluation.'))
    print(json.dumps(dict(stage='complete', seconds=round(time.time() - start, 1))), flush=True)


if __name__ == '__main__':
    main()

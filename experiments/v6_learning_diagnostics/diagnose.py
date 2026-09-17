"""Frozen-environment learning diagnostics with new, locked final histories."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import branching as b
import learning as l

SPLITS = dict(train=list(range(610000, 610128)), dev=list(range(620000, 620048)),
              test=list(range(630000, 630096)))
PILOT = dict(train=list(range(650000, 650012)), dev=list(range(660000, 660006)),
             test=list(range(670000, 670008)))


def tune_baselines(out, bank, config):
    candidates = l.plan_candidates()
    scored = []
    for plan in candidates:
        rows = [b.rollout(item, config, l.plan_chooser(plan)) for item in bank]
        scored.append(dict(plan=plan, metrics=l.endpoint_metrics(rows),
                           rows=[{k: v for k, v in r.items() if k != 'events'} for r in rows]))
    groups = dict(best_fixed=['fixed'], best_time=['fixed', 'time'],
                  best_progress=['fixed', 'progress'], best_hand=['fixed', 'hand'])
    chosen = {}
    for name, kinds in groups.items():
        allowed = [r for r in scored if r['plan']['kind'] in kinds]
        chosen[name] = min(allowed, key=lambda r: l.selection_key(r['metrics'], 'closed_F'))['plan']
    b.save(out / 'baseline_candidates.json', scored)
    b.save(out / 'baselines.json', chosen)
    return chosen


def tiny_subset(states, count):
    chosen, seen = [], set()
    for state in states:
        if state['root'] in seen or state['protected_mass'] <= 1e-8: continue
        _, _, hit, forgetting, _ = b.arrays([state])
        if not (hit == 1).all() or np.ptp(forgetting.mean(2)[0]) <= 1e-5: continue
        chosen.append(state); seen.add(state['root'])
        if len(chosen) == count: break
    if not chosen: raise RuntimeError('No informative feasible training states for memorization')
    return chosen


def lock_protocol(out, plan, registry, positive_registry, baselines):
    source = Path(__file__).parent
    value = dict(plan, registry=registry, positive_registry=positive_registry, baselines=baselines,
        source_hashes={p.name: b.file_hash(p) for p in source.glob('*.py')},
        policy_hashes={path: b.file_hash(out / path) for path in set(registry.values()) | set(positive_registry.values())})
    b.save(out / 'locked_protocol.json', value)
    return value


def offline_evaluate(out, states, registry):
    x = np.array([s['x'] for s in states], dtype=np.float32)
    actions = {}
    metrics = {}
    for name, path in registry.items():
        model, ckpt = b.load_scorer(out / path)
        actions[name] = b.predict_actions(model, x, ckpt['mean'], ckpt['std'], ckpt['feedback'], ckpt['slack']).tolist()
        metrics[name] = l.branch_metrics(states, actions[name])
    b.save(out / 'test_predictions.json', dict(state_ids=[s['state_id'] for s in states], actions=actions))
    b.save(out / 'test_branch_metrics.json', metrics)


def online_evaluate(out, seeds, config, registry, baselines):
    policies = {name: b.load_scorer(out / path) for name, path in registry.items()}
    rows = {name: [] for name in policies}
    baseline_plans = dict(baselines, fixed_zero=dict(kind='fixed', action=0), full_update=dict(kind='fixed', action=4))
    rows.update({name: [] for name in baseline_plans})
    for index, seed in enumerate(seeds):
        item = b.make_bank([seed], config)[0]
        for name, (model, ckpt) in policies.items():
            rows[name].append(b.rollout(item, config, b.scorer_chooser(model, ckpt)))
        for name, plan in baseline_plans.items():
            rows[name].append(b.rollout(item, config, l.plan_chooser(plan)))
        if (index + 1) % 8 == 0 or index + 1 == len(seeds):
            print(json.dumps(dict(stage='test_online', histories=index + 1)), flush=True)
    b.save(out / 'test_online.json', rows)
    b.save(out / 'test_online_cost.json', dict(rollouts=sum(len(v) for v in rows.values()),
        model_updates=sum(r['steps'] for values in rows.values() for r in values),
        work=sum(r['work'] for values in rows.values() for r in values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): parser.error('Use a fresh output directory')
    args.out.mkdir(parents=True)
    started = time.time()
    config = copy.deepcopy(b.base.CONFIG)
    splits = copy.deepcopy(PILOT if args.pilot else SPLITS)
    seeds = l.SEEDS[:1] if args.pilot else l.SEEDS
    checkpoints = [0, 50, 100] if args.pilot else [0, 50, 150, 300, 600]
    tiny_checkpoints = [0, 100, 400] if args.pilot else [0, 600, 3000]
    positive_counts = [32, 8, 24] if args.pilot else [128, 32, 128]
    positive_seeds = [950000, 960000, 970000] if args.pilot else [910000, 920000, 930000]
    replicates = 2 if args.pilot else 4
    plan = dict(config=config, splits=splits, seeds=seeds, checkpoints=checkpoints, tiny_checkpoints=tiny_checkpoints,
        positive_counts=positive_counts, positive_seeds=positive_seeds, replicates=replicates, pilot=args.pilot,
        losses=l.LOSSES, selectors=l.SELECTORS, slacks=l.SLACKS, continuation_action=0,
        source_hashes_at_start={p.name: b.file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        primary=['raw_mse_closed_F_feedback minus raw_mse_closed_F_progress',
                 'raw_mse_closed_F_feedback minus legacy_branch_J_feedback'],
        mechanism='Loss (legacy transformed Huber vs raw-forgetting scaled MSE) crossed with branch-J, branch-raw-F, and closed-loop-raw-F selection. Every selector sees the same checkpoint/slack pool; capacity and observations fixed.',
        loss_limitation='Raw MSE aligns the forgetting regression target with conditional expected raw forgetting; feasibility classification and finite candidate selection still approximate the constrained objective.',
        training='Adam .002, weight decay .0001, grad norm cap 5, full batch, matched initializations; fit progress/feedback on identical new branches. No new aggregation, router dynamics or teacher policy improvement.',
        teacher='Same v4 H=5 intervention followed by fixed m=0, retained to isolate learning and selection. Four paired future minibatch streams per state in formal run.',
        selection='Development data only. Reach first, then selected cost or raw successful forgetting; work breaks ties. Branch roots weighted equally. Closed-loop endpoint selection uses dev evaluator only.',
        positive='Separate auxiliary supervised mapping, NOT a MoE task: five balanced feedback levels per identical progress context encode the optimal multiplier. Independent train/dev/test contexts. Pass per feedback replica: test accuracy>=.90 and regret<=.10 of constant-middle baseline. Progress has exact 20% action accuracy.',
        memorization='First up to 32 distinct train roots with nonempty protection, all branches feasible, action raw-forgetting range>1e-5; same 32-wide architecture, losses, optimizer/regularization. Train to 3000 epochs; assess in-sample regret, not generalization.',
        test='All selected weights, baselines and source hashes locked before creation of any final MoE or auxiliary positive test. No final result changes selection.',
        statistics='Root-paired bootstrap, matched scorer seeds averaged within each root; reach first, common-success raw forgetting and tight-endpoint sensitivity. Multiple secondary ablations are exploratory.',
        scope='Learning-pipeline diagnostics in the frozen v4 .08/300 environment. Comparisons within v6 isolate ablations; new sample sizes/futures prohibit attributing v4-to-v6 changes solely to the loss.',
        limitations=['Auxiliary positive and training memorization do not establish feedback usefulness in MoE.',
                     'Fixed continuation and partial observation remain; this round does not resolve all learning hypotheses.',
                     'Finite selected checkpoints need not be the best learnable controller.',
                     'Online work proxy is shared; extra offline closed-loop model-selection cost is recorded separately.'])
    b.save(args.out / 'plan.json', plan)
    b.save(args.out / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, threads=1, device='cpu'))
    data = args.out / 'data'; data.mkdir()
    train = b.collect(data, 'train', splits['train'], config, 0, replicates)
    dev = b.collect(data, 'dev', splits['dev'], config, 0, replicates)
    tiny = tiny_subset(train, 8 if args.pilot else 32)
    (args.out / 'tiny').mkdir()
    for kind in l.LOSSES:
        for seed in seeds:
            l.tiny_fit(args.out / 'tiny' / f'{kind}_{seed}.json', tiny, kind, seed, tiny_checkpoints)
    print(json.dumps(dict(stage='tiny_complete', states=len(tiny))), flush=True)
    registry, positive_registry = {}, {}
    ptrain = l.positive_states(positive_seeds[0], positive_counts[0])
    pdev = l.positive_states(positive_seeds[1], positive_counts[1])
    for kind in l.LOSSES:
        for feedback in [False, True]:
            mode = 'feedback' if feedback else 'progress'
            for seed in seeds:
                relative = Path('positive') / f'{kind}_{mode}_{seed}'
                l.fit(args.out / relative, ptrain, pdev, [], config, kind, feedback, seed,
                      checkpoints, selectors=['branch_F'])
                positive_registry[f'{kind}_{mode}_{seed}'] = str(relative / 'branch_F.pt')
    dev_bank = b.make_bank(splits['dev'], config)
    baselines = tune_baselines(args.out, dev_bank, config)
    for kind in l.LOSSES:
        for feedback in [False, True]:
            mode = 'feedback' if feedback else 'progress'
            for seed in seeds:
                relative = Path('models') / f'{kind}_{mode}_{seed}'
                choices = l.fit(args.out / relative, train, dev, dev_bank, config,
                                kind, feedback, seed, checkpoints)
                for selector in choices:
                    registry[f'{kind}_{selector}_{mode}_{seed}'] = str(relative / f'{selector}.pt')
    lock = lock_protocol(args.out, plan, registry, positive_registry, baselines)
    assert lock['source_hashes'] == plan['source_hashes_at_start'], 'Source changed during run'
    print(json.dumps(dict(stage='locked_before_test', policies=len(registry))), flush=True)
    # First final-test generation is strictly below the lock.
    ptest = l.positive_states(positive_seeds[2], positive_counts[2])
    positive_results = {}
    for name, relative in positive_registry.items():
        model, ckpt = b.load_scorer(args.out / relative)
        result = l.positive_score(ptest, model, ckpt)
        positive_results[name] = dict(result, feedback=ckpt['feedback'],
            passed=(result['accuracy'] >= .9 and result['relative_regret'] <= .1) if ckpt['feedback'] else None)
    b.save(args.out / 'positive_test.json', positive_results)
    test = b.collect(data, 'test', splits['test'], config, 0, replicates)
    offline_evaluate(args.out, test, registry)
    online_evaluate(args.out, splits['test'], config, registry, baselines)
    b.save(args.out / 'total_cost.json', dict(seconds=time.time() - started,
        note='Includes pretraining, branch collection, fitting, all development candidate rollouts, and final evaluation. Work is a proxy; candidate selection costs are separate in models/*/cost.json.'))
    print(json.dumps(dict(stage='complete', seconds=round(time.time() - started, 1))), flush=True)


if __name__ == '__main__':
    main()

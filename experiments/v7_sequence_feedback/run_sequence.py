"""Prespecified repeated-data experiment; final histories generated after lock."""
import argparse
import copy
from pathlib import Path
import platform
import time

import numpy as np
import torch

import control as l
import sequence_env as e

SPLITS = dict(train=list(range(710000, 710064)), dev=list(range(720000, 720024)), test=list(range(730000, 730064)))
PILOT = dict(train=list(range(750000, 750008)), dev=list(range(760000, 760004)), test=list(range(770000, 770008)))


def tune_baselines(out, bank, config):
    chosen, records = {}, {}
    for frequency in ['block', 'within']:
        candidates = []
        for plan in l.baseline_plans():
            if frequency == 'block' and plan['kind'] == 'time': continue
            rows = [e.rollout(item, config, l.plan_chooser(plan), frequency) for item in bank]
            candidates.append(dict(plan=plan, metrics=l.metrics(rows), rows=rows))
        for name, kinds in [('best_fixed', ['fixed']), ('best_schedule', ['fixed', 'block', 'time']),
                            ('best_hand', ['fixed', 'hand'])]:
            pool = [r for r in candidates if r['plan']['kind'] in kinds]
            chosen[f'{frequency}_{name}'] = dict(frequency=frequency,
                plan=min(pool, key=lambda r: l.selection_key(r['metrics']))['plan'])
        records[frequency] = candidates
    l.save(out/'baseline_candidates.json', records)
    l.save(out/'baselines.json', chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): parser.error('Use a fresh output directory')
    args.out.mkdir(parents=True)
    started = time.time()
    config, splits = copy.deepcopy(e.CONFIG), copy.deepcopy(PILOT if args.pilot else SPLITS)
    seeds = [781] if args.pilot else [781, 782, 783]
    checkpoints = [0, 50, 100] if args.pilot else [0, 150, 450]
    slacks = [0., .1, 1.]
    reps = 1 if args.pilot else 2
    source = Path(__file__).parent
    hashes = {p.name: l.file_hash(p) for p in source.glob('*.py')}
    plan = dict(config=config, splits=splits, seeds=seeds, checkpoints=checkpoints, slacks=slacks,
        replicates=reps, pilot=args.pilot, source_hashes_at_start=hashes,
        primary=['block_feedback minus block_progress', 'within_feedback minus within_progress'],
        features=e.FEATURES, observation='Progress masks only the final two old-knowledge feedback entries AFTER standardization. Global initial A reference is never refreshed. Both see B loss/change, previous B endpoint/hit, block index, budget and prior action.',
        environment='Three distinct new input domains from the existing consistent nonlinear teacher, fixed geometric router, original A count 1-3. Preserve weights, Adam first/second moments and global bias-correction step across B boundaries. Fresh Adam only once at B1 start.',
        memory='Original A only: at most 64 replay and 64 monitor samples, held fixed. No B replay or online B retention measurement after its block. All past-B retention is separately reported by final evaluator.',
        mask='Original-A usage >= 1/3 fixed for all B blocks. Current B initial gate-mass summary available on arrival; no future B lookahead, learned router or parameter-conditioned router.',
        budget='100 parameter updates per B block, total <=300; unused quota expires at boundary. Same per-5-update observation/controller work charged to all arms, even held block decisions. All models stop each block at B-monitor <=.08 or cap, then continue, including after a failure.',
        success='Primary reach means EVERY B reached .08 when encountered; not simultaneous final retention. A forgetting is positive per-A eval loss increase relative to before B1, averaged across original A tasks. Final B monitor/test retention reported separately.',
        teacher='Action through current block (block) or next <=5 updates (within), then m=0 through remaining sequence. Two paired future minibatch streams per state. Offline evaluator targets on training roots; this is not an optimal planning oracle.',
        data='Two behavior paths per train root: m=.5 and seeded random action at the allowed decision frequency. Sample active block starts plus local steps 10/30 for within. Same branches and initialization within each information comparison.',
        selection='Raw-forgetting MSE plus centered MSE and reach BCE, 18-32-32 scorer. Epoch/slack selected exclusively by dev CLOSED LOOP: reach first, successful A forgetting second, work third. Extra offline selection/search work recorded separately.',
        statistics='Average three matched model initializations within root; nominal 95% paired root bootstrap. Report reach and common-success forgetting, all-root forgetting descriptively, and endpoint-loss sensitivity. Two primary comparisons, unadjusted CIs; secondary comparisons exploratory.',
        limitation='This tests repeated adaptation, not a causal dose comparison against K=1 and not complete retention of every learned task. Frequency comparisons change action duration and teacher; old-feedback comparisons within a frequency are matched.',
        final_test='Sources, models and baselines locked before any final-test world is generated. Pilot checks mechanics/runtime; no tuning against final pilot or main outcomes.')
    l.save(args.out/'plan.json', plan)
    l.save(args.out/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
                                           numpy=np.__version__, platform=platform.platform(), threads=1))
    train_bank = [e.make_item(seed, config) for seed in splits['train']]
    dev_bank = [e.make_item(seed, config) for seed in splits['dev']]
    print('Pretraining complete', flush=True)
    baselines = tune_baselines(args.out, dev_bank, config)
    registry = {}
    for frequency in ['block', 'within']:
        states = l.collect(args.out/'data', train_bank, config, frequency, reps)
        for feedback in [False, True]:
            mode = 'feedback' if feedback else 'progress'
            for seed in seeds:
                name = f'{frequency}_{mode}_{seed}'
                relative = Path('models')/name
                l.fit(args.out/relative, states, dev_bank, config, frequency, feedback, seed, checkpoints, slacks)
                registry[name] = str(relative/'selected.pt')
    source_now = {p.name: l.file_hash(p) for p in source.glob('*.py')}
    assert hashes == source_now, 'Code changed during run'
    l.save(args.out/'locked_protocol.json', dict(plan, registry=registry, baselines=baselines,
        source_hashes=source_now, policy_hashes={p: l.file_hash(args.out/p) for p in registry.values()}))
    print('Locked before final-test generation', flush=True)
    models = {name: l.load(args.out/path) for name, path in registry.items()}
    rows = {name: [] for name in models}
    extra = dict(baselines, fixed_zero=dict(frequency='within', plan=dict(kind='fixed', action=0)),
                 full_update=dict(frequency='within', plan=dict(kind='fixed', action=4)))
    rows.update({name: [] for name in extra})
    for i, seed in enumerate(splits['test']):
        item = e.make_item(seed, config)
        for name, (model, ckpt) in models.items():
            rows[name].append(e.rollout(item, config, l.chooser(model, ckpt), ckpt['frequency'], True))
        for name, record in extra.items():
            rows[name].append(e.rollout(item, config, l.plan_chooser(record['plan']), record['frequency'], True))
        if (i+1) % 4 == 0 or i+1 == len(splits['test']):
            print(f'Test histories {i+1}/{len(splits["test"])}', flush=True)
    l.save(args.out/'test_online.json', rows)
    l.save(args.out/'test_cost.json', dict(rollouts=sum(map(len, rows.values())),
        updates=sum(r['steps'] for group in rows.values() for r in group),
        work=sum(r['work'] for group in rows.values() for r in group)))
    l.save(args.out/'total_cost.json', dict(seconds=time.time()-started,
        note='Includes pretraining, branching, candidate tuning, training and evaluation. Work proxy does not equal FLOPs.'))
    print(f'Complete: {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()

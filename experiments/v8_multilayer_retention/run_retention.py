"""Run a class-disjoint CIFAR-100 expert-control pilot into a fresh directory."""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

from control import SharedController, rule_chooser, train_controller
from data import CIFAR100_MD5, build_audit, class_partition, load_dataset, make_history
from environment import DEFAULT, FEATURES, RetentionEnv, evaluate_endpoints, run_remaining

RULES = ['global_0', 'global_0.5', 'global_1', 'group_rule', 'local_rule', 'coordinated_rule']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def digest_sources():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def history_for(images, labels, classes, root, config):
    # Task grouping/order changes by root; train roots choose 25 of the 50
    # training classes, while dev/audit use their disjoint 25-class pools.
    order = np.random.default_rng(root+81000).permutation(classes).tolist()
    order = order[:config['tasks']*5]
    return make_history(images, labels, order, root, tasks=config['tasks'],
        train_per_class=config['train_per_class'], monitor_per_class=config['monitor_per_class'],
        eval_per_class=config['eval_per_class'])


def selection_key(rows):
    # Selection is a DEVELOPMENT heuristic, never a non-inferiority test.
    # Failure fraction is lexicographically primary and includes every root.
    failures = np.mean([not all(t['success'] for t in r['tasks']) for r in rows])
    forgetting = np.mean([np.mean([max(r['loss_matrix'][-1][j]-r['loss_matrix'][j][j], 0.)
        for j in range(len(r['tasks'])-1)]) for r in rows])
    return float(failures), float(forgetting), float(np.mean([r['cost']['learner_updates'] for r in rows]))


def collect(env, chooser, batches, split, method, seed=None):
    run_remaining(env, chooser)
    return evaluate_endpoints(env, batches, split, method, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(__file__).resolve().parents[2]/'.cache/datasets')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists; choose a fresh directory')
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    out = args.out
    started = time.perf_counter()
    config = dict(DEFAULT)
    roots = dict(train=list(range(88100, 88104 if args.pilot else 88116)),
                 dev=list(range(88200, 88202 if args.pilot else 88206)),
                 audit=list(range(88300, 88303 if args.pilot else 88312)))
    seeds = [881] if args.pilot else [881, 882, 883, 884, 885]
    rounds, group_size = (6 if args.pilot else 32), 4
    partition = class_partition()
    protocol = dict(status='pilot' if args.pilot else 'exploratory',
        confirmatory=False, config=config, roots={s:roots[s] for s in ['dev', 'audit']},
        training_roots=roots['train'], class_partition=partition,
        policy_seeds=seeds, rounds=rounds, group_size=group_size,
        expected_policy_seeds={**{r:[None] for r in RULES},
                               'learned_local':seeds, 'learned_coordinated':seeds},
        dataset=dict(name='CIFAR-100', archive_md5=CIFAR100_MD5,
            source='https://www.cs.toronto.edu/~kriz/cifar.html'),
        target_rule='90th percentile of minimum observed development full-update monitor CE + .03, clipped to [.8,1.55]',
        target_calibration='Full-budget global_1 on dev roots only, before controller training; no official test data.',
        optimizer='Adam, no weight decay; actual expert displacements scaled; moments retained at zero multiplier',
        model_scope='trainable CNN stem, two residual fixed-raw-image top2 MoE layers, task-ID-visible 5way heads; all shared modules update normally; all methods use global_1 for first task, control starts at task2',
        actions=[0., .5, 1.], observation_features=FEATURES,
        memory='128 replay + 128 monitor total across all past tasks; task-balanced prefix shrink; fixed per-sample learn-time monitor references',
        evaluation='task mean positive loss increase from its own learn-time diagonal; full lower triangular matrix; dev uses disjoint official-train eval subset; audit uses official test',
        selection_rule='lexicographic dev sequence failure fraction, mean all-root dev endpoint forgetting, learner updates; init/mid/final same candidates for both learned arms',
        non_inferiority='not tested: pilot only, no confirmatory claim or selected-success-only superiority',
        costs='learner updates, train/monitor/endpoint-evaluation/route example passes, controller calls, clone/controller/training/selection seconds; audit_examples field means endpoint evaluation for either split; NOT equal FLOPs',
        data_scope='finite old memory inside each learner and training reward; offline development evaluation has extra disjoint old examples, counted separately',
        source_hashes=digest_sources())
    save(out/'plan.json', protocol)
    save(out/'environment.json', dict(python=sys.version, torch=torch.__version__,
        numpy=np.__version__, platform=platform.platform(), threads=args.threads))
    images, labels = load_dataset(args.cache, 'train', download=args.download)
    histories = {split:[(r, history_for(images, labels, partition[split], r, config)) for r in rr]
                 for split, rr in roots.items() if split != 'audit'}

    # Calibration only establishes attainable target/difficulty, without any
    # controller candidate or audit-class outcomes entering the rule.
    calibration, minima = [], []
    calibration_config = dict(config, target_loss=1e-8)
    for root, history in histories['dev']:
        env = RetentionEnv(history, root, calibration_config, capture=True)
        run_remaining(env, rule_chooser('global_1'))
        for task in range(config['tasks']):
            minima.append(min([t['new_loss'] for t in env.events if t['task']==task]
                              + [env.completed[task]['initial_loss']]))
        calibration.append(dict(root=root, tasks=env.completed, events=env.events, cost=env.cost))
        print(f'calibration root={root} terminal CE={[round(t["terminal_loss"],3) for t in env.completed]}', flush=True)
    config['target_loss'] = float(np.clip(np.quantile(minima, .9)+.03, .8, 1.55))
    save(out/'calibration.json', dict(rows=calibration, minima=minima, target_loss=config['target_loss']))
    print(f'Locked target CE={config["target_loss"]:.6f}', flush=True)
    protocol['config'] = config

    selected, selection, training_cost = {}, {}, {}
    (out/'models').mkdir()
    for coordinated in [False, True]:
        name = 'learned_coordinated' if coordinated else 'learned_local'
        for seed in seeds:
            policy, candidates, logs, cost = train_controller(histories['train'], config, seed,
                rounds=rounds, group_size=group_size, coordinated=coordinated)
            save(out/'models'/f'{name}_{seed}_training.json', dict(log=logs, cost=cost))
            training_cost[f'{name}_{seed}'] = cost
            candidate_scores = []
            best_key, best_state, best_epoch, best_rows = None, None, None, None
            for epoch, state in candidates:
                policy.load_state_dict(state)
                rows = []
                for root, history in histories['dev']:
                    env = RetentionEnv(history, root, config, capture=True)
                    rows.append(collect(env, policy.chooser(), [t['eval'] for t in history], 'dev', name, seed))
                key = selection_key(rows)
                candidate_scores.append(dict(epoch=epoch, key=key, rows=rows))
                print(f'selection {name}/{seed} epoch={epoch} key={key}', flush=True)
                if best_key is None or key < best_key:
                    best_key, best_state, best_epoch, best_rows = key, state, epoch, rows
            policy.load_state_dict(best_state)
            path = out/'models'/f'{name}_{seed}.pt'
            torch.save(best_state, path)
            selection[f'{name}_{seed}'] = dict(epoch=best_epoch, key=best_key,
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            save(out/'models'/f'{name}_{seed}_selection.json', dict(selected=selection[f'{name}_{seed}'], candidates=candidate_scores))
            selected[(name, seed)] = (policy, best_rows)
    protocol['selection'] = selection
    protocol['source_hashes'] = digest_sources()
    # Freeze protocol/model hashes BEFORE materializing official test images or
    # running audit-class histories. Both are forbidden for development tuning.
    save(out/'protocol.json', protocol)
    save(out/'locked_protocol.json', protocol)

    results = []
    for (_, _), (_, rows) in selected.items():
        results.extend(rows)
    for root, history in histories['dev']:
        for method in RULES:
            env = RetentionEnv(history, root, config, capture=True)
            results.append(collect(env, rule_chooser(method), [t['eval'] for t in history], 'dev', method))
            print(f'dev root={root} method={method} hits={sum(t["success"] for t in env.completed)}/{config["tasks"]}', flush=True)
    save(out/'results.json', results)
    test_images, test_labels = load_dataset(args.cache, 'test')
    for root in roots['audit']:
        history = history_for(images, labels, partition['audit'], root, config)
        audits = build_audit(test_images, test_labels, [t['classes'] for t in history])
        choices = [(name, None, rule_chooser(name)) for name in RULES]
        choices += [(name, seed, policy.chooser()) for (name, seed), (policy, _) in selected.items()]
        for name, seed, chooser in choices:
            env = RetentionEnv(history, root, config, capture=True)
            row = collect(env, chooser, audits, 'audit', name, seed)
            results.append(row)
            save(out/'results.json', results)
            print(f'audit root={root} method={name} seed={seed} hits={sum(t["success"] for t in env.completed)}/{config["tasks"]}', flush=True)
    if digest_sources() != protocol['source_hashes']:
        raise RuntimeError('source changed during the locked run')
    save(out/'total_cost.json', dict(controller_training=training_cost,
        calibration=[r['cost'] for r in calibration],
        model_selection='all candidate rollout and dev endpoint-evaluation costs stored in models/*_selection.json; selected dev rows also appear in results, do not double-count',
        execution_rows=len(results), wall_seconds=time.perf_counter()-started))
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()

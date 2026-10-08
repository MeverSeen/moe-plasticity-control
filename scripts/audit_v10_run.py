"""Audit v10 development/smoke completeness without loading any model."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    protocol = read(run / 'locked_protocol.json')
    summary = read(run / 'summary.json')
    cfg, execution = protocol['development'], protocol['execution']
    for relative, expected in execution['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source hash changed: {relative}')
    mode = execution['mode']
    expected_methods = ['global_0', 'global_0.5', 'global_1'] if mode == 'preflight' else [
        f'global_{m:g}' for m in cfg['actions']] + ['time_decay']
    branch_mode = mode in ['smoke', 'preflight', 'branches']
    expected_branches = set(itertools.product(cfg['checkpoint_steps'], cfg['branch_futures'], cfg['actions'])) if branch_mode else set()
    updates = tokens = eval_tokens = eval_calls = branch_count = 0
    for root in cfg['roots']:
        baselines = read(run / f'baselines-{root}.json')
        if [r['method'] for r in baselines] != expected_methods:
            raise ValueError('missing, repeated or extra baseline')
        evaluations = [baselines[0]['initial']]
        for row in baselines:
            if row['root'] != root or len(row['trajectory']) != cfg['max_steps']:
                raise ValueError('baseline root/length mismatch')
            if [s['step'] for s in row['trajectory']] != list(range(1, cfg['max_steps'] + 1)):
                raise ValueError('baseline updates are out of order')
            for step in row['trajectory']:
                updates += 1
                tokens += step['tokens']
                if not math.isfinite(step['train_CE']):
                    raise ValueError('non-finite training loss')
                if step['action'] == 0 and step['displacement']['model_displacement_l2'] != 0:
                    raise ValueError('zero action changed weights')
                if step['evaluation'] is not None:
                    evaluations.append(step['evaluation'])
        path = run / f'branches-{root}.json'
        branches = read(path) if path.exists() else []
        keys = [(r['checkpoint_step'], r['future'], r['action']) for r in branches]
        if len(keys) != len(set(keys)) or set(keys) != expected_branches:
            raise ValueError('missing, duplicate or unexpected branch')
        branch_count += len(branches)
        paired = {}
        for row in branches:
            if row['root'] != root or len(row['updates']) != cfg['branch_steps']:
                raise ValueError('branch root/length mismatch')
            key = (row['checkpoint_step'], row['future'])
            paired.setdefault(key, []).append(row)
            for step in row['updates']:
                updates += 1
                tokens += step['tokens']
                if row['action'] == 0 and step['displacement']['model_displacement_l2'] != 0:
                    raise ValueError('zero branch action changed weights')
            evaluations.append(row['endpoint'])
            if row['action'] == 0:
                if abs(row['A_selection_deterioration']) > 2e-5 or abs(row['B_selection_improvement']) > 2e-5:
                    raise ValueError('zero branch changed evaluation beyond numerical tolerance')
        for group in paired.values():
            future_batches = {tuple(u['batch_digest'] for u in row['updates']) for row in group}
            if len(future_batches) != 1:
                raise ValueError('actions received different future training batches')
            if any(row['features'] != group[0]['features'] for row in group):
                raise ValueError('actions received different checkpoint features')
        for evaluation in evaluations:
            if set(evaluation) != {'A_monitor', 'A_selection', 'B_monitor', 'B_selection'}:
                raise ValueError('unexpected evaluation role, including possible test access')
            for stats in evaluation.values():
                if not math.isfinite(stats['loss']) or stats['tokens'] <= 0:
                    raise ValueError('invalid token-weighted evaluation')
                eval_tokens += stats['tokens']
                eval_calls += 1
    cost = summary['physical_cost']
    for key, expected in [('learner_updates', updates), ('train_tokens', tokens),
                          ('evaluation_tokens', eval_tokens), ('evaluation_calls', eval_calls)]:
        if cost[key] != expected:
            raise ValueError(f'physical cost mismatch: {key}')
    if summary['feedback_benefit_tested'] or summary['final_test_evaluated']:
        raise ValueError('development harness must not claim policy benefit or confirmation')
    report = dict(status=summary['status'], roots=len(cfg['roots']), branches=branch_count,
                  validated_source_files=len(execution['source_hashes']), physical_updates=updates,
                  train_tokens=tokens, evaluated_tokens=eval_tokens, paired_batches_verified=True,
                  final_test_evaluated=False, feedback_benefit_tested=False)
    (run / 'integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

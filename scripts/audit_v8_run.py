"""Post-run integrity, cost and action diagnostics; never retunes v8 policies."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'experiments/v8_multilayer_retention'
sys.path.insert(0, str(SOURCE))
from analyze import row_metrics, validate_protocol_coverage


def totals(items):
    keys = ['learner_updates', 'train_examples', 'monitor_examples', 'audit_examples',
            'route_examples', 'controller_calls', 'seconds', 'audit_seconds']
    return {k: sum(x.get(k, 0) for x in items) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run/'locked_protocol.json').read_text())
    rows = json.loads((run/'results.json').read_text())
    validate_protocol_coverage(rows, protocol)
    for name, expected in protocol['source_hashes'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source mismatch: {name}')
    for name, selected in protocol['selection'].items():
        if hashlib.sha256((run/'models'/f'{name}.pt').read_bytes()).hexdigest() != selected['model_sha256']:
            raise ValueError(f'selected model mismatch: {name}')
    by_root = defaultdict(list)
    for row in rows:
        by_root[(row['split'], row['root'])].append(row)
        for key, capacity in [('replay', 'replay_capacity'), ('monitor', 'monitor_capacity')]:
            assert row['final_memory_counts'][key] <= protocol['config'][capacity]
        assert row['target_loss'] == protocol['config']['target_loss']
    for group in by_root.values():
        first = group[0]
        for row in group[1:]:
            assert row['classes'] == first['classes']
            assert row['tasks'][0] == first['tasks'][0]
            assert row['loss_matrix'][0][0] == first['loss_matrix'][0][0]
    audit = [r for r in rows if r['split']=='audit']
    grouped = defaultdict(list)
    for row in audit:
        grouped[row['method']].append(row)
    action_counts, metrics = {}, {}
    for name, group in grouped.items():
        counter = Counter()
        records = []
        for row in group:
            for event in row['events']:
                if event['task'] > 0:
                    counter.update(np.asarray(event['multipliers']).flatten().tolist())
            m = row_metrics(row)
            loss = row['loss_matrix']
            k, target = len(loss), row['target_loss']
            diagonal_hits = [loss[j][j] <= target for j in range(k)]
            records.append(dict(root=row['root'],
                monitor_sequence_success=m['sequence_success'],
                final_all_test_target=m['final_all_at_target'],
                mean_accuracy=m['final_mean_accuracy'], forgetting=m['final_positive_forgetting'],
                updates=row['cost']['learner_updates'],
                all_test_diagonals_at_target=all(diagonal_hits),
                diagonal_test_task_hits=sum(diagonal_hits),
                old_test_target_losses=sum(diagonal_hits[j] and loss[-1][j]>target for j in range(k-1)),
                initially_solved=sum(t['steps']==0 for t in row['tasks'])))
        action_counts[name] = {str(k):v for k,v in sorted(counter.items())}
        metrics[name] = dict(records=records, **{
            key:float(np.mean([x[key] for x in records])) for key in records[0] if key!='root'})
    # Calibration, all selection candidates, baseline dev and final audit are
    # distinct charges. Selected dev trajectories duplicate candidates, so omit.
    cost_file = json.loads((run/'total_cost.json').read_text())
    selection_cost = []
    candidate_action_counts = {}
    for path in sorted((run/'models').glob('*_selection.json')):
        item = json.loads(path.read_text())
        counts = {}
        for candidate in item['candidates']:
            selection_cost.extend(r['cost'] for r in candidate['rows'])
            counter = Counter(float(v) for r in candidate['rows'] for e in r['events']
                if e['task'] > 0 for layer in e['multipliers'] for v in layer)
            counts[str(candidate['epoch'])] = dict(sorted(counter.items()))
        candidate_action_counts[path.stem] = counts
    charges = dict(calibration=totals(cost_file['calibration']),
        controller_training=totals(list(cost_file['controller_training'].values())),
        development_selection=totals(selection_cost),
        development_rules=totals([r['cost'] for r in rows if r['split']=='dev' and r['policy_seed'] is None]),
        audit_rollouts=totals([r['cost'] for r in audit]))
    charges['total_without_double_count'] = totals(list(charges.values()))
    report = dict(status='post-run descriptive diagnostics, no tuning',
        validated_source_files=len(protocol['source_hashes']), validated_selected_models=len(protocol['selection']),
        validated_rows=len(rows), metrics=metrics, controlled_action_counts=action_counts,
        candidate_dev_action_counts=candidate_action_counts, selected_checkpoints=protocol['selection'],
        costs=charges, wall_seconds=cost_file['wall_seconds'],
        caveats=['Official test target misses include initial generalization gaps as well as subsequent forgetting.',
                 'Three audit roots reuse one 25-class pool; this is not population-wide inference.',
                 'Rule baselines have not undergone matched hyperparameter search; no learned-method advantage is established.'])
    destination = run/'analysis'
    destination.mkdir(exist_ok=True)
    (destination/'diagnostics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = list(metrics)
    labels = ['Freeze experts', 'Global 0.5', 'Global 1.0', 'Group rule', 'Local rule',
              'Coordinated rule', 'Learned local*', 'Learned coord.*']
    mapping = dict(zip(['global_0','global_0.5','global_1','group_rule','local_rule',
                       'coordinated_rule','learned_local','learned_coordinated'], labels))
    labels = [mapping[m] for m in order]
    y = np.arange(len(order))
    colors = ['#397a93' if not m.startswith('learned') else '#b88945' for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), gridspec_kw={'width_ratios':[1.4,1,1]})
    fig.subplots_adjust(left=.14, right=.98, bottom=.25, top=.78, wspace=.20)
    for ax, key, title, scale in zip(axes,
        ['forgetting','mean_accuracy','final_all_test_target'],
        ['Old-task loss increase', 'Final mean accuracy', 'All 5 test tasks at target'], [1,100,100]):
        values = [metrics[m][key]*scale for m in order]
        ax.barh(y, values, color=colors, height=.62)
        ax.set_yticks(y, labels if ax is axes[0] else ['']*len(y))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values)*1.32 if key=='forgetting' else (60 if key=='mean_accuracy' else 128))
        if key == 'final_all_test_target':
            ax.set_xticks([0, 25, 50, 75, 100])
        for i, value in enumerate(values):
            ax.text(value + ax.get_xlim()[1]*.02, i,
                f'{value:.4f}' if key=='forgetting' else f'{value:.1f}%', va='center', fontsize=9)
    fig.suptitle('v8 pilot: learned expert control has not shown an advantage', x=.14, ha='left', fontsize=15, fontweight='bold')
    fig.text(.14,.84,'CIFAR-100 / 5 tasks / 3 audit roots / fixed routes / 128 replay + 128 monitor samples', fontsize=10)
    fig.text(.14,.07,'All methods reached the online target. *Both learned arms selected initialization and executed Global 1.0.\nFreeze experts used 25% more learner updates than Global 1.0; actual compute is not matched.\nTest-target misses mix initial generalization gaps and later forgetting. Small pilot; no confirmatory inference.', fontsize=9, linespacing=1.6)
    fig.savefig(destination/'diagnostics.png', dpi=170)
    plt.close(fig)
    print(json.dumps(dict(validated_rows=len(rows), total_updates=charges['total_without_double_count']['learner_updates'],
                         selected_epochs={k:v['epoch'] for k,v in protocol['selection'].items()})))


if __name__ == '__main__':
    main()

"""Post-run v9 integrity and learning-curve diagnosis; no training or tuning."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/v9_learning_pressure'))
from analyze_pressure import row_metrics, validate_protocol_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run/'locked_protocol.json').read_text())
    rows = json.loads((run/'results.json').read_text())
    prefixes = json.loads((run/'prefixes.json').read_text())
    costs = json.loads((run/'total_cost.json').read_text())
    validate_protocol_coverage(rows, protocol)
    for relative, expected in protocol['source_hashes'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    assert protocol['dataset']['loaded_splits'] == ['train']
    assert protocol['dataset']['official_test_loaded'] is False
    by_prefix = {p['root']:p for p in prefixes}
    grouped, curves = defaultdict(list), defaultdict(lambda:defaultdict(list))
    for row in rows:
        row_metrics(row)
        prefix = by_prefix[row['root']]
        assert row['prefix_fingerprint'] == prefix['fingerprint']
        assert row['tasks'][0] == prefix['task']
        assert row['classes'] == prefix['classes']
        assert row['final_memory_counts']['replay'] <= protocol['config']['replay_capacity']
        assert row['final_memory_counts']['monitor'] <= protocol['config']['monitor_capacity']
        key = '|'.join(map(str, [row['mode'], row['scope'], row['method'], row['target_loss']]))
        grouped[key].append(row)
        for index, task in enumerate(row['tasks'][1:], 1):
            events = [e for e in row['events'] if e['task']==index]
            expected_steps = list(range(protocol['config']['interval'], task['steps']+1, protocol['config']['interval']))
            assert [e['steps'] for e in events] == expected_steps
            if row['mode']=='fixed_budget':
                curves[key][0].append([task['initial_train_loss'],task['initial_loss'],task['initial_eval_loss']])
                for event in events:
                    curves[key][event['steps']].append([event['train_loss'],event['new_loss'],event['eval_loss']])
    comparisons, diagnostics = {}, {}
    for key, group in grouped.items():
        metrics = [row_metrics(r) for r in group]
        tasks = [t for r in group for t in r['tasks'][1:]]
        diagnostics[key] = dict(roots=len(group),
            monitor_ever_tasks=sum(t['ever_hit'] for t in tasks),
            monitor_terminal_tasks=sum(t['success'] for t in tasks), suffix_tasks=len(tasks),
            first_hit_heldout_tasks=sum(t['ever_hit'] and t['first_hit_eval_loss']<=t['target_loss'] for t in tasks),
            heldout_diagonal_tasks=sum(r['loss_matrix'][j][j]<=r['target_loss'] for r in group for j in range(1,5)),
            mean_terminal_train=float(np.mean([t['terminal_train_loss'] for t in tasks])),
            mean_terminal_monitor=float(np.mean([t['terminal_loss'] for t in tasks])),
            mean_terminal_heldout=float(np.mean([t['terminal_eval_loss'] for t in tasks])),
            mean_min_monitor=float(np.mean([t['min_monitor_loss'] for t in tasks])),
            mean_min_heldout=float(np.mean([t['min_eval_loss'] for t in tasks])),
            mean_first_hit_steps_among_hits=float(np.mean([t['first_hit_step'] for t in tasks if t['ever_hit']])) if any(t['ever_hit'] for t in tasks) else None,
            mean_suffix_updates=float(np.mean([sum(t['steps'] for t in r['tasks'][1:]) for r in group])),
            mean_forgetting=float(np.mean([m['final_positive_forgetting'] for m in metrics])),
            mean_final_accuracy=float(np.mean([m['final_mean_accuracy'] for m in metrics])))
    for target in protocol['targets']:
        groups = [grouped[f'early_stop|full|global_{scale}|{target}'] for scale in ['0','1']]
        by_root = [{r['root']:row_metrics(r) for r in group} for group in groups]
        comparisons[str(target)] = {field: [dict(root=root, expert_minus_zero=by_root[1][root][field]-by_root[0][root][field])
            for root in protocol['roots']] for field in ['suffix_monitor_ever_rate','suffix_first_hit_heldout_rate',
                'suffix_mean_eval_loss','final_mean_accuracy','final_positive_forgetting']}
    logical_updates = sum(r['cost']['learner_updates'] for r in rows)
    reused_updates = sum((len(protocol['expected_cells'])-1)*p['cost']['learner_updates'] for p in prefixes)
    physical_updates = costs['total_without_reused_prefix']['learner_updates']
    assert logical_updates-reused_updates == physical_updates
    for key in ['learner_updates','train_examples','monitor_examples','diagnostic_train_examples',
                'diagnostic_eval_examples','route_examples','controller_calls']:
        logical = sum(r['cost'][key] for r in rows)
        reused = sum((len(protocol['expected_cells'])-1)*p['cost'][key] for p in prefixes)
        assert logical-reused == costs['total_without_reused_prefix'][key], key
    serial_curves = {key:{str(step):dict(mean=np.mean(values,axis=0).tolist(), task_curves=len(values))
        for step,values in sorted(steps.items())} for key,steps in curves.items()}
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    result = dict(status='post-run descriptive diagnosis; no checkpoint selection',
        validated_rows=len(rows), validated_sources=len(protocol['source_hashes']),
        physical_learner_updates=physical_updates, logical_learner_updates=logical_updates,
        development_only=True, conditions=diagnostics, paired_expert_update_effect=comparisons,
        fixed_budget_curves=serial_curves, cost=costs,
        warning='Minima are diagnostic and optimistically selected across windows, not deployable evaluation scores. All roots share development classes.')
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = [('full','global_0','Stem + heads; expert displacement = 0'),
             ('full','global_1','Stem + experts + heads'),
             ('frozen_stem','global_1','Experts + heads; stem frozen'),
             ('heads_only','global_0','Heads only; stem and experts frozen')]
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.16,top=.86,hspace=.35,wspace=.18)
    for ax,(scope,method,label) in zip(axes.flat,cases):
        curve = serial_curves[f'fixed_budget|{scope}|{method}|1.35']
        steps = list(map(int,curve))
        values = np.asarray([v['mean'] for v in curve.values()])
        for col,name,color in [(0,'Train','#397a93'),(1,'Online monitor','#b88945'),(2,'Held-out dev','#a04c69')]:
            ax.plot(steps,values[:,col],label=name,color=color,linewidth=2)
        ax.axhline(1.35,color='#555555',linestyle=':',linewidth=1,label='Target 1.35')
        ax.set_title(label,fontsize=11)
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Updates within current task')
        ax.set_ylabel('Mean cross-entropy')
        ax.legend(fontsize=8,loc='best')
    fig.suptitle('v9 development: where does additional training help?',fontsize=15,fontweight='bold',y=.98)
    fig.text(.08,.92,f'{len(protocol["roots"])} roots x 4 new tasks; common first-task checkpoint; exactly 80 updates per new task',fontsize=10)
    fig.text(.08,.05,'Curves average current-task losses over all roots and new tasks; the sequences have different preceding endpoints.\nHeld-out dev is diagnostic only. No official test images, learned policy, or confirmatory claim.',fontsize=9,linespacing=1.5)
    fig.savefig(out/'learning_curves.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(rows=len(rows),physical_updates=physical_updates,development_only=True)))


if __name__ == '__main__':
    main()

"""Report locked test comparisons without selecting or tuning on test roots."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def stats(rows):
    successful = [r for r in rows if r['success']]
    return {'n': len(rows), 'successes': len(successful),
            **{k: mean(rows, k) for k in ('test_F', 'monitor_F', 'new_test_loss',
                'new_monitor_loss', 'steps', 'decisions', 'path_energy',
                'net_displacement_sq', 'mean_multiplier', 'switches', 'seconds')},
            'successful_test_F': mean(successful, 'test_F'),
            'zero_update_histories': sum(r['steps'] == 0 for r in rows)}


def paired(a, b, keys, rng):
    # First average policy seeds within a root. Roots, not repeated executions
    # or policy seeds, are the units in this descriptive paired bootstrap.
    roots = sorted(set(r['root'] for r in a) & set(r['root'] for r in b))
    groups = {root: ([r for r in a if r['root'] == root],
                     [r for r in b if r['root'] == root]) for root in roots}
    common = [root for root, (aa, bb) in groups.items()
              if all(r['success'] for r in aa + bb)]
    output = {'all_roots': len(roots), 'common_success_roots': len(common),
              'differences_are': 'first minus second; three seeds averaged within root'}
    for key in keys:
        use = common if key == 'test_F' else roots
        values = np.array([mean(groups[root][0], key) - mean(groups[root][1], key) for root in use])
        if not len(values):
            output[key] = None; continue
        boot = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
        output[key] = {'mean': float(values.mean()),
                       'root_bootstrap_95pct': np.quantile(boot, [.025, .975]).tolist()}
    return output


def main(directory):
    out = Path(directory)
    with gzip.open(out / 'test_results.json.gz', 'rt') as f:
        data = json.load(f)
    completed = json.loads((out/'completion.json').read_text())
    fixed = json.loads((out/'baseline_selection.json').read_text())['selected_action']
    actions = [0., .25, .5, .75, 1.]
    fits = completed['fits']
    aggregates = {}
    for h in (1, 5):
        for reward in ('forgetting', 'movement'):
            for checkpoint in ('selected', 'last'):
                name = f'h{h}_{reward}_{checkpoint}'
                names = [f'{fit["name"]}_{checkpoint}' for fit in fits
                         if fit['hold'] == h and fit['reward'] == reward]
                aggregates[name] = [row for key in names for row in data[key]]
    stats_all = {key: stats(rows) for key, rows in {**data, **aggregates}.items()}
    selected = {k: v for k, v in aggregates.items() if k.endswith('_selected')}
    roots = sorted(set(r['root'] for rows in selected.values() for r in rows))
    common = [root for root in roots if all(r['success'] for rows in selected.values()
                                           for r in rows if r['root'] == root)]
    common_stats = {k: stats([r for r in rows if r['root'] in common])
                    for k, rows in {**selected,
                        f'fixed_{actions[fixed]}': data[f'fixed_{actions[fixed]}'],
                        'rule_h1': data['rule_h1'], 'rule_h5': data['rule_h5']}.items()}
    rng = np.random.default_rng(168991)
    pairs = [('h1_movement_selected', 'h1_forgetting_selected'),
             ('h5_movement_selected', 'h5_forgetting_selected'),
             ('h1_movement_selected', 'h5_movement_selected'),
             ('h1_forgetting_selected', 'h5_forgetting_selected')]
    comparisons = {a+'__minus__'+b: paired(selected[a], selected[b],
        ['test_F', 'path_energy', 'net_displacement_sq', 'steps', 'decisions'], rng)
        for a, b in pairs}
    fixed_name = f'fixed_{actions[fixed]}'
    comparisons['h1_movement_selected__minus__selected_fixed'] = paired(
        selected['h1_movement_selected'], data[fixed_name],
        ['test_F', 'path_energy', 'net_displacement_sq', 'steps', 'decisions'], rng)
    active = {r['root'] for r in data[fixed_name] if r['steps'] > 0}
    active_stats = {k: stats([r for r in rows if r['root'] in active])
                    for k, rows in {**selected, fixed_name: data[fixed_name]}.items()}
    # Post-hoc diagnostic only: for the same root, does a lower path energy
    # always mean less old-task damage among successful constant actions?
    concordant = discordant = tied = feasible_pairs = roots_with_reversal = 0
    for root in roots:
        candidates = [r for key, rows in data.items() if key.startswith('fixed_')
                      for r in rows if r['root'] == root and r['success']]
        reversal = False
        for i, first in enumerate(candidates):
            for second in candidates[i+1:]:
                feasible_pairs += 1
                sign = (first['path_energy'] - second['path_energy']) * (first['test_F'] - second['test_F'])
                if sign > 1e-14: concordant += 1
                elif sign < -1e-14: discordant += 1; reversal = True
                else: tied += 1
        roots_with_reversal += reversal
    report = {'source_test_sha256': hashlib.sha256((out/'test_results.json.gz').read_bytes()).hexdigest(),
              'selected_fixed_multiplier': actions[fixed], 'summary': stats_all,
              'common_all_four_selected_roots': common, 'common_summary': common_stats,
              'needs_learning_roots': sorted(active), 'needs_learning_summary': active_stats,
              'paired': comparisons, 'fit_selection': fits,
              'parameter_proxy_fixed_action_diagnostic': {
                  'note': 'Post-hoc test diagnostic, not used for policy or configuration selection.',
                  'successful_action_pairs': feasible_pairs, 'concordant': concordant,
                  'discordant': discordant, 'tied': tied,
                  'roots_with_at_least_one_reversal': roots_with_reversal},
              'limitations': 'Three training seeds; bootstrap resamples test roots only, not training runs. Success-conditioned F excludes failures; always report success and unconditional F alongside.'}
    (out/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    names = [f'fixed_{x}' for x in actions]+['rule_h1', 'rule_h5']+list(aggregates)
    table = ['|方法|达标|独立测试遗忘（全部）|成功历史遗忘|更新步数|决策次数|累计位移平方|净位移平方|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r = stats_all[name]
        successful = f'{r["successful_test_F"]:.6f}' if r['successful_test_F'] is not None else 'NA'
        table.append(f'|{name}|{r["successes"]}/{r["n"]}|{r["test_F"]:.6f}|{successful}|'
                     f'{r["steps"]:.1f}|{r["decisions"]:.1f}|{r["path_energy"]:.6f}|{r["net_displacement_sq"]:.4f}|')
    (out/'table.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table))
    print(json.dumps({'common_roots': len(common), 'diagnostic': report['parameter_proxy_fixed_action_diagnostic'],
                      'selected_rounds': {fit['name']: fit['selected_round'] for fit in fits},
                      'seconds': completed['seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory')
    main(parser.parse_args().directory)

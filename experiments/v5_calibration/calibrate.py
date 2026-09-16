"""Calibrate repeatable decision value without requiring low-action failure.

Finite-plan privileged selection is a diagnostic, not a deployed feedback policy.
Select configuration and baselines on calibration histories, then lock and confirm.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

import branching as b

SETTINGS = [dict(name=f't{target:.2f}_s{steps}', target=target, max_steps=steps)
            for target in [.08, .04, .02] for steps in [300, 100]]
SPLITS = dict(calibration=list(range(510000, 510024)), confirmation=list(range(520000, 520064)))
PILOT_SPLITS = dict(calibration=list(range(540000, 540004)), confirmation=list(range(540100, 540106)))


def plans():
    result = [dict(name=f'fixed_{a}', kind='fixed', action=a) for a in range(5)]
    for kind, thresholds in [('time', [.1, .3, .6]), ('progress', [.25, .5, .75])]:
        for early, late in [(0, 4), (4, 0)]:
            for threshold in thresholds:
                result.append(dict(name=f'{kind}_{early}_{late}_{threshold}', kind=kind,
                                   early=early, late=late, threshold=threshold))
    return result


def chooser(plan):
    switched = False

    def decide(env):
        nonlocal switched
        if plan['kind'] == 'fixed':
            return plan['action']
        if plan['kind'] == 'time':
            progress = env.steps / env.c['max_steps']
        elif plan['kind'] == 'progress':
            initial_gap = max(math.log(max(env.initial_loss, 1e-8) / env.c['target']), 1e-6)
            progress = 1 - math.log(max(env.current_loss, 1e-8) / env.c['target']) / initial_gap
        else:
            raise ValueError(plan['kind'])
        switched = switched or progress >= plan['threshold']
        return plan['late'] if switched else plan['early']
    return decide


def paired_future(env, future):
    """Use a common stream prefix even when the candidate step budget changes."""
    result = b.clone_env(env)
    rng = b.base.core.gen(env.world['seed']*100003 + env.steps*97 + future*997 + 8001)
    span = max(300, env.c['max_steps']) - env.steps
    take = env.c['max_steps'] - env.steps
    ni = torch.randint(len(env.new[0]), (span, env.c['batch']), generator=rng)
    oi = torch.randint(len(env.old[0]), (span, env.c['old_batch']), generator=rng)
    result.new_indices[env.steps:] = ni[:take]
    result.old_indices[env.steps:] = oi[:take]
    return result


def run_plan(item, config, plan, future):
    env = paired_future(b.new_env(item, config), future)
    ref = b.evaluator_reference(env)
    decide = chooser(plan)
    initial_done = env.done
    trace = []
    while not env.done:
        action = int(decide(env))
        trace.append([int(env.steps), action])
        env.advance(action)
    row = b.endpoint(env, ref)
    row.update(root=int(env.world['seed']), plan=plan['name'], future=future,
        initial_new_loss=float(env.initial_loss), initially_done=initial_done,
        protected_mass=float(env.protected_mass), protected_fraction=float(env.mask.float().mean()),
        decisions=len(trace), action_trace=trace)
    return row


def evaluate(out, split, seeds, settings, config, candidates):
    """Pair all candidates and settings on the same history and future stream."""
    paths = {s['name']: out / f"{split}_{s['name']}.jsonl" for s in settings}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError('Refusing to overwrite trajectories')
    streams = {name: path.open('w') for name, path in paths.items()}
    records = {name: [] for name in paths}
    start = time.time()
    try:
        for index, seed in enumerate(seeds):
            # Same old-task training and data distribution for all target/budget candidates.
            item = b.make_bank([seed], config)[0]
            for setting in settings:
                current = dict(config, target=setting['target'], levels=[setting['target']],
                               max_steps=setting['max_steps'])
                for future in range(2):
                    for plan in candidates:
                        row = run_plan(item, current, plan, future)
                        records[setting['name']].append(row)
                        streams[setting['name']].write(json.dumps(row) + '\n')
                streams[setting['name']].flush()
            if (index + 1) % 4 == 0 or index + 1 == len(seeds):
                print(json.dumps(dict(stage=split, histories=index + 1,
                    settings=len(settings), seconds=round(time.time() - start, 1))), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    b.save(out / f'{split}_cost.json', dict(seconds=time.time() - start,
        model_updates=sum(r['steps'] for rows in records.values() for r in rows),
        charged_work=sum(r['work'] for rows in records.values() for r in rows),
        rollouts=sum(map(len, records.values())),
        note='All candidate rollouts are offline calibration/diagnostic computation. Work excludes shared old-task pretraining; wall time includes it.'))
    return records


def cube(rows, candidates):
    roots = np.array(sorted({r['root'] for r in rows}))
    keyed = {(r['root'], r['plan'], r['future']): r for r in rows}
    values = {field: np.array([[[float(keyed[root, plan['name'], rep][field])
                for rep in range(2)] for plan in candidates] for root in roots])
              for field in ['success', 'forgetting', 'cost', 'work', 'steps', 'new_monitor_loss', 'target']}
    return roots, values


def best_index(values, allowed, roots=None, future=0):
    """Reachability, then successful mean raw forgetting, then work; stable ties."""
    if roots is None:
        roots = np.arange(len(values['success']))
    def key(index):
        hit = values['success'][roots, index, future] > 0
        forgetting = values['forgetting'][roots, index, future]
        return (-int(hit.sum()), float(forgetting[hit].mean()) if hit.any() else float('inf'),
                float(values['work'][roots, index, future].mean()), index)
    return min(allowed, key=key)


def history_choices(values, allowed):
    return np.array([best_index(values, allowed, roots=np.array([i]), future=0)
                     for i in range(len(values['success']))], dtype=int)


def selected(values, actions, future=None):
    indices = np.arange(len(values['success']))
    actions = np.broadcast_to(actions, (len(indices),))
    result = {field: value[indices, actions] for field, value in values.items()}
    if future is not None:
        result = {field: value[:, future:future+1] for field, value in result.items()}
    return result


def interval(values, draws=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=[None, None], histories=0)
    rng = np.random.default_rng(917)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), histories=len(values))


def compare(first, second, root_mask=None):
    n = len(first['success'])
    root_mask = np.ones(n, dtype=bool) if root_mask is None else np.asarray(root_mask, dtype=bool)
    a = {k: v[root_mask] for k, v in first.items()}
    c = {k: v[root_mask] for k, v in second.items()}
    common = (a['success'] > 0) & (c['success'] > 0)
    differences = [v[m].mean() for v, m in zip(a['forgetting'] - c['forgetting'], common) if m.any()]
    tight = common & (a['new_monitor_loss'] >= .95*a['target']) & (c['new_monitor_loss'] >= .95*c['target'])
    tight_differences = [v[m].mean() for v, m in zip(a['forgetting'] - c['forgetting'], tight) if m.any()]
    return dict(reach_difference=interval((a['success'] - c['success']).mean(1)),
        cost_difference=interval((a['cost'] - c['cost']).mean(1)),
        joint_forgetting_difference=interval(differences),
        tight_endpoint_forgetting_difference=interval(tight_differences),
        work_difference=interval((a['work'] - c['work']).mean(1)),
        joint_success_pairs=int(common.sum()))


def stats(data):
    hits = data['success'] > 0
    conditional = [f[h].mean() for f, h in zip(data['forgetting'], hits) if h.any()]
    return dict(reach=interval(data['success'].mean(1)), cost=interval(data['cost'].mean(1)),
                successful_forgetting=interval(conditional), mean_steps=float(data['steps'].mean()),
                mean_work=float(data['work'].mean()))


def analyze_rows(rows, candidates, fixed_choices=None):
    roots, values = cube(rows, candidates)
    constant = [i for i, p in enumerate(candidates) if p['kind'] == 'fixed']
    time = [i for i, p in enumerate(candidates) if p['kind'] in ['fixed', 'time']]
    progress = [i for i, p in enumerate(candidates) if p['kind'] in ['fixed', 'progress']]
    all_plans = list(range(len(candidates)))
    chosen = fixed_choices or dict(best_constant=best_index(values, constant),
        best_time=best_index(values, time), best_progress=best_index(values, progress),
        best_nonfeedback=best_index(values, all_plans))
    methods = {name: selected(values, index) for name, index in chosen.items()}
    oracle_constant = history_choices(values, constant)
    oracle_plan = history_choices(values, all_plans)
    cross_constant = selected(values, oracle_constant, future=1)
    cross_plan = selected(values, oracle_plan, future=1)
    ref_constant = selected(values, chosen['best_constant'], future=1)
    ref_nonfeedback = selected(values, chosen['best_nonfeedback'], future=1)
    # Cohorts use only selection stream, never filter on evaluation-stream success.
    all_constants_feasible = (values['success'][:, constant, 0] > 0).all(1)
    budget_pressure = (values['success'][:, 0, 0] == 0) & (values['success'][:, 4, 0] > 0)
    seen = {r['root']: r for r in rows if r['future'] == 0 and r['plan'] == candidates[0]['name']}
    active = np.array([not seen[root]['initially_done'] for root in roots])
    nontrivial = active & np.array([seen[root]['protected_mass'] > 1e-8 for root in roots])
    result = dict(histories=len(roots), selected=chosen,
        selected_names={name: candidates[i]['name'] for name, i in chosen.items()},
        methods={name: stats(data) for name, data in methods.items()},
        all_constant_methods={candidates[i]['name']: stats(selected(values, i)) for i in constant},
        paired={f'{name}_vs_constant': compare(data, methods['best_constant'])
                for name, data in methods.items() if name != 'best_constant'},
        cross_future=dict(constant_oracle_vs_constant=compare(cross_constant, ref_constant),
            plan_oracle_vs_nonfeedback=compare(cross_plan, ref_nonfeedback),
            plan_oracle_vs_constant=compare(cross_plan, ref_constant),
            all_feasible_constant_oracle_vs_constant=compare(cross_constant, ref_constant, all_constants_feasible),
            all_feasible_active_constant_oracle_vs_constant=compare(cross_constant, ref_constant, all_constants_feasible & nontrivial),
            all_feasible_plan_oracle_vs_nonfeedback=compare(cross_plan, ref_nonfeedback, all_constants_feasible),
            active_plan_oracle_vs_nonfeedback=compare(cross_plan, ref_nonfeedback, nontrivial)),
        diagnostics=dict(initially_done=int((~active).sum()), active_protected=int(nontrivial.sum()),
            all_constants_feasible_selection_stream=int(all_constants_feasible.sum()),
            all_feasible_active_protected=int((all_constants_feasible & nontrivial).sum()),
            low_fails_high_succeeds_selection_stream=int(budget_pressure.sum()),
            constant_oracle_counts=np.bincount(oracle_constant, minlength=len(candidates)).tolist(),
            plan_oracle_counts=np.bincount(oracle_plan, minlength=len(candidates)).tolist()),
        oracle_selection=[dict(root=int(root), constant=int(c), plan=int(p))
                          for root, c, p in zip(roots, oracle_constant, oracle_plan)])
    return result


def select_setting(summaries, settings):
    scores = []
    for setting in settings:
        s = summaries[setting['name']]
        # Feasibility is checked on the calibration selection stream only below via saved key.
        eligible = s['selection_stream_oracle_reach'] >= .8 and s['selection_stream_active_fraction'] >= .5
        gain = -s['cross_future']['plan_oracle_vs_nonfeedback']['cost_difference']['mean']
        scores.append(dict(name=setting['name'], eligible=eligible, headroom_score=gain))
    eligible = [row for row in scores if row['eligible']]
    selected_row = max(eligible, key=lambda row: row['headroom_score']) if eligible else scores[0]
    return selected_row['name'], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    args.out.mkdir(parents=True)
    started = time.time()
    config = copy.deepcopy(b.base.CONFIG)
    settings = copy.deepcopy(SETTINGS if not args.pilot else SETTINGS[:2])
    splits = copy.deepcopy(PILOT_SPLITS if args.pilot else SPLITS)
    candidates = plans()
    plan = dict(config=config, settings=settings, candidates=candidates, splits=splits, pilot=args.pilot,
        futures=2, selection_future=0, diagnostic_evaluation_future=1,
        scope='Calibration of repeatable state-conditioned/temporal decision value; low-action failure is not required.',
        selection='Choose baselines with future 0 on calibration roots: reachability, then successful raw forgetting, then work. Eligible settings have >=80% finite-plan oracle reach and >=50% initially active roots; no universal static feasibility required. Select maximum cross-future plan-oracle J gain over the best nonfeedback plan; if none eligible confirm reference configuration without claiming it qualifies.',
        confirmation='Freeze one setting and four global baselines before creating any confirmation world. Confirmation cannot alter choices.',
        oracle='For each history, use full future-0 evaluator outcomes to choose one finite plan; evaluate on independent future 1. Privileged diagnostic, not an executable feedback controller or global optimum.',
        objective='Report reach first and paired raw forgetting at common target, including work tie-breaks. J=2(1-hit)+hit*F/(F+.01) is only a bounded diagnostic/setting-selection proxy.',
        all_feasible='Cohort defined by all five constant plans succeeding in selection stream 0, then evaluated unfiltered on stream 1.',
        fixed_controls='Same model capacity, memory, router, grouping, optimizer and exogenous minibatches within each setting. Target and budget vary across settings only.',
        teacher='One consistent nonlinear teacher across tasks; no new conflicting-label task generator.',
        limitations=['Finite 17-plan class cannot establish the best possible dynamic policy.',
                     'Per-history privileged plan choice need not require feedback during training.',
                     'Two futures and finite histories limit uncertainty estimates.',
                     'No learned controller, new state representation, router dynamics or RL in this calibration.'])
    b.save(args.out / 'plan.json', plan)
    b.save(args.out / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, device='cpu', threads=1))
    raw = evaluate(args.out, 'calibration', splits['calibration'], settings, config, candidates)
    summaries = {}
    for setting in settings:
        name = setting['name']
        s = analyze_rows(raw[name], candidates)
        roots, values = cube(raw[name], candidates)
        s['selection_stream_reach'] = float(values['success'][:, s['selected']['best_nonfeedback'], 0].mean())
        s['selection_stream_oracle_reach'] = float(values['success'][:, :, 0].max(1).mean())
        s['selection_stream_active_fraction'] = 1 - s['diagnostics']['initially_done'] / len(roots)
        summaries[name] = s
    selected_name, selection_scores = select_setting(summaries, settings)
    chosen_setting = next(s for s in settings if s['name'] == selected_name)
    fixed = summaries[selected_name]['selected']
    b.save(args.out / 'calibration_summary.json', summaries)
    lock = dict(plan, selected_setting=chosen_setting, selected_baselines=fixed,
        selected_baseline_names=summaries[selected_name]['selected_names'], selection_scores=selection_scores,
        source_hashes={p.name: b.file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    b.save(args.out / 'locked_protocol.json', lock)
    print(json.dumps(dict(stage='locked_before_confirmation', setting=chosen_setting,
        baselines=lock['selected_baseline_names'])), flush=True)
    confirm = evaluate(args.out, 'confirmation', splits['confirmation'], [chosen_setting], config, candidates)
    result = analyze_rows(confirm[selected_name], candidates, fixed)
    b.save(args.out / 'confirmation_summary.json', result)
    b.save(args.out / 'total_cost.json', dict(seconds=time.time() - started))
    print(json.dumps(dict(stage='complete', seconds=round(time.time() - started, 1))), flush=True)


if __name__ == '__main__':
    main()

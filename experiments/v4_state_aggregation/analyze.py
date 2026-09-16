"""Paired history-level estimates; no final-test result feeds model selection."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import branching as b


def interval(values, seed=731, draws=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=[None, None], histories=0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                histories=len(values))


def clustered(values, roots, mask=None):
    result = []
    for root in np.unique(roots):
        v = np.asarray(values)[roots == root]
        if mask is not None:
            v = v[np.asarray(mask)[roots == root]]
        if v.size:
            result.append(v.mean())
    return np.asarray(result)


def selected_data(actions, hit, forget, cost):
    idx = np.arange(len(hit))
    return [np.stack([arr[idx, a] for a in actions], axis=1) for arr in [hit, forget, cost]]


def stats(data, roots):
    hit, forget, cost = data
    return dict(reach=interval(clustered(hit, roots)), cost=interval(clustered(cost, roots)),
        successful_forgetting=interval(clustered(forget, roots, hit.astype(bool))),
        all_endpoint_forgetting=interval(clustered(forget, roots)))


def compare(a, c, roots):
    ah, af, ac = a
    ch, cf, cc = c
    ah, ch = np.broadcast_arrays(ah, ch)
    af, cf = np.broadcast_arrays(af, cf)
    ac, cc = np.broadcast_arrays(ac, cc)
    common = (ah > 0) & (ch > 0)
    return dict(reach_difference=interval(clustered(ah - ch, roots)),
        cost_difference=interval(clustered(ac - cc, roots)),
        joint_forgetting_difference=interval(clustered(af - cf, roots, common)),
        all_forgetting_difference=interval(clustered(af - cf, roots)),
        joint_success_pairs=int(common.sum()))


def comparisons(bundles, roots, rounds):
    pairs = []
    for prefix in ['bc'] + [f'{kind}_r{r}' for r in range(1, rounds + 1)
                            for kind in ['aggregate', 'matched']]:
        pairs.append((f'{prefix}_feedback', f'{prefix}_progress'))
    for mode in ['progress', 'feedback']:
        agg = f'aggregate_r{rounds}_{mode}'
        pairs += [(agg, f'bc_{mode}'), (agg, f'matched_r{rounds}_{mode}')]
    results = {f'{a}_vs_{c}': compare(bundles[a], bundles[c], roots) for a, c in pairs}
    af, ap = bundles[f'aggregate_r{rounds}_feedback'], bundles[f'aggregate_r{rounds}_progress']
    bf, bp = bundles['bc_feedback'], bundles['bc_progress']
    results['feedback_gain_change_after_aggregation'] = {
        'cost_difference_in_differences': interval(clustered((af[2] - ap[2]) - (bf[2] - bp[2]), roots)),
        'note': 'Negative means aggregation improves the feedback-minus-progress cost gap.'}
    return results


def offline(run, domain, groups, lock):
    states = b.load_states(run / f'test_{domain}_branches.jsonl')
    _, roots, hit, forget, cost = b.arrays(states)
    predictions = json.loads((run / f'test_{domain}_predictions.json').read_text())
    assert predictions['state_ids'] == [s['state_id'] for s in states]
    actions = predictions['actions']
    bundles = {group: selected_data([actions[name] for name in names], hit, forget, cost)
               for group, names in groups.items()}
    idx = np.arange(len(states))
    oracle = cost.mean(2).argmin(1)
    summaries = {group: stats(data, roots) for group, data in bundles.items()}
    for group, names in groups.items():
        selected = np.array([actions[name] for name in names]).T
        summaries[group]['action_mae_to_in_sample_oracle'] = interval(
            clustered(np.abs(selected - oracle[:, None]) * .25, roots))
        regret = np.stack([cost[idx, actions[name]].mean(1) - cost.mean(2).min(1) for name in names], axis=1)
        summaries[group]['in_sample_cost_regret'] = interval(clustered(regret, roots))
    cross_action = cost[:, :, 0].argmin(1)
    cross = selected_data([cross_action], hit[:, :, 1:], forget[:, :, 1:], cost[:, :, 1:])
    static = selected_data([np.full(len(states), lock['static_branch_action'])],
                           hit[:, :, 1:], forget[:, :, 1:], cost[:, :, 1:])
    return dict(states=len(states), histories=len(np.unique(roots)), methods=summaries,
        paired=comparisons(bundles, roots, lock['rounds']),
        cross_future_oracle_vs_static=compare(cross, static, roots),
        note='In-sample oracle regret is optimistic; cross-future oracle is privileged diagnostic only.')


def online(run, groups, lock):
    rows = json.loads((run / 'test_online.json').read_text())
    keyed = {(r['method'], r['root']): r for r in rows}
    roots = np.array(lock['splits']['test'])
    online_groups = dict(groups, best_static=['best_static'], hand_feedback=['hand_feedback'],
                         full_update=['full_update'])
    bundles = {group: [np.array([[[float(keyed[name, root][field])] for name in names] for root in roots])
                      for field in ['success', 'forgetting', 'cost']]
               for group, names in online_groups.items()}
    summary = {name: stats(data, roots) for name, data in bundles.items()}
    for group, names in online_groups.items():
        subset = [r for r in rows if r['method'] in names]
        summary[group].update(mean_steps=float(np.mean([r['steps'] for r in subset])),
            mean_work=float(np.mean([r['work'] for r in subset])),
            mean_multiplier=float(np.mean([r['mean_multiplier'] for r in subset if r['decisions']])))
    paired = comparisons(bundles, roots, lock['rounds'])
    f, p = f"aggregate_r{lock['rounds']}_feedback", f"aggregate_r{lock['rounds']}_progress"
    primary = paired[f'{f}_vs_{p}']
    active = np.array([keyed['full_update', r]['initial_new_loss'] > lock['config']['target'] for r in roots])
    primary['active_histories'] = compare([a[active] for a in bundles[f]],
                                          [a[active] for a in bundles[p]], roots[active])
    tight = []
    for root in roots:
        differences = []
        for seed in lock['policy_seeds']:
            fr, pr = keyed[f'{f}_{seed}', root], keyed[f'{p}_{seed}', root]
            if fr['success'] and pr['success'] and min(fr['new_monitor_loss'], pr['new_monitor_loss']) >= .95 * lock['config']['target']:
                differences.append(fr['forgetting'] - pr['forgetting'])
        if differences:
            tight.append(np.mean(differences))
    primary['tight_endpoint_forgetting'] = interval(tight)
    primary['per_policy_seed'] = {}
    for seed in lock['policy_seeds']:
        individual = [[np.array([[[float(keyed[f'{group}_{seed}', root][field])]] for root in roots])
                       for field in ['success', 'forgetting', 'cost']] for group in [f, p]]
        primary['per_policy_seed'][str(seed)] = compare(*individual, roots)
    return dict(methods=summary, paired=paired, initial_already_success=int((~active).sum()),
                histories=len(roots)), rows


def continuation_analysis(run):
    rows = b.load_states(run / 'test_continuations.jsonl')
    families = {mode: [r['families'][mode] for r in rows] for mode in ['constant', 'fixed', 'adaptive']}
    arrays = {mode: b.arrays(states) for mode, states in families.items()}
    roots = arrays['adaptive'][1]
    _, _, hit, forget, cost = arrays['adaptive']
    choices = {mode: value[4][:, :, 0].argmin(1) for mode, value in arrays.items()}
    bundles = {mode: selected_data([selected], hit[:, :, 1:], forget[:, :, 1:], cost[:, :, 1:])
               for mode, selected in choices.items()}
    overlap = {}
    for mode in ['constant', 'fixed']:
        ca, aa = arrays[mode][4][:, :, 0], cost[:, :, 0]
        tied_c = ca <= ca.min(1, keepdims=True) + 1e-7
        tied_a = aa <= aa.min(1, keepdims=True) + 1e-7
        overlap[mode] = dict(argmin_disagreement=float(np.mean(choices[mode] != choices['adaptive'])),
            disjoint_best_action_sets=float(np.mean(~(tied_c & tied_a).any(1))))
    return dict(states=len(rows), same_adaptive_future_methods={mode: stats(data, roots) for mode, data in bundles.items()},
        paired={f'adaptive_vs_{mode}': compare(bundles['adaptive'], bundles[mode], roots)
                for mode in ['constant', 'fixed']}, disagreements=overlap,
        note='Select first action with privileged future-0 branch labels, score on independent future 1 under one identical frozen adaptive continuation. Not global optimal planning or a deployed controller.')


def audit(run, lock, online_rows):
    sources = Path(__file__).parent
    source_ok = all(b.file_hash(sources / name) == value for name, value in lock['source_hashes'].items())
    policy_ok = all(b.file_hash(run / name) == value for name, value in lock['policy_hashes'].items())
    sets = [set(v) for v in lock['splits'].values()]
    disjoint = sum(map(len, sets)) == len(set.union(*sets))
    collections = [json.loads(p.read_text()) for p in sorted(run.glob('collection_r*/collection_cost.json'))]
    matched = all(r['aggregate'] == r['matched'] for c in collections for r in c['roots'])
    labels_ok = True
    for split in ['train', 'dev']:
        labels_ok &= all(r['root'] in lock['splits'][split]
                         for r in b.load_states(run / 'initial' / f'{split}_branches.jsonl'))
    for path in run.glob('collection_r*/*_branches.jsonl'):
        labels_ok &= all(r['root'] in lock['splits']['train'] for r in b.load_states(path))
    budgets_ok = all(r['work'] <= r['budget'] + 1e-6 and r['steps'] <= lock['config']['max_steps'] for r in online_rows)
    thresholds_ok = all(r['target'] == lock['config']['target'] and
        bool(r['success']) == (r['new_monitor_loss'] <= r['target']) for r in online_rows)
    result = dict(source_hashes_match=source_ok, policy_hashes_match=policy_ok,
        disjoint_root_splits=disjoint, matched_queries_per_root_round=matched,
        training_label_roots_valid=bool(labels_ok), online_budgets_valid=budgets_ok,
        recorded_thresholds_valid=thresholds_ok,
        matched_duplicate_queries=sum(c['costs']['matched']['duplicate_queries'] for c in collections),
        collection_costs=collections)
    assert all([source_ok, policy_ok, disjoint, matched, labels_ok, budgets_ok, thresholds_ok]), result
    return result


def estimate(value):
    if value['mean'] is None:
        return '无可比较历史'
    lo, hi = value['ci95']
    return f"{value['mean']:.6f}，95% CI [{lo:.6f}, {hi:.6f}]"


def write_report(out, summary, lock):
    rounds = lock['rounds']
    key = f'aggregate_r{rounds}_feedback_vs_aggregate_r{rounds}_progress'
    primary = summary['online']['paired'][key]
    reach = primary['reach_difference']['mean']
    ci = primary['joint_forgetting_difference']['ci95']
    positive = reach >= 0 and ci[1] is not None and ci[1] < 0
    verdict = ('本轮观察到数据聚合后反馈策略的配对遗忘优势；仍需结合达标率与其他对照解释。'
               if positive else '本轮尚未验证数据聚合能让反馈策略在不降低达标率的情况下，稳定优于进度策略。')
    text = ['# MoE 状态聚合与后续策略实验', '', '**' + verdict + '**', '',
        '## 实验设置', '',
        f"- 校准/训练/开发/最终测试历史：{'/'.join(str(len(lock['splits'][k])) for k in ['calibration','train','dev','test'])}；所有最终测试历史为本轮新种子。",
        f"- 新任务阈值为 **{lock['config']['target']}**，最多 300 更新及共同工作量代理预算。v3 配置实际上也是 0.08，其报告中的 0.06 是文字错误。",
        '- 固定 6-expert 非线性 MoE、固定路由及保护组；动作仍为五档 Adam 参数位移倍率，每 5 步决策。',
        '- Progress 已有 ΔB、上次动作和相同初始信息。Feedback 仅额外看到旧监测遗忘与近期旧 loss 变化。',
        f'- BC 使用初始固定/随机路径数据；Aggregate 进行 {rounds} 轮学习器访问状态聚合；Matched 使用相同逐历史查询数的非学习策略状态。',
        '- Aggregate 汇集所有 progress/feedback 初始化访问的状态，再给双方同一数据池；按完整状态去重，每条轨迹最多取 3 个时间分散的快照。',
        '- 分支教师和拟合算法保持不变：先试当前动作 H 步，再采用校准后固定的 continuation 至终点；两个独立未来流、每流各动作配对。',
        '- 相同 3 个初始化、网络、最终拟合 epochs、开发集检查点/可行性阈值选择方法。BC 仅拟合一次；聚合额外采样、标注及中间拟合成本单列。',
        '- 独立评估样本只作为训练/开发世界的离线标签；最终测试标签仅作冻结策略评分及特权诊断。',
        '- 主要终点先看达标率，再看共同成功的配对遗忘；J=2(1-hit)+hit·F/(F+0.01) 是包含失败的诊断代价。',
        '- 区间按根历史配对 bootstrap，先在历史内平均策略初始化。它不完整覆盖重新训练的不确定性。', '',
        '## 闭环结果', '', '| 策略 | 达标率 | 成功子集遗忘 MSE | 代价 J | 平均步数（含失败） |',
        '|---|---:|---:|---:|---:|']
    order = ['full_update', 'best_static', 'hand_feedback', 'bc_progress', 'bc_feedback']
    order += [f'{kind}_r{r}_{mode}' for r in range(1, rounds + 1)
              for kind in ['matched', 'aggregate'] for mode in ['progress', 'feedback']]
    for name in order:
        row = summary['online']['methods'][name]
        fm = row['successful_forgetting']['mean']
        text.append(f"| {name} | {100*row['reach']['mean']:.2f}% | {fm:.6f} | {row['cost']['mean']:.6f} | {row['mean_steps']:.2f} |")
    text += ['', '成功子集可能不同，上表不能单独用于宣称遗忘优势。以下差值均为前者减后者，遗忘/代价越小越好。', '',
        '## 主要比较：最终轮 Feedback − Progress', '',
        '- 达标率差：' + estimate(primary['reach_difference']),
        '- 共同成功遗忘差：' + estimate(primary['joint_forgetting_difference']),
        '- 全样本代价差：' + estimate(primary['cost_difference']),
        '- 排除初始已达标历史的遗忘差：' + estimate(primary['active_histories']['joint_forgetting_difference']),
        '- 双方端点都落在 [0.95×target, target] 内的遗忘差：' + estimate(primary['tight_endpoint_forgetting']), '',
        '## 数据聚合是否优于增加普通标签', '']
    for mode in ['progress', 'feedback']:
        for other in ['bc', f'matched_r{rounds}']:
            name = f'aggregate_r{rounds}_{mode}_vs_{other}_{mode}'
            row = summary['online']['paired'][name]
            text += [f"- {name}：达标率差 {estimate(row['reach_difference'])}；遗忘差 {estimate(row['joint_forgetting_difference'])}；J 差 {estimate(row['cost_difference'])}。"]
    interaction = summary['online']['paired']['feedback_gain_change_after_aggregation']['cost_difference_in_differences']
    text += ['- 聚合前后 Feedback−Progress 的 J 差变化：' + estimate(interaction), '',
        '## 离线决策代价', '', '| 测试状态来源 | Feedback−Progress J 差（最终聚合轮） | 共同成功遗忘差 |', '|---|---|---|']
    for domain in ['behavior', 'visited']:
        row = summary['offline'][domain]['paired'][key]
        text.append(f"| {domain} | {estimate(row['cost_difference'])} | {estimate(row['joint_forgetting_difference'])} |")
    text += ['', 'Visited 状态来自冻结的 BC 和最终聚合策略的实际轨迹，数据不会回流训练。动作 MAE、事后最优代价差及跨未来复测保留在 summary.json；不把 MAE 改善当作控制收益。', '',
        '## 后续策略不匹配诊断', '',
        '同一状态比较：Qconstant（当前动作一直用到终点）、Qfixed（H 步后转校准固定策略）、Qadaptive（H 步后转预先指定的最终反馈策略）。三者都到终点，Qconstant 不是短视 oracle。',
        '用未来流 0 的特权分支标签选第一动作，在独立未来流 1 上统一接同一个 Qadaptive 后续策略。这样比较的是第一动作的选择，不混用不同后续策略下的代价。']
    for mode in ['constant', 'fixed']:
        row = summary['continuation']['paired'][f'adaptive_vs_{mode}']
        disagree = summary['continuation']['disagreements'][mode]
        text += [f"- adaptive vs {mode}：首动作 argmin 不同 {100*disagree['argmin_disagreement']:.1f}%，最优动作集合不相交 {100*disagree['disjoint_best_action_sets']:.1f}%；跨未来 J 差 {estimate(row['cost_difference'])}。"]
    text += ['', '这里的 adaptive 是一个冻结学习策略，既非最优规划也非全知 oracle。动作排序变化只说明 continuation 会影响标签，不能单独证明 RL 必要；所有分支计算均是额外离线成本。', '',
        '## 审计与成本', '',
        f"- 全流程墙钟时间：{summary['total_cost']['seconds']:.1f} 秒；这只是本机小模型运行时间。",
        f"- 配对查询数校验、源码/权重哈希、数据划分、在线预算及实际终点阈值校验均通过。Matched 因候选不足重复查询 {summary['audit']['matched_duplicate_queries']} 次。",
        '- 查询数量相同不表示离线更新工作量相同；各轮 collection_cost.json 保存分支与轨迹成本，fit_cost.json 保存各次拟合时间。在线所有方法承担相同观测/评分计算额度，属于信息消融。',
        '- 原始标签、轨迹、预测、锁定协议与机器可读区间均保留。正式测试已经查看；后续调参必须换新测试历史。',
        '- 本轮没有引入新的状态表示、RL 算法或 learned router；结论限于当前环境与教师。', '',
        '数据聚合动机参考 [Ross、Gordon、Bagnell，2011](https://proceedings.mlr.press/v15/ross11a.html)。本实验是代价敏感的 DAgger 式实现，不直接援引其理论保证。', '']
    (out / '实验报告.md').write_text('\n'.join(text))


def plot(out, summary, rounds):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [f'aggregate_r{rounds}_feedback_vs_aggregate_r{rounds}_progress',
             f'aggregate_r{rounds}_feedback_vs_bc_feedback',
             f'aggregate_r{rounds}_feedback_vs_matched_r{rounds}_feedback']
    labels = ['Feedback vs progress (aggregated)', 'Aggregation vs original BC (feedback)',
              'Aggregation vs matched queries (feedback)']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, field, title in zip(axes, ['cost_difference', 'joint_forgetting_difference'],
                               ['Endpoint cost difference (all histories)', 'Forgetting difference (joint success)']):
        for index, name in enumerate(names):
            value = summary['online']['paired'][name][field]
            mean, (lo, hi) = value['mean'], value['ci95']
            if mean is not None:
                ax.errorbar(mean, index, xerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt='o', color='#2563a6', capsize=4)
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [''] * len(labels))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('First minus second; lower is better')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.savefig(out / 'results.png', dpi=180)
    fig.savefig(out / 'results.pdf')
    plt.close(fig)


def analyze(run, out):
    out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((run / 'locked_protocol.json').read_text())
    groups = {}
    for name in lock['registry']:
        groups.setdefault(name.rsplit('_', 1)[0], []).append(name)
    online_summary, rows = online(run, groups, lock)
    summary = dict(online=online_summary,
        offline={domain: offline(run, domain, groups, lock) for domain in ['behavior', 'visited']},
        continuation=continuation_analysis(run), audit=audit(run, lock, rows),
        total_cost=json.loads((run / 'total_cost.json').read_text()))
    b.save(out / 'summary.json', summary)
    b.save(out / 'audit.json', summary['audit'])
    flat = [{k: v for k, v in r.items() if k != 'events'} for r in rows]
    with (out / 'online_endpoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write_report(out, summary, lock)
    plot(out, summary, lock['rounds'])
    key = f"aggregate_r{lock['rounds']}_feedback_vs_aggregate_r{lock['rounds']}_progress"
    print(json.dumps(dict(primary=summary['online']['paired'][key],
        continuation=summary['continuation']['paired']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)

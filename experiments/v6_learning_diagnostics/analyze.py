"""Audit locked learning diagnostics and summarize paired held-out results."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import branching as b
import learning as l


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values): return dict(mean=None, ci95=[None, None], roots=0)
    rng = np.random.default_rng(6801)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(), roots=len(values))


def grouped(values, mask=None):
    # First dimension is independent root, second is matched policy seed.
    return [float(row.mean()) for row in values] if mask is None else [float(row[m].mean()) for row, m in zip(values, mask) if m.any()]


def compare(a, c):
    ah, ch = np.broadcast_arrays(a['success'], c['success'])
    af, cf = np.broadcast_arrays(a['forgetting'], c['forgetting'])
    aj, cj = np.broadcast_arrays(a['cost'], c['cost'])
    aw, cw = np.broadcast_arrays(a['work'], c['work'])
    common = (ah > 0) & (ch > 0)
    al, cl = np.broadcast_arrays(a['new_monitor_loss'], c['new_monitor_loss'])
    target = np.broadcast_to(a['target'], common.shape)
    tight = common & (al >= .95 * target) & (cl >= .95 * target)
    return dict(reach_difference=interval(grouped(ah - ch)),
        forgetting_difference=interval(grouped(af - cf, common)),
        tight_forgetting_difference=interval(grouped(af - cf, tight)),
        J_difference=interval(grouped(aj - cj)), work_difference=interval(grouped(aw - cw)),
        joint_success_pairs=int(common.sum()))


def bundle(rows, names, roots):
    lookup = {name: {r['root']: r for r in rows[name]} for name in names}
    return {field: np.array([[lookup[name][root][field] for name in names] for root in roots], dtype=float)
            for field in ['success', 'forgetting', 'cost', 'work', 'steps', 'new_monitor_loss', 'target']}


def metrics(data):
    hit = data['success'] > 0
    return dict(reach=interval(grouped(data['success'])),
        successful_forgetting=interval(grouped(data['forgetting'], hit)),
        J=interval(grouped(data['cost'])), steps=float(data['steps'].mean()), work=float(data['work'].mean()))


def audit(run, lock):
    source = Path(__file__).parent
    source_ok = all(b.file_hash(source / name) == expected for name, expected in lock['source_hashes'].items())
    source_at_start = lock['source_hashes'] == lock['source_hashes_at_start']
    policies_ok = all(b.file_hash(run / name) == expected for name, expected in lock['policy_hashes'].items())
    splits = lock['splits']; roots = [root for values in splits.values() for root in values]
    splits_ok = len(roots) == len(set(roots))
    data_ok = True; counts = {}
    for split in ['train', 'dev', 'test']:
        rows = b.load_states(run / 'data' / f'{split}_branches.jsonl')
        counts[split] = dict(states=len(rows), effective_roots=len({s['root'] for s in rows}), planned_roots=len(splits[split]))
        data_ok &= all(s['root'] in splits[split] and len(s['x']) == 14 and len(s['branches']) == 5
                       and all(len(a) == lock['replicates'] for a in s['branches']) for s in rows)
    online = json.loads((run / 'test_online.json').read_text())
    online_ok = True
    for rows in online.values():
        online_ok &= sorted(r['root'] for r in rows) == splits['test']
        online_ok &= all(r['work'] <= r['budget'] + 1e-6 and r['success'] == (r['new_monitor_loss'] <= r['target']) for r in rows)
    selection_ok = True
    for folder in (run / 'models').iterdir():
        candidates = json.loads((folder / 'candidates.json').read_text())
        selected = json.loads((folder / 'selection.json').read_text())
        for selector, value in selected.items():
            field = 'closed' if selector == 'closed_F' else 'branch'
            chosen = min(candidates, key=lambda x: l.selection_key(x[field], selector))
            selection_ok &= chosen['epoch'] == value['epoch'] and chosen['slack'] == value['slack'] and chosen[field] == value['dev']
    checks = dict(source_hashes_match=source_ok, source_unchanged_during_run=source_at_start,
        policy_hashes_match=policies_ok, splits_disjoint=splits_ok, branch_labels_valid=bool(data_ok),
        online_budgets_endpoints_and_roots_valid=bool(online_ok), all_selections_reproduce=bool(selection_ok))
    assert all(checks.values()), checks
    return dict(checks=checks, counts=counts)


def summarize(run, lock):
    rows = json.loads((run / 'test_online.json').read_text())
    groups = {}
    for name in lock['registry']:
        groups.setdefault(name.rsplit('_', 1)[0], []).append(name)
    for name in rows:
        if name not in lock['registry']: groups[name] = [name]
    bundles = {name: bundle(rows, names, lock['splits']['test']) for name, names in groups.items()}
    pairs = [('raw_mse_closed_F_feedback', 'raw_mse_closed_F_progress'),
             ('raw_mse_closed_F_feedback', 'legacy_branch_J_feedback')]
    for kind in l.LOSSES:
        for mode in ['progress', 'feedback']:
            pairs += [(f'{kind}_branch_F_{mode}', f'{kind}_branch_J_{mode}'),
                      (f'{kind}_closed_F_{mode}', f'{kind}_branch_F_{mode}')]
    for mode in ['progress', 'feedback']:
        for selector in l.SELECTORS:
            pairs += [(f'raw_mse_{selector}_{mode}', f'legacy_{selector}_{mode}')]
    for baseline in ['best_fixed', 'best_time', 'best_progress', 'best_hand', 'fixed_zero']:
        pairs += [('raw_mse_closed_F_feedback', baseline)]
    paired = {f'{a}_vs_{c}': compare(bundles[a], bundles[c]) for a, c in pairs}
    positive = json.loads((run / 'positive_test.json').read_text())
    tiny = {p.stem: json.loads(p.read_text()) for p in (run / 'tiny').glob('*.json')}
    costs = [json.loads(p.read_text()) for p in (run / 'models').glob('*/cost.json')]
    return dict(methods={name: metrics(data) for name, data in bundles.items()}, paired=paired,
        positive=positive, tiny=tiny,
        selection_cost=dict(rollouts=sum(c['closed_dev_rollouts'] for c in costs), work=sum(c['closed_dev_work'] for c in costs)),
        per_replica={name: metrics(bundle(rows, [name], lock['splits']['test'])) for name in lock['registry']})


def fmt(value):
    if value['mean'] is None: return '无共同达标历史'
    return f"{value['mean']:.6f} [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}]"


def report(out, run, lock, summary, audited):
    primary = summary['paired']['raw_mse_closed_F_feedback_vs_raw_mse_closed_F_progress']
    positive = summary['positive']
    passed = sum(v['passed'] is True for v in positive.values())
    total = sum(v['feedback'] for v in positive.values())
    lines = ['# v6：控制器学习流程诊断', '',
        '本轮冻结 v4 的 MoE 环境、状态、动作与固定后续策略，检验学习损失和模型筛选是否限制控制表现。它不是新的 RL 算法，也没有加入动态路由。', '',
        '## 协议与可归因范围', '',
        f"- 新训练/开发/测试根历史 {len(lock['splits']['train'])}/{len(lock['splits']['dev'])}/{len(lock['splits']['test'])}，目标 0.08、最多 300 步和原工作量预算；每分支 {lock['replicates']} 个配对未来流。",
        '- legacy 复用 BCE + log1p 遗忘 Huber + 居中动作差异 Huber；raw_mse 改为原始遗忘除固定尺度后的 MSE 与居中 MSE，保留可行性头。后者对齐的是遗忘回归目标，仍不是精确求解整个约束控制问题。',
        '- 每个训练过程共享给三种筛选：branch_J、branch_F、closed_F。前两者按开发分支标签，后者按开发历史真实闭环终点。均先达标率，再代价/原始遗忘，最后工作量。',
        f"- {len(lock['seeds'])} 个匹配初始化；检查点 {lock['checkpoints']} × 四个 slack；所有选择在新测试历史生成前锁定。没有选择测试表现最佳的种子或检查点。",
        '- 新数据量与分支复测次数和旧版本不同；只能把 v6 内匹配消融归因于损失或筛选，不能把跨版本变化全部归因于某一改进。',
        '- 所有区间按根历史配对 bootstrap，先在历史内平均匹配策略种子；以下为未校正多重比较的名义 95% 区间，次要消融属于探索性诊断。', '',
        '## 1. 小样本训练集拟合', '',
        '按预先规则选不同根历史、非空保护组、全部动作可行且动作遗忘差异大于 1e-5 的最多 32 个训练状态。保持架构、优化器与正则，延长训练。以下是同一训练标签上的误差，不代表泛化能力。', '',
        '| 损失 / seed | 状态数 | 初始 regret | 中间 regret | 最终 regret | 最终动作命中（允许 1e-6 原始遗忘误差） |',
        '|---|---:|---:|---:|---:|---:|']
    for name, value in summary['tiny'].items():
        rec = value['records']; first, mid, last = [r['metrics'] for r in [rec[0], rec[1], rec[-1]]]
        lines.append(f"| {name} | {len(value['states'])} | {first['all_feasible_raw_regret']:.7f} | {mid['all_feasible_raw_regret']:.7f} | {last['all_feasible_raw_regret']:.7f} | {100*last['informative_action_accuracy']:.1f}% |")
    lines += ['', f"中间/最终训练步数分别为 {lock['tiny_checkpoints'][1]}/{lock['tiny_checkpoints'][-1]}；regret 是所选动作相对同标签最小遗忘的差值。", '',
        '## 2. 明确可学习的辅助反馈正对照', '',
        '这是单独的监督映射任务，不修改 MoE：相同进度上下文配五种均衡反馈，反馈明确编码最优倍率。Progress 理论上只能达到 20% 精确动作命中；新上下文用来测试映射泛化。预设 Feedback 通过要求是≥90% 命中且 regret≤固定中档基线的 10%。', '',
        f"Feedback 通过：**{passed}/{total}**。正对照成功不能证明真实 MoE 的两个反馈量足够，也不检验 PPO 的长时信用分配。", '',
        '| 损失 / 输入 / seed | 新上下文动作准确率 | regret / 固定中档基线 |', '|---|---:|---:|']
    for name, value in positive.items():
        lines.append(f"| {name} | {100*value['accuracy']:.2f}% | {value['relative_regret']:.4f} |")
    lines += ['', '## 3. 新历史真实闭环评价', '',
        '| 方法 | 达标率 | 成功子集遗忘 MSE | 平均步数 |', '|---|---:|---:|---:|']
    for name, value in summary['methods'].items():
        f = value['successful_forgetting']['mean']
        lines.append(f"| {name} | {100*value['reach']['mean']:.2f}% | {f if f is not None else 'NA'} | {value['steps']:.2f} |")
    lines += ['', '不同方法成功子集可能不同，以下看配对结果。差值均为前者减后者：', '',
        '| 比较 | 达标率差（95% CI） | 共同成功遗忘差（95% CI） |', '|---|---|---|']
    names = ['raw_mse_closed_F_feedback_vs_raw_mse_closed_F_progress',
             'raw_mse_closed_F_feedback_vs_legacy_branch_J_feedback',
             'raw_mse_closed_F_feedback_vs_best_fixed',
             'raw_mse_closed_F_feedback_vs_best_progress',
             'raw_mse_closed_F_feedback_vs_best_hand',
             'legacy_branch_F_feedback_vs_legacy_branch_J_feedback',
             'legacy_closed_F_feedback_vs_legacy_branch_F_feedback',
             'raw_mse_branch_F_feedback_vs_legacy_branch_F_feedback',
             'raw_mse_closed_F_feedback_vs_raw_mse_branch_F_feedback']
    for name in names:
        v = summary['paired'][name]
        lines.append(f"| {name} | {fmt(v['reach_difference'])} | {fmt(v['forgetting_difference'])} |")
    lines += ['', '- 主比较的紧端点匹配遗忘差：' + fmt(primary['tight_forgetting_difference']), '',
        '## 4. 审计、成本与边界', '',
        f"- 训练/开发/测试有效状态与根历史：`{json.dumps(audited['counts'])}`。初始已达标历史保留在闭环测试，不凭空制造训练状态。",
        f"- 7 项锁定/划分/预算/选择复算审计全部通过；全流程约 {json.loads((run/'total_cost.json').read_text())['seconds']:.1f} 秒。",
        f"- 闭环开发筛选额外评估 {summary['selection_cost']['rollouts']} 条候选轨迹，work={summary['selection_cost']['work']:.1f}；此开销不是控制器在线预算的一部分。",
        '- 模型权重、实际训练损失、训练集动作 regret、全部开发候选、选择记录和新测试轨迹全部保留。分支工作量另记在 data/*_cost.json。',
        '- 原始/当前路由改进继续留待真实 MoE；当前固定后续策略教师和观测的部分可见性仍未改变。',
        '- 小样本过拟合、辅助正对照、真实 MoE 闭环是三种不同证据，不能互相替代。若辅助任务能学而 MoE 无收益，仍不能排除标签噪声、观测不足、教师不匹配或更强学习方法的可能性。',
        '- 新测试历史已查看；后续方法变更必须使用新的最终测试历史。', '']
    (out / '实验报告.md').write_text('\n'.join(lines))


def plot(out, summary):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels, values = [], []
    for kind in l.LOSSES:
        for mode in ['progress', 'feedback']:
            labels.append(f'{kind}\n{mode}')
            values.append(np.mean([v['accuracy'] for k, v in summary['positive'].items() if k.startswith(f'{kind}_{mode}_')]))
    axes[0].bar(labels, values, color=['#9ba9b3', '#26799d']*2)
    axes[0].axhline(.2, color='grey', linestyle='--'); axes[0].set_ylim(0, 1.07)
    axes[0].set_ylabel('Held-out action accuracy'); axes[0].set_title('Auxiliary positive control (not MoE)')
    names = [('Raw + closed feedback vs progress', 'raw_mse_closed_F_feedback_vs_raw_mse_closed_F_progress'),
             ('Raw + closed vs legacy + branch (feedback)', 'raw_mse_closed_F_feedback_vs_legacy_branch_J_feedback'),
             ('Raw + closed feedback vs fixed', 'raw_mse_closed_F_feedback_vs_best_fixed'),
             ('Raw + closed feedback vs progress rule', 'raw_mse_closed_F_feedback_vs_best_progress')]
    for i, (_, key) in enumerate(names):
        v = summary['paired'][key]['forgetting_difference']; m = v['mean']; lo, hi = v['ci95']
        if m is not None: axes[1].errorbar(m, i, xerr=[[max(0,m-lo)], [max(0,hi-m)]], fmt='o', capsize=3, color='#26799d')
    axes[1].set_yticks(range(len(names)), [x[0] for x in names]); axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', linestyle='--'); axes[1].set_title('MoE: paired forgetting difference')
    axes[1].set_xlabel('First minus second; nominal 95% root bootstrap CI')
    fig.savefig(out / 'results.png', dpi=180); fig.savefig(out / 'results.pdf'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.run / 'locked_protocol.json').read_text())
    audited = audit(args.run, lock); summary = summarize(args.run, lock)
    b.save(args.out / 'audit.json', audited); b.save(args.out / 'summary.json', summary)
    report(args.out, args.run, lock, summary, audited); plot(args.out, summary)
    print(json.dumps({k: v for k, v in summary['paired'].items() if k in [
        'raw_mse_closed_F_feedback_vs_raw_mse_closed_F_progress',
        'raw_mse_closed_F_feedback_vs_legacy_branch_J_feedback']}, indent=2))


if __name__ == '__main__': main()

"""Paired sequence analysis; all outcomes including reach failures retained."""
import argparse
import json
from pathlib import Path

import numpy as np

import control as l


def interval(values, seed=7821):
    values = np.asarray(values, float)
    if not len(values): return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(n=len(values), mean=float(values.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist())


def group_arrays(names, rows):
    roots = [r['root'] for r in rows[names[0]]]
    assert all([r['root'] for r in rows[n]] == roots for n in names)
    field = lambda key: np.array([[r[key] for r in rows[n]] for n in names])
    return dict(roots=np.array(roots), hit=field('success').astype(float),
        forgetting=field('forgetting'), steps=field('steps'), work=field('work'),
        final_B=field('final_B_test'), final_B_monitor=field('final_B_monitor'),
        final_all_B=field('final_all_B_monitor_hit').astype(float),
        terminal_B=np.array([[[s['B_loss'] for s in r['segments']] for r in rows[n]] for n in names]),
        segments=field('segment_hits').astype(float), initially_solved=field('initially_solved'))


def compare(a, b):
    assert np.array_equal(a['roots'], b['roots'])
    # Conservative common set: all replicas of both methods succeeded on a root.
    common = (a['hit']==1).all(0)&(b['hit']==1).all(0)
    delta = a['forgetting'].mean(0)-b['forgetting'].mean(0)
    gap = np.max(np.abs(a['terminal_B'].mean(0)-b['terminal_B'].mean(0)), axis=1)
    matched = common & (gap <= .008)
    return dict(reach_difference=interval(a['hit'].mean(0)-b['hit'].mean(0)),
        common_success_forgetting=interval(delta[common]), all_root_forgetting=interval(delta),
        tight_B_endpoint_sensitivity=interval(delta[matched]),
        steps_difference=interval(a['steps'].mean(0)-b['steps'].mean(0)),
        final_all_B_retention_difference=interval(a['final_all_B'].mean(0)-b['final_all_B'].mean(0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run/'test_online.json').read_text())
    protocol = json.loads((args.run/'locked_protocol.json').read_text())
    groups = {}
    for frequency in ['block', 'within']:
        for mode in ['progress', 'feedback']:
            name = f'{frequency}_{mode}'
            groups[name] = [f'{name}_{seed}' for seed in protocol['seeds']]
    groups.update({n: [n] for n in rows if n not in protocol['registry']})
    arrays = {name: group_arrays(names, rows) for name, names in groups.items()}
    summary = {}
    for name, a in arrays.items():
        hit = a['hit'].astype(bool)
        summary[name] = dict(roots=len(a['roots']), replicas=len(a['hit']), reach=float(a['hit'].mean()),
            per_replica_hits=a['hit'].sum(1).astype(int).tolist(),
            forgetting_successful=float(a['forgetting'][hit].mean()) if hit.any() else None,
            forgetting_all=float(a['forgetting'].mean()), steps=float(a['steps'].mean()),
            work=float(a['work'].mean()), segment_reach=a['segments'].mean((0, 1)).tolist(),
            final_B_test_MSE=a['final_B'].mean((0, 1)).tolist(),
            final_all_B_monitor_reach=float(a['final_all_B'].mean()),
            initially_solved_segments=float(a['initially_solved'].mean()),
            mean_terminal_B_MSE=a['terminal_B'].mean((0, 1)).tolist())
    pairs = [('block_feedback', 'block_progress'), ('within_feedback', 'within_progress'),
             ('within_feedback', 'block_feedback'), ('within_progress', 'block_progress'),
             ('block_feedback', 'block_best_schedule'), ('within_feedback', 'within_best_schedule')]
    comparisons = {f'{a}_minus_{b}': compare(arrays[a], arrays[b]) for a, b in pairs}
    # Descriptive interaction on roots successful in ALL four learned arms/replicas.
    common = np.logical_and.reduce([(arrays[n]['hit']==1).all(0) for n in list(groups)[:4]])
    diff = ((arrays['within_feedback']['forgetting']-arrays['within_progress']['forgetting']).mean(0)-
            (arrays['block_feedback']['forgetting']-arrays['block_progress']['forgetting']).mean(0))
    selections = {n: json.loads((args.run/Path(p).parent/'selection.json').read_text())
                  for n, p in protocol['registry'].items()}
    cost_files = list((args.run/'models').glob('*/cost.json'))
    costs = [json.loads(p.read_text()) for p in cost_files]
    result = dict(groups=summary, comparisons=comparisons,
        frequency_by_feedback_interaction=interval(diff[common]), selections=selections,
        offline_selection=dict(rollouts=sum(c['dev_rollouts'] for c in costs),
            updates=sum(c['updates'] for c in costs), work=sum(c['work'] for c in costs)),
        statistics='Nominal unadjusted 95% root-paired bootstrap; replicas averaged inside root. Common-success set requires all replicas of both methods. Tight endpoint maximum per-block B-loss gap <=.008. Reach failures always reported.')
    l.save(args.out/'summary.json', result)
    lines = ['# 连续多段学习实验', '', '原始 A → B1 → B2 → B3；三段保留参数与 Adam，原始 A 记忆和遗忘基准不变。',
        '', '**达标率表示三个 B 在各自段末均达标，不表示序列结束仍全部保留。**', '',
        '| 方法 | 每个模型达标历史数 | 成功历史 A 遗忘 | 平均更新步数 | 最终全部 B 仍达标 |',
        '|---|---|---:|---:|---:|']
    for name, s in summary.items():
        f = 'NA' if s['forgetting_successful'] is None else f"{s['forgetting_successful']:.6f}"
        lines.append(f"| {name} | {s['per_replica_hits']} / {s['roots']} | {f} | {s['steps']:.1f} | {s['final_all_B_monitor_reach']:.1%} |")
    lines += ['', '不同成功集合的遗忘均值不能直接作为收益；主要结论使用下面共同成功历史的配对差。', '',
              '| 比较（前者−后者） | 共同成功根历史 | A 遗忘差 | 名义 95% CI |', '|---|---:|---:|---|']
    for name, c in comparisons.items():
        d = c['common_success_forgetting']
        lines.append(f"| {name} | {d['n']} | {d['mean']} | {d['ci95']} |")
    lines += ['', '## 解释边界', '',
        '- Progress 已使用新任务反馈。所消融的是实时旧知识反馈，不是所有反馈。',
        '- 固定几何路由、统一非线性教师；三段分别为新的输入分布。未要求低倍率一定失败。',
        '- 原始 A 的回放和监测各最多 64 个样本，前面的 B 不加入记忆；最终 B 保留情况另行报告。',
        '- 每段最多 100 步，未用完的步数不转移；不同方法共享预算上限而非强制同样实际步数。',
        '- 两组信息对照共用训练分支、网络、初始化、候选数和开发闭环选择标准。',
        '- 标签来自固定后续 m=0 的反事实分支，不代表全局最优控制；段间/段内频率改变了动作持续时间。',
        '- 额外离线分支查询和开发筛选成本单列。没有同轮 K=1 对照，不能把跨版本变化单独归因于增加段数。',
        '- 所有模型与基线在最终测试历史生成前锁定；置信区间未做多重比较校正。', '',
        f"开发模型筛选：{result['offline_selection']}", '']
    (args.out/'实验报告.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    colors = ['#476aaf', '#d48135']
    for j, frequency in enumerate(['block', 'within']):
        a, b = arrays[f'{frequency}_feedback'], arrays[f'{frequency}_progress']
        common = (a['hit']==1).all(0)&(b['hit']==1).all(0)
        for mode, color in zip(['progress', 'feedback'], colors):
            key = f'{frequency}_{mode}'; ar = arrays[key]
            axes[0].bar(j*3+(mode=='feedback'), ar['forgetting'][:, common].mean() if common.any() else 0, color=color,
                        label=mode if j==0 else None)
        axes[1].scatter(b['forgetting'].mean(0)[common], a['forgetting'].mean(0)[common], s=14, alpha=.65, label=frequency)
    axes[0].set_xticks([.5, 3.5]); axes[0].set_xticklabels(['Block', 'Within'])
    axes[0].set_ylabel('A forgetting (common successful roots)'); axes[0].legend()
    low, high = axes[1].get_xlim(); axes[1].plot([0, high], [0, high], color='gray', lw=1)
    axes[1].set_xlabel('Progress A forgetting'); axes[1].set_ylabel('Feedback A forgetting'); axes[1].legend()
    for mode, color in zip(['progress', 'feedback'], colors):
        values = [summary[f'{freq}_{mode}']['reach'] for freq in ['block', 'within']]
        axes[2].plot(['Block', 'Within'], values, marker='o', color=color, label=mode)
    axes[2].set_ylim(0, 1.05); axes[2].set_ylabel('All three segments reached target'); axes[2].legend()
    fig.savefig(args.out/'sequence_results.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(primary={k: v for k, v in comparisons.items() if k in
        ['block_feedback_minus_block_progress', 'within_feedback_minus_within_progress']}), indent=2))


if __name__ == '__main__':
    main()

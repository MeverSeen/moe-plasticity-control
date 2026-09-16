"""Report calibration and frozen confirmation without changing any selection."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import calibrate as c


def fmt(value, scale=1):
    if value['mean'] is None:
        return '无可比较历史'
    lo, hi = value['ci95']
    return f"{value['mean']*scale:.6f} [{lo*scale:.6f}, {hi*scale:.6f}]"


def show_name(plan):
    if plan['kind'] == 'fixed':
        return f"固定 m={c.b.ACTIONS[plan['action']]}"
    return f"{plan['kind']} {c.b.ACTIONS[plan['early']]}→{c.b.ACTIONS[plan['late']]}，切换点 {plan['threshold']}"


def audit(run, lock):
    source = Path(__file__).parent
    source_match = all(c.b.file_hash(source / name) == value for name, value in lock['source_hashes'].items())
    splits = lock['splits']
    disjoint = not set(splits['calibration']).intersection(splits['confirmation'])
    budgets, thresholds, pairing, labels = True, True, True, True
    counts = {}
    for path in sorted(run.glob('*.jsonl')):
        rows = c.b.load_states(path)
        split = path.name.split('_', 1)[0]
        labels &= all(r['root'] in splits[split] for r in rows)
        budgets &= all(r['work'] <= r['budget'] + 1e-6 for r in rows)
        thresholds &= all(r['success'] == (r['new_monitor_loss'] <= r['target']) for r in rows)
        keys = {(r['root'], r['plan'], r['future']) for r in rows}
        pairing &= len(keys) == len(rows) == len(splits[split]) * len(lock['candidates']) * 2
        counts[path.name] = len(rows)
    calibration = json.loads((run / 'calibration_summary.json').read_text())
    chosen, scores = c.select_setting(calibration, lock['settings'])
    selection_matches = chosen == lock['selected_setting']['name'] and scores == lock['selection_scores']
    baseline_matches = lock['selected_baselines'] == calibration[chosen]['selected']
    result = dict(source_hashes_match=source_match, disjoint_roots=disjoint,
        online_budgets_valid=bool(budgets), endpoint_thresholds_valid=bool(thresholds),
        complete_paired_plan_future_grid=bool(pairing), root_labels_valid=bool(labels),
        selection_reproduces=selection_matches, baseline_selection_reproduces=baseline_matches,
        rollout_counts=counts)
    assert all([source_match, disjoint, budgets, thresholds, pairing, labels, selection_matches, baseline_matches]), result
    return result


def plot(out, calibration, confirmation, settings):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    entries = [(s['name'], calibration[s['name']]['cross_future']['plan_oracle_vs_nonfeedback']['cost_difference']) for s in settings]
    for i, (label, v) in enumerate(entries):
        lo, hi = v['ci95']; mean = v['mean']
        axes[0].errorbar(mean, i, xerr=[[max(0,mean-lo)], [max(0,hi-mean)]], fmt='o', capsize=3, color='#1c658c')
    axes[0].set_yticks(range(len(entries)), [s[0] for s in entries])
    axes[0].set_title('Calibration: privileged plan vs global plan')
    entries2 = [('Selected nonfeedback vs fixed', confirmation['paired']['best_nonfeedback_vs_constant']),
                ('Cross-future plan oracle vs nonfeedback', confirmation['cross_future']['plan_oracle_vs_nonfeedback']),
                ('All constants feasible: oracle vs fixed', confirmation['cross_future']['all_feasible_constant_oracle_vs_constant'])]
    for i, (_, row) in enumerate(entries2):
        v = row['cost_difference']
        if v['mean'] is None:
            continue
        lo,hi=v['ci95']; mean=v['mean']
        axes[1].errorbar(mean, i, xerr=[[max(0,mean-lo)], [max(0,hi-mean)]], fmt='o', capsize=3, color='#1c658c')
    axes[1].set_yticks(range(len(entries2)), [x[0] for x in entries2])
    axes[1].set_title('Frozen independent confirmation')
    for ax in axes:
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Endpoint J difference; first minus second')
    fig.savefig(out / 'results.png', dpi=180)
    fig.savefig(out / 'results.pdf')
    plt.close(fig)


def report(out, lock, calibration, confirmation, audit_result, cost):
    chosen = lock['selected_setting']
    primary = confirmation['cross_future']['plan_oracle_vs_nonfeedback']
    f = primary['joint_forgetting_difference']
    positive = primary['reach_difference']['mean'] >= 0 and f['ci95'][1] is not None and f['ci95'][1] < 0
    verdict = ('独立确认中观察到特权历史相关计划选择的遗忘收益；这提供后续可学习性验证的理由，尚未验证动态反馈控制。'
               if positive else '独立确认尚未证明特权历史相关计划选择能在不降低达标率的情况下稳定减少遗忘；保留所有候选与负结果。')
    lines = ['# v5：不限定冲突形态的最小校准实验', '', '**' + verdict + '**', '',
        '## 为什么修正校准目标', '',
        '低倍率失败、高倍率遗忘大只是可能的机制，不是必要条件。本轮同时检验固定策略、时序安排和历史相关计划选择，并单独报告所有固定倍率均可行的情形。动作 argmin 不同本身不构成控制收益。', '',
        '## 预先指定的设置', '',
        f"- 新校准/确认根历史：{len(lock['splits']['calibration'])}/{len(lock['splits']['confirmation'])}；每历史两条独立未来 minibatch 流。同一未来流内所有计划共享批次，完整参数与 Adam 状态起点相同。",
        '- 复用固定 router、6 个非线性 experts、旧回放/监测各最多 64 样本、五档保护组位移倍率。保持原来的共同非线性教师，没有另造标签冲突。',
        f"- {len(lock['settings'])} 个目标/预算候选，正式扫描目标 0.08/0.04/0.02、最大步数 300/100，包含 v4 原始设置；各设置中所有方法共享相同在线工作量上限。",
        '- 有限计划共 17 个：5 个固定倍率，6 个按时间切换、6 个按 B 学习进度切换。切换方向是 0→1 或 1→0，最多切换一次；时间点 10%/30%/60%，进度点 25%/50%/75%。',
        '- B 进度由初始和当前 log(loss/target) 的相对改善定义，切换后不回退。当前旧知识反馈不进入这些可部署计划。',
        '- 校准未来流 0 选择全局固定、时间、进度及总候选最优计划：先达标数，再成功端点原始遗忘，再工作量。时间/进度候选各包含固定计划，避免强迫切换。',
        '- 配置候选只需有限计划逐历史可行覆盖≥80%、初始未达标比例≥50%；不要求同一固定计划对所有历史可行，不要求低倍率失败或遗忘单调增加。在合格配置中按跨未来的特权计划 J 改善选配置；若无合格者则确认原始参考配置并说明。',
        '- 配置和全局计划在创建任何确认历史前锁定。配置选择使用校准结果，不能把校准区间当作独立发现；独立确认不再调整任何选择。',
        '- 特权参考在每历史用未来流 0 的所有计划终点标签选计划，只在流 1 评分。其额外查询成本另计，不是可部署控制器或理论上界。',
        '- 先报告达标率和共同成功原始遗忘，J=2(1-hit)+hit·F/(F+0.01) 仅作包含失败的诊断/配置选择代理。相同阈值不是严格相同数值，保留紧端点匹配诊断。', '',
        '## 完整校准扫描', '',
        '| 设置 | 初始已达标 | 最佳固定达标率，两流 | 五个固定动作均可行，流0 | m=0失败且m=1成功，流0 | 特权计划−全局计划 ΔJ，独立流1（95% CI） |',
        '|---|---:|---:|---:|---:|---|']
    for setting in lock['settings']:
        s=calibration[setting['name']]; d=s['diagnostics']
        lines.append(f"| {setting['name']} | {d['initially_done']} | {100*s['methods']['best_constant']['reach']['mean']:.2f}% | {d['all_constants_feasible_selection_stream']} | {d['low_fails_high_succeeds_selection_stream']} | {fmt(s['cross_future']['plan_oracle_vs_nonfeedback']['cost_difference'])} |")
    lines += ['', '低倍率失败和高倍率达标这列只表示可行性差异：不能把失败端点的遗忘与达标端点直接比较，宣称相同新任务表现下的遗忘取舍。', '',
        '## 锁定后独立确认', '',
        f"选择的配置：**目标 MSE≤{chosen['target']}，最多 {chosen['max_steps']} 步及对应工作量预算**。"]
    for name,index in lock['selected_baselines'].items():
        lines.append(f"- {name}：{show_name(lock['candidates'][index])}。")
    lines += ['', '| 方法 | 达标率，两流 | 成功子集遗忘 MSE | 平均步数 |', '|---|---:|---:|---:|']
    for name, s in list(confirmation['all_constant_methods'].items()) + list(confirmation['methods'].items()):
        fmean=s['successful_forgetting']['mean']
        lines.append(f"| {name} | {100*s['reach']['mean']:.2f}% | {fmean if fmean is not None else 'NA'} | {s['mean_steps']:.2f} |")
    lines += ['', '不同方法成功子集可能不同，下表使用配对结果；差值为前者减后者。', '',
        '| 比较 | 达标率差（95% CI） | 共同成功遗忘差（95% CI） | ΔJ（95% CI） |', '|---|---|---|---|']
    pairs = [('固定前锁定的总计划 vs 固定倍率', confirmation['paired']['best_nonfeedback_vs_constant']),
             ('特权固定倍率选择 vs 全局固定，独立流1', confirmation['cross_future']['constant_oracle_vs_constant']),
             ('特权计划选择 vs 全局总计划，独立流1', primary)]
    for name, r in pairs:
        lines.append(f"| {name} | {fmt(r['reach_difference'])} | {fmt(r['joint_forgetting_difference'])} | {fmt(r['cost_difference'])} |")
    lines += ['', '- 特权计划 vs 全局计划的紧端点匹配遗忘差：' + fmt(primary['tight_endpoint_forgetting_difference']),
        '- 排除初始已达标、保护组为空历史后的对应遗忘差：' + fmt(confirmation['cross_future']['active_plan_oracle_vs_nonfeedback']['joint_forgetting_difference']), '',
        '## 不要求低倍率失败的检验', '',
        f"确认历史中，选择流 0 上五个固定倍率均可行的历史有 {confirmation['diagnostics']['all_constants_feasible_selection_stream']} 个；其中初始未达标且保护组非空的有 {confirmation['diagnostics']['all_feasible_active_protected']} 个。子集只由选择流定义，评价流失败仍保留。"]
    for key in ['all_feasible_constant_oracle_vs_constant','all_feasible_active_constant_oracle_vs_constant','all_feasible_plan_oracle_vs_nonfeedback']:
        r=confirmation['cross_future'][key]
        lines.append(f"- {key}：达标率差 {fmt(r['reach_difference'])}；遗忘差 {fmt(r['joint_forgetting_difference'])}；ΔJ {fmt(r['cost_difference'])}。")
    lines += ['', '## 可支持的结论范围', '',
        '- 全局时间/进度计划相对固定倍率的收益，只支持调度价值，不自动支持旧知识反馈。',
        '- 特权每历史计划参考若有独立未来收益，只支持历史相关的计划选择空间；可能仅需初始状态，仍未证明必须实时反馈。',
        '- 只有可用反馈在同一强进度基线上带来可部署闭环收益，才支持反馈控制主张。本轮不训练新控制器或 RL。',
        '- 17 个有限计划不是全部动态策略；负结果不能证明所有控制均无效。',
        '- 所有固定倍率可行的子集仍需同时检查新任务达标率与遗忘，不能用样本内 oracle 优势代替独立验证。', '',
        '## 审计与复现', '',
        f"- 全流程 {cost['seconds']:.1f} 秒；小模型本机时间不代表真实 MoE 成本。各阶段全部分支更新次数和工作量记录在 *_cost.json。",
        '- 所有锁定源码哈希、划分、目标阈值、预算、17×2 候选网格及配置/基线选择复算检查通过。',
        '- 原始 JSONL 保留每个计划/未来流的成功、原始遗忘、实际新 loss、工作量和动作轨迹。区间按根历史 bootstrap，未来复测不作为独立历史。',
        '- 确认集已经查看；后续训练和测试应使用新历史，不能围绕本次确认结果调参。',
        '- 动态路由改进已另记真实 MoE 待办，当前最小系统保持原实现。', '']
    (out/'实验报告.md').write_text('\n'.join(lines))


def analyze(run, out):
    out.mkdir(parents=True, exist_ok=True)
    lock=json.loads((run/'locked_protocol.json').read_text())
    calibration=json.loads((run/'calibration_summary.json').read_text())
    confirmation=json.loads((run/'confirmation_summary.json').read_text())
    cost=json.loads((run/'total_cost.json').read_text())
    checked=audit(run,lock)
    c.b.save(out/'audit.json',checked)
    c.b.save(out/'summary.json',dict(calibration=calibration,confirmation=confirmation,
        selected_setting=lock['selected_setting'],selection_scores=lock['selection_scores'],cost=cost))
    rows=c.b.load_states(run/f"confirmation_{lock['selected_setting']['name']}.jsonl")
    flat=[{k:v for k,v in r.items() if k!='action_trace'} for r in rows]
    with (out/'confirmation_endpoints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report(out,lock,calibration,confirmation,checked,cost)
    plot(out,calibration,confirmation,lock['settings'])
    print(json.dumps(dict(selected=lock['selected_setting'],
        confirmation=confirmation['cross_future']['plan_oracle_vs_nonfeedback']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();analyze(args.run,args.out)

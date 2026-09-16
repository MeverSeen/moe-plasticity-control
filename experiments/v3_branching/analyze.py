"""Clustered, paired analysis. No policy selection uses these test results."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import branching as b


def interval(values,seed=606,draws=10000):
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(mean=None,ci95=[None,None],histories=0)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(draws,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),histories=len(values))


def clustered(values,roots):
    values=np.asarray(values)
    return np.array([values[roots==r].mean() for r in np.unique(roots)])


def conditional_clustered(values,mask,roots):
    result=[]
    for root in np.unique(roots):
        v,m=values[roots==root],mask[roots==root]
        if m.any():result.append(v[m].mean())
    return np.array(result)


def selected_data(selected,hit,forget,cost):
    # result has state x policy-replica x future-replica dimensions
    selected=np.atleast_2d(selected)
    idx=np.arange(hit.shape[0])
    return [np.stack([a[idx,sel] for sel in selected],axis=1) for a in [hit,forget,cost]]


def bundle_stats(data,roots):
    hit,forget,cost=data
    return dict(reach=interval(clustered(hit,roots)),cost=interval(clustered(cost,roots)),
        endpoint_forgetting=interval(clustered(forget,roots)),
        successful_forgetting=interval(conditional_clustered(forget,hit.astype(bool),roots)))


def compare(a,c,roots):
    ah,af,ac=a;ch,cf,cc=c
    ah,ch=np.broadcast_arrays(ah,ch)
    af,cf=np.broadcast_arrays(af,cf)
    ac,cc=np.broadcast_arrays(ac,cc)
    common=(ah>0)&(ch>0)
    wins=np.where(ah!=ch,np.sign(ah-ch),np.where(common,np.sign(cf-af),0))
    return dict(reach_difference=interval(clustered(ah-ch,roots)),
        cost_difference=interval(clustered(ac-cc,roots)),
        all_endpoint_forgetting_difference=interval(clustered(af-cf,roots)),
        forgetting_on_joint_success=interval(conditional_clustered(af-cf,common,roots)),
        joint_success_pairs=int(common.sum()),
        win_minus_loss=interval(clustered(wins,roots)))


def online_data(rows,names,seeds):
    data={(r['method'],r['root']):r for r in rows}
    return [np.array([[[float(data[name,seed][field])] for name in names] for seed in seeds])
        for field in ['success','forgetting','cost']]


def estimate_text(value,scale=1):
    if value['mean'] is None:return '无共同样本'
    lo,hi=value['ci95']
    return f"{value['mean']*scale:.6f}，95% CI [{lo*scale:.6f}, {hi*scale:.6f}]"


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(run,out):
    out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((run/'locked_protocol.json').read_text())
    states=b.load_states(run/'test_branches.jsonl')
    x,roots,hit,forget,cost=b.arrays(states)
    preds=json.loads((run/'test_predictions.json').read_text())
    assert preds['state_ids']==[s['state_id'] for s in states]
    n=len(states);idx=np.arange(n)
    selections={name:np.array(a) for name,a in preds['actions'].items()}
    groups={mode:[name for name in selections if name.startswith(mode)] for mode in ['progress','feedback']}
    bundles={name:selected_data(a,hit,forget,cost) for name,a in selections.items()}
    for mode,names in groups.items():
        bundles[mode]=selected_data([selections[name] for name in names],hit,forget,cost)
    bundles['static_intervention']=selected_data(np.full(n,lock['static_branch_action']),hit,forget,cost)
    bundles['full_intervention']=selected_data(np.full(n,4),hit,forget,cost)
    oracle=cost.mean(2).argmin(1)
    bundles['oracle_in_sample']=selected_data(oracle,hit,forget,cost)
    offline={name:bundle_stats(data,roots) for name,data in bundles.items()}
    paired={'feedback_vs_progress':compare(bundles['feedback'],bundles['progress'],roots),
        'feedback_vs_static':compare(bundles['feedback'],bundles['static_intervention'],roots),
        'oracle_vs_static':compare(bundles['oracle_in_sample'],bundles['static_intervention'],roots)}
    for seed in lock['policy_seeds']:
        paired[f'feedback_vs_progress_{seed}']=compare(bundles[f'feedback_{seed}'],bundles[f'progress_{seed}'],roots)
    if hit.shape[2]>=2:
        cross=cost[:,:,0].argmin(1)
        oracle_cross=selected_data(cross,hit[:,:,1:],forget[:,:,1:],cost[:,:,1:])
        static_cross=selected_data(np.full(n,lock['static_branch_action']),hit[:,:,1:],forget[:,:,1:],cost[:,:,1:])
        paired['cross_future_oracle_vs_static']=compare(oracle_cross,static_cross,roots)
        offline['oracle_cross_future']=bundle_stats(oracle_cross,roots)
    rows=json.loads((run/'test_online.json').read_text())
    seeds=lock['splits']['test']
    root_array=np.array(seeds)
    online_groups=dict(groups,best_static=['best_static'],hand_feedback=['hand_feedback'],full_update=['full_update'])
    online_bundles={name:online_data(rows,names,seeds) for name,names in online_groups.items()}
    online={name:bundle_stats(data,root_array) for name,data in online_bundles.items()}
    for name,names in online_groups.items():
        rr=[r for r in rows if r['method'] in names]
        online[name].update(mean_steps=float(np.mean([r['steps'] for r in rr])),
            mean_work=float(np.mean([r['work'] for r in rr])),
            mean_switches=float(np.mean([r['switches'] for r in rr])),
            mean_multiplier=float(np.mean([r['mean_multiplier'] for r in rr])))
    online_paired={f'feedback_vs_{name}':compare(online_bundles['feedback'],online_bundles[name],root_array)
        for name in ['progress','best_static','hand_feedback','full_update']}
    # Tight endpoint sensitivity: both selected models cross within 5% below target.
    keyed={(r['method'],r['root']):r for r in rows}
    tight=[]
    for seed in seeds:
        dd=[]
        for ps in lock['policy_seeds']:
            f,p=keyed[f'feedback_{ps}',seed],keyed[f'progress_{ps}',seed]
            if f['success'] and p['success'] and min(f['new_monitor_loss'],p['new_monitor_loss'])>=.95*lock['config']['target']:
                dd.append(f['forgetting']-p['forgetting'])
        if dd:tight.append(np.mean(dd))
    online_paired['feedback_vs_progress']['tight_target_forgetting']=interval(tight)
    active=np.array([keyed['full_update',seed]['initial_new_loss']>lock['config']['target'] for seed in seeds])
    online_paired['feedback_vs_progress']['active_histories_only']=compare(
        [v[active] for v in online_bundles['feedback']],
        [v[active] for v in online_bundles['progress']],root_array[active])
    costs={p.stem:json.loads(p.read_text()) for p in run.glob('*_cost.json')}
    mean_cost=cost.mean(2)
    diagnostics=dict(states=n,histories_with_states=len(np.unique(roots)),
        total_test_histories=len(seeds),initial_states=sum(s['step']==0 for s in states),
        no_protected_states=sum(s['protected_mass']<1e-8 for s in states),
        almost_all_protected_states=sum(s['protected_mass']>.99 for s in states),
        oracle_action_counts=np.bincount(oracle,minlength=5).tolist(),
        states_with_cost_spread_gt_001=int((np.ptp(mean_cost,axis=1)>.01).sum()),
        mean_cost_spread=float(np.ptp(mean_cost,axis=1).mean()),
        initially_finished_test_histories=sum(r['initially_done'] for r in costs['test_cost']['roots']))
    diagnostics['action_agreement']={str(seed):float(np.mean(
        selections[f'feedback_{seed}']==selections[f'progress_{seed}'])) for seed in lock['policy_seeds']}
    diagnostics['selected_action_counts']={name:np.bincount(a,minlength=5).tolist() for name,a in selections.items()}
    summary=dict(offline=offline,offline_paired=paired,online=online,online_paired=online_paired,
        diagnostics=diagnostics,costs=costs,
        note='Equal history weighting. Policy replicas averaged within each history; CI is conditional on fitted policies.')
    b.save(out/'summary.json',summary)
    flat=[]
    for state in states:
        for action,rr in enumerate(state['branches']):
            for r in rr:
                flat.append(dict(root=state['root'],state_id=state['state_id'],step=state['step'],
                    behavior=state['path'],multiplier=b.ACTIONS[action],
                    **{k:v for k,v in r.items() if k!='local'}))
    write_csv(out/'branch_endpoints.csv',flat)
    write_csv(out/'online_endpoints.csv',[{k:v for k,v in r.items() if k!='events'} for r in rows])
    plot(out,summary)
    report(out,run,lock,summary)
    print(json.dumps(dict(offline_primary=paired['feedback_vs_progress'],
        online_primary=online_paired['feedback_vs_progress'],diagnostics=diagnostics),indent=2))


def plot(out,s):
    os.environ['MPLCONFIGDIR']=str(out/'.matplotlib')
    cache=out.resolve()/'.cache'
    cache.mkdir(exist_ok=True)
    os.environ['XDG_CACHE_HOME']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,names,title in [
        (axes[0,0],'offline',['static_intervention','progress','feedback','oracle_in_sample'],'Single intervention: terminal cost'),
        (axes[1,0],'online',['best_static','hand_feedback','progress','feedback'],'Repeated control: terminal cost')]:
        labels={'static_intervention':'Static action','oracle_in_sample':'Oracle*',
            'best_static':'Best static','hand_feedback':'Hand feedback','progress':'Progress','feedback':'Feedback'}
        for i,name in enumerate(names):
            metric=s[key][name]['cost'];mean=metric['mean'];lo,hi=metric['ci95']
            color='#287d8e' if name=='feedback' else '#8999ad'
            ax.bar(i,mean,color=color,width=.65)
            ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='none',color='#23313f',capsize=3)
        ax.set_xticks(range(len(names)),[labels[n] for n in names])
        ax.set_title(title,loc='left');ax.set_ylabel('J (lower is better; every failure costs 2)')
    names=['best_static','hand_feedback','progress','feedback']
    vals=[s['online'][n]['reach']['mean']*100 for n in names]
    axes[1,1].bar(range(4),vals,color=['#8999ad']*3+['#287d8e'])
    axes[1,1].set_xticks(range(4),['Best static','Hand feedback','Progress','Feedback'])
    axes[1,1].set_ylim(0,105);axes[1,1].set_ylabel('Reached target (%)')
    axes[1,1].set_title('Repeated control: reachability',loc='left')
    for i,(key,label) in enumerate([('offline_paired','Single intervention'),('online_paired','Repeated control')]):
        metric=s[key]['feedback_vs_progress']['cost_difference']
        mean=metric['mean'];lo,hi=metric['ci95']
        axes[0,1].errorbar(mean,i,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color='#287d8e',capsize=5)
    axes[0,1].axvline(0,color='#7b8490',ls='--',lw=1)
    axes[0,1].set_yticks([0,1],['Single intervention','Repeated control'])
    axes[0,1].set_ylim(-.7,1.7);axes[0,1].set_xlabel('Feedback - progress cost; paired 95% CI')
    axes[0,1].set_title('Primary comparison (left favors feedback)',loc='left')
    fig.suptitle('MoE branching experiment | independent histories, fixed router',fontsize=15)
    fig.supxlabel('* Oracle uses realized outcomes and is an optimistic diagnostic, not a deployable policy.',fontsize=9)
    fig.savefig(out/'results.png',dpi=180)
    fig.savefig(out/'results.pdf')
    plt.close(fig)


def report(out,run,lock,s):
    op=s['offline_paired']['feedback_vs_progress'];cp=s['online_paired']['feedback_vs_progress']
    if op['cost_difference']['ci95'][0]<=0<=op['cost_difference']['ci95'][1] and cp['cost_difference']['ci95'][0]<=0<=cp['cost_difference']['ci95'][1]:
        conclusion='**主要结论：本轮未验证当前旧知识反馈优于充分提供进度信息的策略，一次干预和反复闭环两项主要比较的区间均跨零。不能把这次实验作为进入 RL 的正证据。**'
    else:
        conclusion='**主要结论请结合下面的一次干预与闭环配对结果，以及达标率共同判断。**'
    lines=['# MoE 标量反馈控制：分支实验报告','',
        conclusion,'',
        '本报告同时保留成功与失败，并区分一次干预的动作价值与反复闭环控制。',
        '本轮使用已有合成环境的新历史种子；不是对先前 4-expert 线性结果的复现，也不是真实 learned-router 验证。','',
        '## 已锁定的实验设置','',
        f"- 校准/训练/验证/测试历史：{len(lock['splits']['calibration'])}/{len(lock['splits']['train'])}/{len(lock['splits']['dev'])}/{len(lock['splits']['test'])}；同一历史所有分支不跨划分。",
        '- 6 个 10→12→1 tanh experts，top-2 固定几何路由；每个历史先学习 1–3 个旧任务。不同世界的教师、任务顺序、路由角度与初始化变化。',
        '- 固定初始保护 mask：历史路由频率 ≥ 1/3 且新任务频率 ≥ 0.05。固定 replay/monitor memory 各最多 64 个旧样本。',
        '- 动作 m∈{0,0.25,0.5,0.75,1}，H=5。新梯度 + 0.1 回放梯度形成 Adam proposal，再按 expert mask 缩放实际位移；本轮没有 weight decay。',
        '- 新任务目标为监测 MSE≤0.06，最多 300 个模型更新，并受包含监测/策略等额外工作的共同 work 上限约束。work 是前向样本等价量，不是精确 FLOPs。',
        f"- 后续策略只在校准历史上选择，本轮锁定 m={b.ACTIONS[lock['continuation_action']]}。每个动作执行最多 H 步，然后用此相同策略至达标/预算耗尽。",
        '- 预先指定两条状态生成路径（固定 0.5 与随机动作前缀），在步数 0/10/30/60 取未结束快照；不按是否出现动作排序反转筛选。',
        f"- 每状态每动作 {lock['future_replicates']} 个独立未来样本流；同一复测内所有动作共享未来 minibatch 索引。复制权重、Adam 一二阶矩、步数、历史观测、预算和数据流。",
        '- Progress 与 Feedback 使用相同的 14→32→32 网络、3 个匹配初始化、相同训练/调参预算。Progress 的最后两个旧知识反馈输入置零；它仍有初始路由摘要、初始能力、当前 B loss、ΔB、剩余预算、步数和上一动作。',
        '- 两个监督头预测达标概率与 log(1+遗忘/0.01)，加入动作间相对损失。推理依据预测的达标概率选候选，绝不依据该动作真实测试达标标签筛选。',
        '- Train/dev 历史的独立评估样本用于离线元训练标签；test 历史评估样本只在锁定策略后用于评分。所有在线观测/更新只读取限定训练与记忆数据。',
        '- 所有方法承担相同的监测与评分器计算额度：这是信息消融，不是部署净效率证明。离线分支代价与在线训练代价分开记录。','',
        '## 评价定义','',
        '遗忘 F 是每个旧任务独立评估 MSE 相对 B 开始时的增量取正部，再在旧任务间平均。另保存有符号变化与新任务独立评估损失。',
        '先报告达标率；为保留失败另报告诊断代价 J=2(1-hit)+hit·F/(F+0.01)。这个标量的改善不能替代达标率与共同成功遗忘比较。',
        '首次穿过统一阈值不等于 loss 数字精确相等：保留实际端点 loss，另作双方落在 [0.057,0.06] 内的敏感性分析。',
        '每个训练历史权重相同，先在历史内平均策略初始化和未来复测，再进行 10,000 次配对历史 bootstrap。区间不充分包含策略重新训练的不确定性；三个训练初始化只作稳健性检查。','',
        '## 第一关：一次干预','',
        f"测试集中 {s['diagnostics']['histories_with_states']} 个有可用状态的历史，共 {s['diagnostics']['states']} 个 checkpoint；{s['diagnostics']['initially_finished_test_histories']} 个历史在 B 开始即已满足目标，单独报告且不制造训练状态。",'',
        '| 策略 | 达标率 | 终点代价 J | 成功分支遗忘 MSE（各自子集） |',
        '|---|---:|---:|---:|']
    for name in ['static_intervention','full_intervention','progress','feedback','oracle_in_sample','oracle_cross_future']:
        if name not in s['offline']:continue
        v=s['offline'][name];ff=v['successful_forgetting']['mean']
        lines.append(f"| {name} | {100*v['reach']['mean']:.2f}% | {v['cost']['mean']:.6f} | {ff:.6f} |")
    lines += ['', '这里 static_intervention 只固定当前 H 步的动作；后续仍使用共同 continuation。它与下表的全程 Best Static 是不同对照。',
        'oracle_in_sample 读取所有分支终点，是有乐观偏差的诊断上界。oracle_cross_future 用第一条未来流选择动作，仅在第二条独立未来流评分，检查排序是否只是噪声；它也不是可部署策略。','',
        '**主要配对结果，差值为 Feedback − Progress：**','',
        f"- 达标率差（百分点）：{estimate_text(op['reach_difference'],100)}。",
        f"- 全部成功/失败保留的代价差：{estimate_text(op['cost_difference'])}。",
        f"- 共同成功上的遗忘差：{estimate_text(op['forgetting_on_joint_success'])}；该项是条件诊断，不能单独证明总体改善。",'',
        '## 第二关：反复闭环使用','',
        '监督评分器在完整 B 适应过程中每 H 步重新决策；Best Static 与 Hand Feedback 均在开发历史上充分比较候选，所有方法使用相同 learner、mask、状态测量与计算上限。未进行 RL 训练。','',
        '| 策略 | 达标率 | 终点代价 J | 成功历史遗忘 MSE（各自子集） | 平均步数（含失败） |',
        '|---|---:|---:|---:|---:|']
    for name in ['full_update','best_static','hand_feedback','progress','feedback']:
        v=s['online'][name]
        lines.append(f"| {name} | {100*v['reach']['mean']:.2f}% | {v['cost']['mean']:.6f} | {v['successful_forgetting']['mean']:.6f} | {v['mean_steps']:.2f} |")
    lines += ['', '**主要配对结果，差值为 Feedback − Progress：**','',
        f"- 达标率差（百分点）：{estimate_text(cp['reach_difference'],100)}。",
        f"- 全部历史代价差：{estimate_text(cp['cost_difference'])}。",
        f"- 共同成功遗忘差：{estimate_text(cp['forgetting_on_joint_success'])}。",
        f"- 更紧端点匹配后的遗忘差：{estimate_text(cp['tight_target_forgetting'])}。",'',
        f"- 排除初始即达标的历史后，遗忘差：{estimate_text(cp['active_histories_only']['forgetting_on_joint_success'])}。",'',
        '## 如何解释','']
    for title,r in [('一次干预',op),('反复闭环',cp)]:
        lo,hi=r['cost_difference']['ci95']
        if hi<0:
            judgement='诊断代价的区间支持反馈更低；仍须同时看达标率与遗忘。'
        elif lo>0:
            judgement='诊断代价的区间支持反馈更高，本轮反馈策略更差。'
        else:
            judgement='诊断代价差的区间跨零，本轮未验证反馈优于进度策略。'
        lines.append(f'- {title}：{judgement}')
    cross=s['offline_paired'].get('cross_future_oracle_vs_static')
    if cross:
        lines += ['',f"独立未来流复测的 oracle 相对固定动作代价差：{estimate_text(cross['cost_difference'])}。",
            '该诊断用于区分可重复的动作差异与事后挑选噪声；只有两条未来流，不能据此断言全部差异都来自噪声。']
    lines += ['', '这些结果只评价当前合成环境、固定分组、五个倍率、H=5 与选定后续策略。不能据此宣称 RL 必要、真实 MoE 有效或所有动态控制无效。',
        '即使离线动作评分有用，闭环后也可能遇到不同于训练快照的状态；本轮没有用最终测试结果回流训练或调参。',
        '“最优动作分布”包含并列和噪声，不能仅凭某动作出现最多判定策略必要性。重点看跨未来复测的可实现增益与未见历史上的决策效果。','',
        '## 检查与复现','',
        '- 9 项机制检查通过：完整状态克隆、共享/独立未来流、评估标签隔离、带动量零位移冻结、共同 continuation、窗口内首次达标停止、进度策略屏蔽旧反馈、无保护组阴性对照、根历史划分互斥。',
        f"- 正式运行端到端时间：{s['costs']['total_cost']['seconds']:.1f} 秒。原始分支更新成本见各 split_cost.json；其 branch_work 不含共同预训练和快照路径构造成本，端到端时间包含它们。",
        '- 锁定协议保存了策略与实验源码 SHA-256，且写入发生在生成任何正式 test 历史之前。',
        '- 原始结果：results/main/*_branches.jsonl、test_predictions.json、test_online.json；本目录提供 CSV、summary.json、PNG/PDF 图。',
        '- 运行：python3 branching.py --out results/new_run；分析：python3 analyze.py --run results/new_run --out results/new_run/analysis。',
        '- 这批最终测试历史已被查看；后续修改应改用新的测试历史。','']
    (out/'实验报告.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.run,a.out)

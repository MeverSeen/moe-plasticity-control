"""Root-cluster paired analysis; controller seeds are not independent histories."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import control as l

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path);a=ap.parse_args()
    out=a.out or a.run/'analysis';out.mkdir(parents=True,exist_ok=True)
    raw=json.loads((a.run/'test_online.json').read_text());groups={}
    for name,rows in raw.items():
        key=name.rsplit('_',1)[0] if name.rsplit('_',1)[-1].isdigit() else name
        groups.setdefault(key,[]).append(rows)
    summary={}
    for key,sets in groups.items():
        summary[key]=dict(reach=float(np.mean([[r['success'] for r in rows] for rows in sets])),
            segment_reach=np.mean([[r['segment_hits'] for r in rows] for rows in sets],axis=(0,1)).tolist(),
            final_all_B_monitor=float(np.mean([[r['final_all_B_monitor_hit'] for r in rows] for rows in sets])),
            final_all_B_test=float(np.mean([[max(r['final_B_test'])<=.08 for r in rows] for rows in sets])),
            all_root_forgetting=float(np.mean([[r['forgetting'] for r in rows] for rows in sets])),
            steps=float(np.mean([[r['steps'] for r in rows] for rows in sets])))
    pairs=[]
    for algo in ['sup','ppo']:
        for info in ['progress','feedback']:pairs.append((f'{algo}_a15_{info}',f'{algo}_a5_{info}'))
        for n in [5,15]:pairs.append((f'{algo}_a{n}_feedback',f'{algo}_a{n}_progress'))
    for n in [5,15]:
        for info in ['progress','feedback']:pairs.append((f'ppo_a{n}_{info}',f'sup_a{n}_{info}'))
    primary=len(pairs)
    for n in [5,15]:
        for info in ['progress','feedback']:
            pairs.append((f'ppo_a{n}_{info}',f'warm_a{n}_{info}'))
            pairs.append((f'last_a{n}_{info}',f'warm_a{n}_{info}'))
    for base in ['fixed','global_lr','schedule']:pairs.append(('ppo_a15_feedback',base))
    rng=np.random.default_rng(88001)
    def ci(x):
        if not len(x):return dict(mean=None,ci95=None,ci_corrected=None,lower95=None)
        draws=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
            ci_corrected=np.quantile(draws,[.025/primary,1-.025/primary]).tolist(),lower95=float(np.quantile(draws,.05)))
    comparisons=[]
    for i,(left,right) in enumerate(pairs):
        aa,bb=groups[left],groups[right]
        assert all([r['root'] for r in rows]==[r['root'] for r in aa[0]] for rows in aa+bb)
        reach=lambda sets:np.mean([[r['success'] for r in rows] for rows in sets],axis=0)
        forgetting=lambda sets:np.mean([[r['forgetting'] for r in rows] for rows in sets],axis=0)
        common=np.all([[r['success'] for r in rows] for rows in aa+bb],axis=0)
        delta=forgetting(aa)-forgetting(bb)
        endpoint_a=np.mean([[[s['B_loss'] for s in r['segments']] for r in rows] for rows in aa],axis=0)
        endpoint_b=np.mean([[[s['B_loss'] for s in r['segments']] for r in rows] for rows in bb],axis=0)
        sensitive=common & (np.max(np.abs(endpoint_a-endpoint_b),axis=1)<=.008)
        reach_ci=ci(reach(aa)-reach(bb))
        comparisons.append(dict(left=left,right=right,primary=i<primary,reach=reach_ci,
            noninferiority_5pp=reach_ci['lower95']>-.05,common_roots=int(common.sum()),
            left_forgetting=float(forgetting(aa)[common].mean()) if common.any() else None,
            right_forgetting=float(forgetting(bb)[common].mean()) if common.any() else None,
            forgetting=ci(delta[common]),sensitivity_roots=int(sensitive.sum()),sensitivity=ci(delta[sensitive])))
    l.save(out/'summary.json',dict(groups=summary,comparisons=comparisons))
    lines=['# v8 测试报告','', '达标是三个B各自预算内曾达标；最终B保留另列。下表为根历史内初始化平均。','', '| 方法 | 达标率 | 最终全部B monitor | 全根A遗忘（描述性） | 平均步数 |','|---|---:|---:|---:|---:|']
    for key,v in summary.items():lines.append(f"| {key} | {v['reach']:.2%} | {v['final_all_B_monitor']:.2%} | {v['all_root_forgetting']:.6f} | {v['steps']:.1f} |")
    lines+=['','## 配对比较','','差值均为左−右；遗忘仅双方所有初始化共同成功根历史。12项主要比较另给Bonferroni敏感性区间。非劣效按单侧95%下界大于−5个百分点判断，未经多重校正。','']
    for c in comparisons:
        lines.append(f"- {c['left']} − {c['right']}: 达标差 {c['reach']['mean']:.4f}，95% {c['reach']['ci95']}；共同根 {c['common_roots']}，遗忘差 {c['forgetting']['mean']}，95% {c['forgetting']['ci95']}，校正 {c['forgetting']['ci_corrected']}；非劣效 {c['noninferiority_5pp']}。")
    lines+=['','不能将低达标率下的低遗忘称为无条件收益。PPO近似训练不保证概率约束；被选中warm时不代表RL学习获益。固定A记忆未覆盖已学B，固定几何router与合成任务限制外推。有限基线候选不代表所有全局学习率或调度的最优。']
    (out/'实验报告.md').write_text('\n'.join(lines)+'\n')
    names=[k for k in summary if k.startswith(('sup_a','ppo_a'))]+['fixed','global_lr','schedule']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,field,label in zip(axs,['reach','final_all_B_monitor'],['All segments reached target','All B retained at endpoint (monitor)']):
        ax.barh(names,[summary[k][field] for k in names]);ax.set_xlim(0,1);ax.set_title(label);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(out/'results.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

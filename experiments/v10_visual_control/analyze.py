"""Descriptive paired-root analysis; pilot intervals are not confirmatory claims."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

def load(p): return json.loads(p.read_text())
def summary(rows):
    return {'sequence_success':float(np.mean([r['success'] for r in rows])),
      'task_success':float(np.mean([t['success'] for r in rows for t in r['tasks']])),
      **{f:float(np.mean([r[f] for r in rows])) for f in ['forgetting','signed_forgetting','final_loss','final_accuracy','wall_seconds']},
      'learner_updates':sum(r['cost']['updates'] for r in rows),
      'initial_successes':sum(t['initial_success'] for r in rows for t in r['tasks']),
      'failures':sum(not t['success'] for r in rows for t in r['tasks'])}

def analyze(out):
    test=load(out/'test.json');protocol=load(out/'protocol.json');variants={}
    for k,rows in test.items():
        name=k.rsplit('_',1)[0] if k.rsplit('_',1)[-1].isdigit() else k
        variants.setdefault(name,[]).append(rows)
    result={'methods':{name:summary([r for seed in seeds for r in seed]) for name,seeds in variants.items()},'paired':{},'scope':'Exploratory pilot; roots are sampling units, controller seeds averaged within root. Class identities overlap across roots within a split; inference conditional on heldout class pool.'}
    pairs=[('grpo_feedback','grpo_progress'),('grpo_feedback','ppo_feedback'),('grpo_feedback','grpo_local'),('grpo_feedback','grpo_factorized'),('grpo_feedback','grpo_groups'),('grpo_feedback','grpo_global'),('grpo_feedback','coordinated'),('grpo_feedback','best_fixed')]
    rng=np.random.default_rng(1040000)
    for a,b in pairs:
        if a not in variants or b not in variants:continue
        aa,bb=variants[a],variants[b]
        fa=np.mean([[r['forgetting'] for r in seed] for seed in aa],axis=0);fb=np.mean([[r['forgetting'] for r in seed] for seed in bb],axis=0)
        sa=np.array([[r['success'] for r in seed] for seed in aa]);sb=np.array([[r['success'] for r in seed] for seed in bb]);common=sa.all(0)&sb.all(0)
        d=fa-fb;dd=d[common]
        bootstrap=lambda x: np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist() if len(x) else None
        # Predefined endpoint matching sensitivity: all task-monitor endpoints differ <= .05 after seed averaging.
        la=np.mean([[[t['loss'] for t in r['tasks']] for r in seed] for seed in aa],axis=0);lb=np.mean([[[t['loss'] for t in r['tasks']] for r in seed] for seed in bb],axis=0)
        matched=common&(np.max(np.abs(la-lb),axis=1)<=.05)
        result['paired'][a+' - '+b]={'all_root_delta':float(d.mean()),'all_root_bootstrap95':bootstrap(d),'common_success_roots':int(common.sum()),'common_delta':float(dd.mean()) if len(dd) else None,'common_bootstrap95':bootstrap(dd),'success_difference':float(sa.mean()-sb.mean()),'loss_matched_roots':int(matched.sum()),'loss_matched_delta':float(d[matched].mean()) if matched.any() else None,'confirmatory_claim':False}
    training=[]
    for p in out.glob('train_*.json'):
        hist=load(p);training.append({'name':p.stem,'rounds':len(hist),'distinct_roots':len(set(x['root'] for x in hist)),'parameter_changes':[x.get('parameter_change') for x in hist],'reward_std':[x.get('reward_std') for x in hist],'learner_updates':sum(r['branch_cost']['updates'] for batch in hist for r in batch['results']),'wall_seconds':sum(x['wall_seconds'] for x in hist),'zero_std_groups':sum(x.get('reward_std',0)==0 for x in hist)})
    result['training']=training
    result['training_total_seconds']=sum(x['wall_seconds'] for x in training);result['training_total_updates']=sum(x['learner_updates'] for x in training)
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    lines=['# v10 独立闭环 pilot 结果','','类不相交的四任务 CIFAR-100 pilot；未作统计优越性确认。失败保留，模型只由开发闭环选择。','','| 方法 | 全序列达标 | 累计遗忘 | 最终平均loss | 最终accuracy |','|---|---:|---:|---:|---:|']
    for name,r in result['methods'].items():lines.append(f"| {name} | {r['sequence_success']:.1%} | {r['forgetting']:.5f} | {r['final_loss']:.4f} | {r['final_accuracy']:.1%} |")
    lines+=['',f"外层真实分支训练合计 {result['training_total_updates']} 次 learner 更新，{result['training_total_seconds']/60:.1f} 分钟（不含前缀、开发筛选和最终评价）。",'', '完整配对差、共同成功样本数、终点匹配敏感性与学习变化见 analysis.json。小样本区间仅作描述，不声称通过90%达标或5pp非劣效门槛。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

def archive(out,dest):
    dest.mkdir(parents=True,exist_ok=False)
    for p in sorted(out.glob('*')):
        if p.suffix not in ['.json','.md'] or p.name.endswith('.partial.json'):continue
        raw=p.read_bytes()
        if len(raw)>100000:
            with gzip.GzipFile(filename=str(dest/(p.name+'.gz')),mode='wb',mtime=0) as f:f.write(raw)
        else:(dest/p.name).write_bytes(raw)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--archive',type=Path);a=ap.parse_args();analyze(a.out)
    if a.archive:archive(a.out,a.archive)

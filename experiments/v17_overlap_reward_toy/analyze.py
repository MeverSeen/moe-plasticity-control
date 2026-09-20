"""Locked cohort analysis; never selects methods using test outcomes."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v16_analysis',HERE.parent/'v16_step_reward_toy'/'analyze.py')
base = importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
FIELDS = ['test_F','steps','decisions','path_energy','overlap_path_energy','other_path_energy',
          'net_displacement_sq','overlap_net_displacement_sq','other_net_displacement_sq']


def stats(rows):
    return {**base.stats(rows),**{k:base.mean(rows,k) for k in FIELDS},
            'empty_mask_histories':sum(r['protected_experts']==0 for r in rows)}


def proxy_ranking(data,field):
    roots = {r['root'] for r in data['fixed_0.0']}
    concordant = discordant = tied = reversals = 0
    for root in roots:
        rows = [r for name,rr in data.items() if name.startswith('fixed_')
                for r in rr if r['root']==root and r['success']]
        reversed_here = False
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                sign=(a[field]-b[field])*(a['test_F']-b['test_F'])
                if sign>1e-14:concordant+=1
                elif sign< -1e-14:discordant+=1;reversed_here=True
                else:tied+=1
        reversals+=reversed_here
    return dict(concordant=concordant,discordant=discordant,tied=tied,roots_with_reversal=reversals)


def main(directory):
    out=Path(directory)
    complete=json.loads((out/'completion.json').read_text())
    fits=complete['fits']; fixed=json.loads((out/'baseline_selection.json').read_text())['selected_action']
    fixed_name=f'fixed_{[0.,.25,.5,.75,1.][fixed]}'
    report={};tables=[]
    for cohort in ('confirmation','diagnostic'):
        with gzip.open(out/f'{cohort}_results.json.gz','rt') as file:data=json.load(file)
        groups={}
        for reward in ('forgetting','movement','overlap'):
            for kind in ('selected','last'):
                groups[f'{reward}_{kind}']=[row for fit in fits if fit['reward']==reward
                    for row in data[f'{fit["name"]}_{kind}']]
        selected={k:v for k,v in groups.items() if k.endswith('_selected')}
        roots={r['root'] for r in data[fixed_name]}
        common={root for root in roots if all(r['success'] for rows in selected.values()
                                            for r in rows if r['root']==root)}
        active={r['root'] for r in data[fixed_name] if r['steps']>0}
        controllable={r['root'] for r in data[fixed_name] if r['steps']>0 and r['protected_experts']>0}
        all_rows={**data,**groups}
        paired_to={**selected,'fixed':data[fixed_name],'frozen':data['fixed_0.0']}
        comparisons={name:base.paired(groups['overlap_selected'],rows,FIELDS,np.random.default_rng(178992))
                     for name,rows in paired_to.items() if name!='overlap_selected'}
        reproduction={}
        if cohort=='diagnostic':
            previous=HERE.parent/'v16_step_reward_toy'/'archive'/'test_endpoints.json.gz'
            with gzip.open(previous,'rt') as file:prior=json.load(file)
            keys=['test_F','monitor_F','steps','success','new_monitor_loss','path_energy',
                  'net_displacement_sq','new_test_loss','actions']
            for name,rows in data.items():
                if name in prior and name.startswith('h1_'):
                    assert [r['root'] for r in rows]==[r['root'] for r in prior[name]]
                    equal=all(r[key]==q[key] for r,q in zip(rows,prior[name]) for key in keys)
                    reproduction[name]=equal
            assert all(reproduction.values()),reproduction
        report[cohort]={
            'summary':{name:stats(rows) for name,rows in all_rows.items()},
            'common_success_roots':sorted(common),
            'common_summary':{name:stats([r for r in rows if r['root'] in common]) for name,rows in paired_to.items()},
            'needs_learning_roots':sorted(active),
            'needs_learning_summary':{name:stats([r for r in rows if r['root'] in active]) for name,rows in paired_to.items()},
            'active_controllable_roots':sorted(controllable),
            'active_controllable_summary':{name:stats([r for r in rows if r['root'] in controllable]) for name,rows in paired_to.items()},
            'paired_overlap_minus_comparator':comparisons,
            'fixed_action_proxy_diagnostic':{key:proxy_ranking(data,key) for key in ('path_energy','overlap_path_energy')},
            'v16_exact_reproduction':reproduction,
            'seed_behavior':{fit['name']:{'selected_round':fit['selected_round'],
                'actions_seen':sorted({a for r in data[f'{fit["name"]}_selected'] for a in r['actions']}),
                'total_switches':sum(r['switches'] for r in data[f'{fit["name"]}_selected'])} for fit in fits}}
        table=[f'## {cohort}', '',
               '|方法|达标|测试遗忘（含失败）|重叠组位移平方|其余组位移平方|更新数|切换次数/历史|',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name in [f'fixed_{x}' for x in [0.,.25,.5,.75,1.]]+list(groups):
            r=report[cohort]['summary'][name]
            table.append(f'|{name}|{r["successes"]}/{r["n"]}|{r["test_F"]:.6f}|{r["overlap_path_energy"]:.6f}|'
                         f'{r["other_path_energy"]:.6f}|{r["steps"]:.1f}|{r["switches"]:.2f}|')
        tables.extend(table+[''])
    report['limits']='Root-bootstrap first averages three policy seeds within each root; does not cover training-seed population uncertainty. Success-conditioned F always accompanied by overall success. Old cohort diagnostic only; all inferential summaries exploratory.'
    (out/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (out/'table.md').write_text('\n'.join(tables).rstrip()+'\n')
    print('\n'.join(tables))
    print('Selected rounds',json.dumps({x['name']:x['selected_round'] for x in fits}))
    print('v16 reproduction',report['diagnostic']['v16_exact_reproduction'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    main(args.directory)

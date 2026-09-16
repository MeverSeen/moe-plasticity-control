"""Summarize a sealed run; paired uncertainty is clustered by entire sequence."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np


METHODS = ['uniform','activation','importance','learned','untrained']
LABELS = ['Uniform','Activation-based','Importance-based','Learned','Untrained policy']
COLORS = ['#65748b','#38a6a6','#8c6cc1','#e27a3f','#afb7c0']


def avg(values):
    return float(np.mean(values)) if len(values) else None


def cluster_estimate(records):
    """records=(sequence_id,paired_difference); transitions are not IID."""
    if not records:
        return dict(mean=None,ci95=[None,None],pairs=0,sequences=0)
    grouped=defaultdict(list)
    for seed,value in records:grouped[seed].append(value)
    groups=list(grouped.values())
    sums=np.array([sum(v) for v in groups])
    counts=np.array([len(v) for v in groups])
    ix=np.random.default_rng(916).integers(len(groups),size=(10000,len(groups)))
    draws=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                pairs=len(records),sequences=len(groups))


def summarize(source,dest):
    dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((source/'test_endpoints.json').read_text())
    terminal=json.loads((source/'test_terminal.json').read_text())
    protocol=json.loads((source/'locked_protocol.json').read_text())
    history=json.loads((source/'training_history.json').read_text())
    paths=json.loads((source/'test_trajectories.json').read_text())
    levels=protocol['config']['levels']
    target=protocol['config']['target']
    result={'thresholds':{},'primary_threshold':target,'paired_comparisons':{},
            'cost':{},'final_all_task_loss':{},'action_diagnostics':{}}
    for threshold in levels:
        result['thresholds'][str(threshold)]={}
        for method in METHODS:
            rr=[r for r in rows if r['method']==method and r['threshold']==threshold]
            ok=[r for r in rr if r['reached']]
            result['thresholds'][str(threshold)][method]=dict(
                reached=len(ok),total=len(rr),reach_rate=len(ok)/len(rr),
                forgetting_conditional=avg([r['old_forgetting'] for r in ok]),
                old_loss_conditional=avg([r['old_absolute_loss'] for r in ok]),
                new_test_loss_conditional=avg([r['new_test_loss'] for r in ok]),
                new_monitor_loss_conditional=avg([r['new_monitor_loss'] for r in ok]),
                overshoot_conditional=avg([r['overshoot'] for r in ok]),
                mean_steps_including_failures=avg([r['steps'] for r in rr]),
                zero_update_endpoints=sum(r['steps']==0 for r in ok))
        by_method={method:{(r['seed'],r['task']):r for r in rows
                           if r['method']==method and r['threshold']==threshold}
                   for method in METHODS}
        comparison={}
        for baseline in [m for m in METHODS if m!='learned']:
            pairs=[];tight=[];nonzero=[];reach=[];absolute=[];new_loss=[]
            for key,a in by_method['learned'].items():
                b=by_method[baseline][key]
                reach.append((key[0],int(a['reached'])-int(b['reached'])))
                if not(a['reached'] and b['reached']):continue
                rec=(key[0],a['old_forgetting']-b['old_forgetting'])
                pairs.append(rec)
                absolute.append((key[0],a['old_absolute_loss']-b['old_absolute_loss']))
                new_loss.append((key[0],a['new_test_loss']-b['new_test_loss']))
                if min(a['new_monitor_loss'],b['new_monitor_loss'])>=threshold*.95:tight.append(rec)
                if min(a['steps'],b['steps'])>0:nonzero.append(rec)
            comparison[baseline]=dict(forgetting=cluster_estimate(pairs),
                tight_loss_sensitivity=cluster_estimate(tight),
                nonzero_update_sensitivity=cluster_estimate(nonzero),
                reach_rate_difference=cluster_estimate(reach),
                old_absolute_loss_difference=cluster_estimate(absolute),
                new_test_loss_difference=cluster_estimate(new_loss))
        result['paired_comparisons'][str(threshold)]=comparison
    for method in METHODS:
        rr=[r for r in terminal if r['method']==method]
        result['cost'][method]=dict(total_online_seconds=sum(r['seconds'] for r in rr),
                total_optimizer_steps=sum(r['steps'] for r in rr))
        result['final_all_task_loss'][method]=cluster_estimate([(r['seed'],r['final_loss']) for r in rr])
        events=[e for p in paths if p['method']==method for e in p['events']]
        alpha=np.array([e['alpha'] for e in events]);protection=np.array([e['protection'] for e in events])
        result['action_diagnostics'][method]=dict(mean_alpha=float(alpha.mean()),
            mean_protection=float(protection.mean()),
            mean_within_decision_expert_alpha_std=float(alpha.std(1).mean()),
            mean_within_decision_expert_protection_std=float(protection.std(1).mean()))
    for filename in ['training_cost.json','tuning_cost.json']:
        result['cost'][filename.replace('.json','')]=json.loads((source/filename).read_text())
    final_map={(r['method'],r['seed']):r['final_loss'] for r in terminal}
    result['final_loss_paired_differences']={b:cluster_estimate([
        (seed,final_map[('learned',seed)]-final_map[(b,seed)]) for seed in protocol['test_seeds']])
        for b in METHODS if b!='learned'}
    (dest/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))

    os.environ.setdefault('MPLCONFIGDIR',str(dest/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(dest/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8.6))
    fig.suptitle('Expert update control: first held-out experiment',fontsize=17,fontweight='bold',y=.98)
    fig.text(.5,.935,'Fixed top-2 MoE | 20 unseen sequences | 80 transitions per method | synthetic regression',
             ha='center',color='#555555',fontsize=10)
    primary=result['thresholds'][str(target)]
    x=np.arange(len(METHODS))
    vals=[primary[m]['reach_rate']*100 for m in METHODS]
    ax[0,0].bar(x,vals,color=COLORS,width=.68)
    for i,v in enumerate(vals):ax[0,0].text(i,v+2,f"{primary[METHODS[i]]['reached']}/80",ha='center',fontsize=9)
    ax[0,0].set_xticks(x,LABELS,rotation=22,ha='right')
    ax[0,0].set_ylim(0,100);ax[0,0].set_ylabel('Target reached (%)')
    ax[0,0].set_title(f'New-task monitor MSE ≤ {target:g}',loc='left',fontweight='bold')

    bs=['uniform','activation','importance','untrained']
    comps=result['paired_comparisons'][str(target)]
    for i,b in enumerate(bs):
        st=comps[b]['forgetting'];mean=st['mean']*1000;lo,hi=np.array(st['ci95'])*1000
        ax[0,1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color=COLORS[METHODS.index(b)])
    ax[0,1].axvline(0,color='#777777',linewidth=1,linestyle='--')
    ax[0,1].set_yticks(range(4),[f"vs {LABELS[METHODS.index(b)]}\n(n={comps[b]['forgetting']['pairs']})" for b in bs])
    ax[0,1].invert_yaxis();ax[0,1].set_xlabel('Learned − baseline forgetting (MSE × 10⁻³)')
    ax[0,1].set_title('Paired difference; sequence bootstrap 95% CI',loc='left',fontweight='bold',fontsize=10)
    ax[0,1].text(.5,-.29,'Left: learned forgets less. Only jointly reached endpoints.',
                  transform=ax[0,1].transAxes,ha='center',fontsize=9,color='#555555')

    for method,label,color in zip(METHODS,LABELS,COLORS):
        ax[1,0].plot(levels,[result['thresholds'][str(t)][method]['reach_rate']*100 for t in levels],
                     marker='o',label=label,color=color,linestyle='--' if method=='untrained' else '-')
    ax[1,0].invert_xaxis();ax[1,0].set_xticks(levels)
    ax[1,0].set_xlabel('Common new-task monitor MSE threshold (stricter →)')
    ax[1,0].set_ylabel('Threshold reached (%)');ax[1,0].set_ylim(45,103)
    ax[1,0].set_title('Reachability across thresholds',loc='left',fontweight='bold')
    ax[1,0].legend(fontsize=8,frameon=False,ncol=2,loc='lower left')

    final=[result['final_all_task_loss'][m]['mean'] for m in METHODS]
    intervals=[result['final_all_task_loss'][m]['ci95'] for m in METHODS]
    err=np.array([[v-r[0],r[1]-v] for v,r in zip(final,intervals)]).T
    ax[1,1].bar(x,final,yerr=err,capsize=4,color=COLORS,width=.68)
    ax[1,1].set_xticks(x,LABELS,rotation=22,ha='right')
    ax[1,1].set_ylabel('All-task test MSE (lower is better)')
    ax[1,1].set_title('End of sequence; includes target failures',loc='left',fontweight='bold',fontsize=10)
    fig.text(.5,.02,'Forgetting = mean positive increase over each old task’s completion loss.\n'
             'Conditional forgetting alone does not establish a gain when reachability is lower.',
             ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.86,bottom=.15,hspace=.72,wspace=.46)
    fig.savefig(dest/'results.png',dpi=180)
    fig.savefig(dest/'results.pdf')
    plt.close(fig)
    print(json.dumps({'primary':primary,'paired':comps,'final':result['final_all_task_loss'],
                      'actions':result['action_diagnostics']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.run,a.out)

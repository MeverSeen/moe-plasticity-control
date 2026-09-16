"""Held-out comparisons: failures retained; histories are the sampling unit."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def estimate(values,seed=281,draws=10000):
    values = np.asarray(values,dtype=float)
    if len(values) == 0:
        return dict(mean=None,ci95=[None,None],histories=0)
    idx = np.random.default_rng(seed).integers(len(values),size=(draws,len(values)))
    samples = values[idx].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),histories=len(values))


def nullable_mean(x):
    return float(np.mean(x)) if len(x) else None


def compare(a,b,seeds):
    paired = [(a[s],b[s]) for s in seeds]
    successes = [(x,y) for x,y in paired if x['success'] and y['success']]
    tight = [(x,y) for x,y in successes if
             min(x['new_monitor_loss'],y['new_monitor_loss']) >= .95*TARGET]
    nonzero = [(x,y) for x,y in successes if min(x['steps'],y['steps']) > 0]
    def diff(rows,field):
        return estimate([x[field]-y[field] for x,y in rows])
    # Positive win score = feedback wins; reachability is compared first.
    wins = []
    for x,y in paired:
        if x['success'] != y['success']:
            wins.append(1 if x['success'] else -1)
        elif x['success']:
            d = x['test_forgetting']-y['test_forgetting']
            wins.append(0 if abs(d) <= 1e-6 else (-1 if d>0 else 1))
        else:
            wins.append(0)
    return dict(reach_difference=estimate([int(x['success'])-int(y['success']) for x,y in paired]),
        forgetting_on_joint_success=diff(successes,'test_forgetting'),
        tight_loss_sensitivity=diff(tight,'test_forgetting'),
        nonzero_update_sensitivity=diff(nonzero,'test_forgetting'),
        all_endpoint_forgetting=diff(paired,'test_forgetting'),
        work_difference=diff(paired,'work'),new_test_difference=diff(successes,'new_test_loss'),
        paired_win_minus_loss=estimate(wins),both_failed=sum(not x['success'] and not y['success'] for x,y in paired))


def analyze(run,out):
    global TARGET
    out.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((run/'locked_protocol.json').read_text())
    rows = json.loads((run/'test_results.json').read_text())
    seeds = protocol['test_seeds']
    replicas = protocol['policy_seeds']
    TARGET = protocol['config']['target']
    methods = ['full_update','fixed','schedule','rule']+[
        f'{mode}_{r}' for mode in ['progress','feedback'] for r in replicas]
    data = {m:{r['seed']:r for r in rows if r['method']==m} for m in methods}
    if any(set(v)!=set(seeds) for v in data.values()):
        raise RuntimeError('Missing or unexpected test histories')
    summary = dict(target=TARGET,methods={},paired={},primary={},mask_diagnostics={})
    for method in methods:
        rr = list(data[method].values())
        ok = [r for r in rr if r['success']]
        summary['methods'][method] = dict(reached=len(ok),total=len(rr),
            reach_rate=len(ok)/len(rr),
            forgetting_success=nullable_mean([r['test_forgetting'] for r in ok]),
            forgetting_all=float(np.mean([r['test_forgetting'] for r in rr])),
            mean_steps=float(np.mean([r['steps'] for r in rr])),
            mean_work=float(np.mean([r['work'] for r in rr])),
            mean_seconds=float(np.mean([r['seconds'] for r in rr])),
            mean_multiplier=float(np.mean([r['mean_multiplier'] for r in rr])),
            mean_action_changes=float(np.mean([r['action_changes'] for r in rr])),
            new_monitor_success=nullable_mean([r['new_monitor_loss'] for r in ok]),
            new_test_success=nullable_mean([r['new_test_loss'] for r in ok]),
            zero_update_success=sum(r['steps']==0 for r in ok))
    for r in replicas:
        feedback = data[f'feedback_{r}']
        for baseline in ['fixed','schedule','rule','full_update',f'progress_{r}']:
            summary['paired'][f'feedback_{r}_vs_{baseline}'] = compare(feedback,data[baseline],seeds)
    # Primary paired matrices; average replicas inside each independent history.
    reach_matrix = np.array([[int(data[f'feedback_{r}'][s]['success'])-
                             int(data[f'progress_{r}'][s]['success']) for s in seeds] for r in replicas])
    common = [s for s in seeds if all(data[f'{m}_{r}'][s]['success']
                                     for r in replicas for m in ['feedback','progress'])]
    forget_matrix = np.array([[data[f'feedback_{r}'][s]['test_forgetting']-
                              data[f'progress_{r}'][s]['test_forgetting'] for s in common] for r in replicas])
    summary['primary']['reach_difference_history_bootstrap'] = estimate(reach_matrix.mean(0))
    summary['primary']['forgetting_difference_all_six_success'] = estimate(forget_matrix.mean(0)) if common else estimate([])
    rng = np.random.default_rng(917)
    for label,matrix in [('reach',reach_matrix),('forgetting',forget_matrix)]:
        if matrix.size:
            ri = rng.integers(len(replicas),size=(10000,len(replicas)))
            si = rng.integers(matrix.shape[1],size=(10000,matrix.shape[1]))
            samples = matrix[ri[:,:,None],si[:,None,:]].mean((1,2))
            summary['primary'][f'{label}_two_way_bootstrap'] = dict(mean=float(matrix.mean()),
                ci95=np.quantile(samples,[.025,.975]).tolist(),training_replicas=len(replicas),
                histories=matrix.shape[1],note='Exploratory two-way bootstrap; only three trained replicas.')
    masses = [data['fixed'][s]['protected_mass'] for s in seeds]
    summary['mask_diagnostics'] = dict(mean_protected_gate_mass=float(np.mean(masses)),
        no_protected_B_mass=sum(x<1e-6 for x in masses),
        nearly_all_protected_B_mass=sum(x>.99 for x in masses),
        initial_target_already_reached=sum(data['fixed'][s]['steps']==0 for s in seeds))
    summary['cost'] = dict(tuning=json.loads((run/'tuning_cost.json').read_text()),
        training={p.stem:json.loads(p.read_text()) for p in run.glob('*_cost.json')
                  if p.stem not in ['tuning_cost','evaluation_cost']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    flat = [{k:v for k,v in r.items() if k not in ['events','mask','initial_old_usage','initial_new_usage']} for r in rows]
    with (out/'endpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    groups=['full_update','fixed','schedule','rule','progress','feedback']
    labels=['Full update','Tuned fixed','Tuned schedule','Feedback rule','Progress PPO','Feedback PPO']
    colors=['#8c98a7','#517c9c','#5c9f95','#ab86b1','#d5a44c','#cf6845']
    group_methods={g:[g] if g in data else [f'{g}_{r}' for r in replicas] for g in groups}
    for i,g in enumerate(groups):
        metrics=[summary['methods'][m] for m in group_methods[g]]
        reach=[r['reach_rate']*100 for r in metrics]
        axes[0,0].bar(i,np.mean(reach),color=colors[i],width=.65)
        axes[0,0].scatter([i]*len(reach),reach,color='#222222',s=15,zorder=3)
        work=[r['mean_work'] for r in metrics]
        axes[1,0].bar(i,np.mean(work),color=colors[i],width=.65)
        axes[1,0].scatter([i]*len(work),work,color='#222222',s=15,zorder=3)
    axes[0,0].set_ylim(0,105)
    axes[0,0].set_title('Target reachability; dots = trained replicas',loc='left')
    axes[0,0].set_ylabel('Reached (%)')
    axes[1,0].set_title('Online workload including monitoring and policy',loc='left')
    axes[1,0].set_ylabel('Forward-example equivalent work')
    for ax in [axes[0,0],axes[1,0]]:
        ax.set_xticks(range(len(groups)),labels,rotation=25,ha='right')
    for i,r in enumerate(replicas):
        result=summary['paired'][f'feedback_{r}_vs_progress_{r}']['forgetting_on_joint_success']
        if result['mean'] is not None:
            mean=result['mean']*1000;lo,hi=np.array(result['ci95'])*1000
            axes[0,1].errorbar(mean,i,xerr=[[max(mean-lo,0)],[max(hi-mean,0)]],fmt='o',capsize=4,color=colors[-1])
    axes[0,1].axvline(0,color='#777777',linestyle='--',linewidth=1)
    axes[0,1].set_yticks(range(len(replicas)),[f'Policy seed {r}' for r in replicas])
    axes[0,1].set_title('Feedback - progress forgetting; paired 95% CI',loc='left')
    axes[0,1].set_xlabel('Old-task MSE difference x 1000 (left is better)')
    # Show ALL history endpoints; unsuccessful histories use hollow markers.
    for mode,color in [('progress',colors[-2]),('feedback',colors[-1])]:
        rr=[r for r in rows if r['method']==f'{mode}_{replicas[0]}']
        for success in [True,False]:
            subset=[r for r in rr if r['success']==success]
            axes[1,1].scatter([r['new_monitor_loss'] for r in subset],
                [r['test_forgetting'] for r in subset],s=18,alpha=.65,
                facecolors=color if success else 'none',edgecolors=color,
                label=f'{mode}: '+('hit' if success else 'miss'))
    axes[1,1].axvline(TARGET,color='#777777',linestyle='--',linewidth=1)
    axes[1,1].set_xlabel('Final new-task monitor MSE')
    axes[1,1].set_ylabel('Old-task forgetting MSE')
    axes[1,1].set_title('All endpoints, first training replica',loc='left')
    axes[1,1].legend(fontsize=8,frameon=False)
    fig.suptitle('Scalar MoE control: does old-task feedback help?',fontsize=17,fontweight='bold')
    fig.text(.5,.935,f'{len(seeds)} unseen histories | 3 matched policy initializations | fixed masks | synthetic regression',ha='center',color='#555555')
    fig.text(.5,.02,'Conditional forgetting is secondary when reachability differs. Work units are a declared proxy, not hardware FLOPs.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.87,bottom=.16,hspace=.62,wspace=.35)
    fig.savefig(out/'results.png',dpi=180)
    fig.savefig(out/'results.pdf')
    plt.close(fig)
    print(json.dumps({'methods':summary['methods'],'primary':summary['primary'],
                      'mask':summary['mask_diagnostics']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.run,a.out)

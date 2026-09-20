"""Standalone scientific figures from completed numerical outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main(a):
 out=Path(__file__).parent/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 d=json.loads((a.diagnostics/'analysis.json').read_text());fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 methods=['retained_positive','retained_balanced','retained_signed','full_monitor_privileged'];labels=['Original','Class-balanced','Signed','Full monitor*'];s=[d['reward'][m]['summary'] for m in methods]
 axes[0].bar(labels,[r['concordance'] for r in s],color=['#758b9f','#197d8c','#bb7549','#b9bdc5']);axes[0].axhline(.5,color='gray',ls=':',lw=1);axes[0].set(ylim=(0,1),ylabel='Non-tied pair concordance',title='Reward ranking (6 eligible roots)');axes[0].tick_params(axis='x',rotation=25)
 for i,r in enumerate(s):axes[0].text(i,r['concordance']+.025,f"{r['concordance']:.1%}",ha='center')
 means=np.array([r['selected_minus_random'] for r in s]);ci=np.array([r['root_bootstrap95'] for r in s]);axes[1].errorbar(means,np.arange(4),xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',color='#197d8c',capsize=4);axes[1].axvline(0,color='gray',ls=':');axes[1].set(yticks=range(4),yticklabels=labels,xlabel='Selected F minus random-choice F',title='Root-bootstrap 95% intervals');axes[1].invert_yaxis();axes[1].xaxis.set_major_locator(MaxNLocator(5))
 raw=json.loads((a.diagnostics/'scope_diagnostic.json').read_text());names=['expert_only','trunk_only','heads_only','nonexpert_only','joint'];lab=['Experts','Trunk','Heads','Trunk + heads','Joint']
 for i,name in enumerate(names):
  v=[r['one_step']['old_loss']-r['initial']['old_loss'] for r in raw if r['mode']==name];axes[2].scatter(i+np.linspace(-.16,.16,len(v)),v,s=13,color='#758b9f',alpha=.65);axes[2].plot(i,np.mean(v),'D',color='#b94b47',ms=6)
 axes[2].axhline(0,color='gray',ls=':');axes[2].set(xticks=range(5),xticklabels=lab,ylabel='One-step old-loss change',title='16 paired starts; red = mean');axes[2].tick_params(axis='x',rotation=25)
 fig.suptitle('v11 CPU diagnostics — exploratory training/development histories',fontsize=14);fig.text(.01,-.035,'* Full monitor is a privileged diagnostic, never a deployable reward. 24/96 branches failed at task 0. Pairs are dependent.',fontsize=9);fig.savefig(out/'diagnostics.png',bbox_inches='tight');plt.close(fig)
 if not a.learning.exists():return
 x=json.loads(a.learning.read_text());arms=x['plan']['arms'];names=['Scratch','Warm','Repair','Repair PPO','Balanced reward','Trunk action'];fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');colors=['#758b9f','#197d8c','#d18c49']
 for mi,mode in enumerate(['greedy','stochastic']):
  ax=axes[mi]
  for j,stage in enumerate(['initial','terminal','selected']):ax.plot(range(6),[x['summary'][arm][mode][stage]['forgetting'] for arm in arms],marker=['o','s','^'][j],label=stage,color=colors[j])
  for key,style,label in [('expert0_trunk1','--','Fixed expert 0, trunk 1'),('expert0_trunk0.5',':','Fixed expert 0, trunk .5')]:ax.axhline(x['baselines'][key]['audit']['forgetting'],color='black',ls=style,lw=1,label=label)
  ax.set(xticks=range(6),xticklabels=names,ylabel='Cumulative forgetting (all histories)',title=mode.capitalize());ax.tick_params(axis='x',rotation=35)
 axes[0].legend(fontsize=8)
 for i,arm in enumerate(arms):
  c=x['contrasts'][f'{arm}/stochastic/terminal-initial']['forgetting'];lo,hi=c['crossed_seed_root_bootstrap95'];axes[2].errorbar(c['difference'],i,xerr=[[c['difference']-lo],[hi-c['difference']]],fmt='o',capsize=4,color='#197d8c')
 axes[2].axvline(0,color='gray',ls=':');axes[2].set(yticks=range(6),yticklabels=names,xlabel='Terminal minus initial F (stochastic)',title='Crossed root/seed bootstrap 95%');axes[2].invert_yaxis();fig.suptitle('v11 learning — 4 controller seeds × 12 new development histories',fontsize=14);fig.text(.01,-.025,'Same development class/image pool; one action stream per root. Intervals are exploratory and unadjusted. Feasibility and final losses are reported separately.',fontsize=9);fig.savefig(out/'learning.png',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,default=Path('runs/v11-diagnostics'));p.add_argument('--learning',type=Path,default=Path('runs/v11-learning/analysis.json'));main(p.parse_args())

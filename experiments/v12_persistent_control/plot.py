import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def main():
 x=json.loads(Path('runs/v12-learning/analysis.json').read_text());out=Path(__file__).parent/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170});fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');arms=['A','B','C'];labels=['A: sparse / 10','B: sparse / 50','C: joint / 50'];colors=['#758b9f','#197d8c','#d18c49']
 for ax,mode in zip(axes[0],['greedy','stochastic']):
  for stage,color in zip(['initial','selected','terminal'],colors):ax.plot(range(3),[x['summary'][a][mode][stage]['forgetting'] for a in arms],marker='o',label=stage,color=color)
  for key,ls,label in [('fixed_best','--','Calibration-selected fixed'),('fixed_v11',':','v11 fixed reference')]:ax.axhline(x['baselines'][key]['audit']['forgetting'],ls=ls,color='black',lw=1,label=label)
  ax.set(xticks=range(3),xticklabels=labels,ylabel='Cumulative forgetting (all histories)',title=mode.capitalize());ax.legend(fontsize=8)
 ax=axes[1,0];names=[]
 for i,(pair,mode) in enumerate([(p,m) for p in ['B-A','C-B'] for m in ['greedy','stochastic']]):
  c=x['contrasts'][f'{pair}/{mode}/terminal']['forgetting'];lo,hi=c['crossed_bootstrap95'];ax.errorbar(c['mean'],i,xerr=[[c['mean']-lo],[hi-c['mean']]],fmt='o',color='#197d8c',capsize=4);names.append(pair+' '+mode)
 ax.axvline(0,ls=':',color='gray');ax.set(yticks=range(4),yticklabels=names,xlabel='Terminal F difference (left minus right)',title='Crossed root/seed bootstrap 95%');ax.xaxis.set_major_locator(MaxNLocator(5));ax.invert_yaxis()
 ax=axes[1,1];means=np.array([[np.mean([c[k] for c in x['costs'][a]])/1000 for a in arms] for k in ['training_updates','training_decisions']]);pos=np.arange(3)
 ax.bar(pos-.17,means[0],.34,label='Learner updates / 1000',color='#758b9f');ax.bar(pos+.17,means[1],.34,label='Control decisions / 1000',color='#197d8c');ax.axhline(20,color='gray',ls=':',lw=1);ax.set(xticks=pos,xticklabels=labels,title='Actual training work (mean of four seeds)');ax.legend(fontsize=8)
 for i,a in enumerate(arms):ax.text(i,means[0,i]+.5,f"{np.mean([c['policy_optimizer_steps'] for c in x['costs'][a]]):.1f} policy steps",ha='center',fontsize=8)
 ax.set_ylim(0,float(means.max()+6));fig.suptitle('v12 GRPO: persistent actions and joint module control',fontsize=15);fig.text(.01,-.02,'Reused development class/image pool; 4 controller seeds × 12 roots, one action stream per root. All failures retained; intervals exploratory and unadjusted.',fontsize=9);fig.savefig(out/'results.png',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()

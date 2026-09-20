"""Static scientific figures from the archived paired root table."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args();r=json.loads((a.analysis/'analysis.json').read_text());rows=r['main'];names=[x['method'] for x in rows];colors=['#386cb0','#7fc97f','#fdc086','#8c6bb1'][:len(names)];plt.rcParams.update({'font.size':10})
 fig,axes=plt.subplots(1,3,figsize=(12,4.5))
 for ax,key,title in zip(axes,['F_common_success','final_mean_EM_common','path_updates_common'],['CE forgetting on common-success roots','Final mean task exact match','Total logical-path updates']):
  for i,row in enumerate(rows):
   value=row[key]
   if value is not None:ax.bar(i,value,color=colors[i]);ax.text(i,value,f'{value:.5f}' if key!='path_updates_common' else str(value),ha='center',va='bottom',fontsize=8)
  if all(row[key] is None for row in rows):ax.text(.5,.5,'No common-success roots',ha='center',transform=ax.transAxes)
  ax.set_title(title,fontsize=10);ax.set_xticks(range(len(names)),names,rotation=20);ax.spines[['top','right']].set_visible(False);ax.margins(y=.25)
 fig.suptitle('One jointly completed root; identical all-one policies diverged\nCausal attribution unresolved; one PPO seed',fontsize=12);fig.tight_layout();fig.savefig(a.analysis/'method_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
 b=r['training']['batch_statistics']
 if b:
  fig,axes=plt.subplots(1,2,figsize=(9,3.5));x=[z['batch'] for z in b]
  axes[0].plot(x,[z['statistics']['parameter_change'] for z in b],marker='o');axes[0].set_ylabel('Policy parameter change (L2)');axes[1].bar(x,[z['statistics']['decisions'] for z in b],color='#8c6bb1');axes[1].set_ylabel('New on-policy decisions')
  for ax in axes:ax.set_xlabel('Two-history PPO rollout batch');ax.set_xticks(x);ax.spines[['top','right']].set_visible(False)
  fig.suptitle('Optimization activity; not a cross-root learning-curve superiority claim');fig.tight_layout();fig.savefig(a.analysis/'ppo_updates.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()

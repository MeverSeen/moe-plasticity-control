"""Plot prespecified paired comparisons from the immutable v8 summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    summary=json.loads((a.run/'analysis/summary.json').read_text())
    rows=[r for r in summary['comparisons'] if r['primary']]
    labels=[r['left']+' minus '+r['right'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    for ax,field,title in zip(axes,['reach','forgetting'],['Reach difference (left minus right)','A forgetting difference on common-success roots']):
        for y,r in enumerate(rows):
            v=r[field]
            if v['mean'] is None:continue
            ax.plot(v['ci_corrected'],[y,y],color='#b8c8d9',lw=4)
            ax.plot(v['ci95'],[y,y],color='#255b89',lw=2)
            ax.scatter(v['mean'],y,color='#18364e',s=20,zorder=3)
        ax.axvline(0,color='black',lw=1);ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[0].axvline(-.05,color='#b45c37',ls='--',label='-5 percentage points')
    axes[0].legend(loc='best');axes[0].set_yticks(range(len(rows)));axes[0].set_yticklabels(labels,fontsize=9);axes[0].invert_yaxis()
    fig.suptitle('Dark: nominal 95% CI; light: 12-comparison Bonferroni sensitivity CI',fontsize=11)
    fig.tight_layout();fig.savefig(a.run/'analysis/paired_comparisons.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

"""Static scientific plots from sealed experiment outputs."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('XDG_CACHE_HOME',str(Path('.cache').resolve()))
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(out):
 data=json.loads((out/'analysis.json').read_text());m=data['methods'];names=list(m)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
 y=np.arange(len(names));colors=['#247ba0' if n.startswith('grpo') else '#677587' for n in names]
 for ax,field,label in zip(axes,['sequence_success','forgetting','final_accuracy'],['Sequence success','Cumulative forgetting (loss)','Final mean accuracy']):
  ax.barh(y,[m[n][field] for n in names],color=colors);ax.set_yticks(y,names if ax==axes[0] else []);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.15)
 axes[0].set_xlim(0,1);axes[2].set_xlim(0,1);fig.suptitle('CIFAR-100: four tasks, unseen classes — descriptive pilot')
 fig.savefig(out/'outcomes.png',dpi=170);plt.close(fig)
 dev=json.loads((out/'dev.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for key,checkpoints in dev.items():
  if not key.startswith(('grpo_feedback','grpo_progress','ppo_feedback')):continue
  epochs=sorted(map(int,checkpoints));rows=[checkpoints[str(e)] for e in epochs]
  axes[0].plot(epochs,[np.mean([r['success'] for r in rr]) for rr in rows],'.-',label=key)
  axes[1].plot(epochs,[np.mean([r['forgetting'] for r in rr]) for rr in rows],'.-',label=key)
 axes[0].set_ylabel('Development sequence success');axes[1].set_ylabel('Development forgetting');axes[1].legend(fontsize=7)
 for ax in axes:ax.set_xlabel('Outer training rounds');ax.grid(alpha=.2)
 fig.savefig(out/'learning.png',dpi=170);plt.close(fig)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);main(ap.parse_args().out)

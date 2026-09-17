"""Exploratory fixed-baseline comparisons and deployment action usage; no retuning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    raw=json.loads((a.run/'test_online.json').read_text());groups={}
    for name,rows in raw.items():
        key=name.rsplit('_',1)[0] if name.rsplit('_',1)[-1].isdigit() else name
        groups.setdefault(key,[]).append(rows)
    rng=np.random.default_rng(88441)
    def stat(values):
        if not len(values):return dict(mean=None,ci95=None)
        boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
        return dict(mean=float(np.mean(values)),ci95=np.quantile(boot,[.025,.975]).tolist())
    comparisons=[]
    for name in groups:
        if not name.startswith(('sup_a','ppo_a')):continue
        for base in ['fixed','global_lr','schedule']:
            aa,bb=groups[name],groups[base]
            reach=lambda rows:np.mean([[r['success'] for r in group] for group in rows],axis=0)
            fg=lambda rows:np.mean([[r['forgetting'] for r in group] for group in rows],axis=0)
            common=np.all([[r['success'] for r in group] for group in aa+bb],axis=0)
            comparisons.append(dict(left=name,right=base,common_roots=int(common.sum()),reach=stat(reach(aa)-reach(bb)),
                forgetting=stat((fg(aa)-fg(bb))[common]),
                left_forgetting=float(fg(aa)[common].mean()) if common.any() else None,
                right_forgetting=float(fg(bb)[common].mean()) if common.any() else None))
    usage={}
    for name,sets in groups.items():
        counts=Counter(d['action'] for rows in sets for r in rows for d in r['decisions'])
        usage[name]=dict(counts=dict(counts),decisions=sum(counts.values()),
            note='Descriptive decisions, not independent samples; histories with more updates contribute more decisions.')
    result=dict(exploratory=True,comparisons=comparisons,action_usage=usage,
                note='Supplementary paired comparisons on already locked data; nominal 95% intervals, not adjusted for these 24 exploratory comparisons. No model or test selection.')
    (a.run/'analysis/baseline_supplement.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

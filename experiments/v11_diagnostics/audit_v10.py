"""Read-only reconstruction of v10 diagnostics, clustered by root/group."""
import gzip, json, itertools, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'v10_visual_control'
OUT=Path(__file__).resolve().parent/'results/v10_audit.json'
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def ranking(rows,proxy='proxy_forgetting',truth='forgetting'):
 eligible=[r for r in rows if r['success']];same=opp=ties=0; gaps=[]
 for a,b in itertools.combinations(eligible,2):
  x=a[proxy]-b[proxy];y=a[truth]-b[truth]
  if abs(x)<=1e-8 or abs(y)<=1e-8:ties+=1;continue
  same+=int(x*y>0);opp+=int(x*y<0);gaps.append([abs(x),abs(y)])
 if eligible:
  v=min(r[proxy] for r in eligible);best=[r for r in eligible if abs(r[proxy]-v)<=1e-8]
  selected=float(np.mean([r[truth] for r in best]));random=float(np.mean([r[truth] for r in eligible]));oracle=min(r[truth] for r in eligible)
 else:selected=random=oracle=None
 return {'eligible':len(eligible),'failures':len(rows)-len(eligible),'concordant':same,'discordant':opp,'ties':ties,'gaps':gaps,'selected':selected,'first_argmin':None if not eligible else min(eligible,key=lambda r:r[proxy])[truth],'random_expectation':random,'privileged_best':oracle,'selection_regret':None if selected is None else selected-oracle}
def main():
 test=read(ROOT/'results/main/test.json.gz');selection=read(ROOT/'results/main/selection.json');beh={}
 for method in ['grpo_feedback','ppo_feedback','grpo_progress']:
  for seed in [1001,1002]:
   key=f'{method}_{seed}';rows=test[key];counts=Counter();alloc=Counter()
   for r in rows:
    first=r['tasks'][0]['steps']//10
    for action in r['actions'][first:]:
     a=np.array(action);counts.update(a.ravel().tolist());alloc.update([str(a.tolist())])
   beh[key]={'selected_epoch':selection[key]['epoch'],'forgetting':float(np.mean([r['forgetting'] for r in rows])),'multiplier_counts':dict(counts),'allocation_counts':dict(alloc),'sequence_success':float(np.mean([r['success'] for r in rows]))}
 groups=[];all_sources=[]
 for seed in [1001,1002]:
  p=ROOT/f'results/main/train_grpo_feedback_{seed}.json.gz';all_sources.append(p)
  for b in read(p):
   g=ranking(b['results']);g.update({'seed':seed,'round':b['round'],'root':b['root'],'all_branch_success':all(r['success'] for r in b['results'])});groups.append(g)
 valid=[g for g in groups if g['selected'] is not None];same=sum(g['concordant'] for g in groups);pairs=same+sum(g['discordant'] for g in groups)
 root_changes={str(root):float(np.mean([g['selected']-g['random_expectation'] for g in valid if g['root']==root])) for root in sorted({g['root'] for g in valid})}
 x=np.array(list(root_changes.values()));rng=np.random.default_rng(11001);ci=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist()
 report={'behavior':beh,'reward_groups':groups,'reward_summary':{'groups':len(groups),'roots':len(root_changes),'concordant':same,'pairs_excluding_ties':pairs,'rate':same/pairs,'first_argmin_eval':np.mean([g['first_argmin'] for g in valid]),'selected_eval':np.mean([g['selected'] for g in valid]),'random_eval':np.mean([g['random_expectation'] for g in valid]),'root_selection_advantage':root_changes,'root_bootstrap95_selected_minus_random':ci,'failure_branches':sum(g['failures'] for g in groups),'ties':sum(g['ties'] for g in groups)},'dynamic_probes':[r for r in read(ROOT/'results/interaction_probe.json') if r['routing']=='dynamic'],'scope':'Exploratory audit of already viewed v10; pairs are not independent; privileged eval never used as deployed reward.','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in all_sources}}
 for seed in [1001,1002]:
  report.setdefault('matrix_equality',{})[str(seed)]={other:all(a['matrix']==b['matrix'] for a,b in zip(test[f'grpo_feedback_{seed}'],test[f'{other}_{seed}'])) for other in ['ppo_feedback','grpo_progress']}
 OUT.write_text(json.dumps(report,indent=2));print(json.dumps(report['reward_summary'],indent=2));print(json.dumps(beh,indent=2))
if __name__=='__main__':main()

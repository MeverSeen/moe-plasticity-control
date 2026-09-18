"""Verify v11 archives, paired histories, frozen plans, and fixed completion metrics."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'experiments/v11_diagnostics'
def read(p):
 if not p.exists():p=p.with_suffix('.json.gz')
 return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def check_rows(rows,roots,classes):
 assert [r['seed'] for r in rows]==roots
 for r in rows:
  assert set(sum(r['classes'],[]))<=set(classes)
  m=np.array(r['matrix']);assert m.shape==(4,4)
  assert abs(np.maximum(m[-1,:-1]-np.diag(m)[:-1],0).mean()-r['forgetting'])<1e-9
  assert r['success']==all(t['success'] for t in r['tasks'])
  assert all(t['success']==(t['loss']<=1.25) and t['steps']<=200 for t in r['tasks'])
  assert r['cost']['updates']==sum(t['steps'] for t in r['tasks'])
  assert all(np.isfinite(t['behavior_logp']) for t in r['policy_trace'])
  assert all(sum(t['class_counts'])==t['n'] and t['n']<=25 for t in r['reward_tasks'])
 return len(rows)
def main():
 count=0;files=0;plan=read(ROOT/'learning_plan.json');ph=hashlib.sha256((ROOT/'learning_plan.json').read_bytes()).hexdigest()
 for name,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 for folder in (ROOT/'results').iterdir():
  if not folder.is_dir():continue
  for name,h in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h;files+=1
  assert not list(folder.rglob('*.pt'))
 for arm in plan['arms']:
  for seed in plan['controller_seeds']:
   folder=ROOT/'results/learning'/f'{arm}_{seed}';job=read(folder/'job.json');assert job['plan_sha256']==ph
   train=read(folder/'training.json');assert len(train)==12
   for b,root in zip(train,plan['train_roots']):
    assert b['root']==root and len(b['results'])==4;count+=check_rows(b['results'],[root]*4,range(60))
   for stage in ['development','audit']:
    allrows=read(folder/(stage+'.json'))
    for modes in allrows.values():
     for mode,rows in modes.items():count+=check_rows(rows,plan['selection_roots' if stage=='development' else 'audit_roots'],range(60,80))
   lock=read(folder/'locked_selection.json');dev=read(folder/'development.json')
   for mode,epoch in lock['selected'].items():
    score=lambda u:(-np.mean([r['success'] for r in dev[str(u)][mode]]),np.mean([r['forgetting'] for r in dev[str(u)][mode]]),np.mean([r['final_loss'] for r in dev[str(u)][mode]]))
    assert epoch==min(plan['checkpoints'],key=score)
   for epoch,h in lock['model_sha256'].items():
    local=ROOT.parents[1]/'runs/v11-learning'/f'{arm}_{seed}'/f'epoch{epoch}.pt'
    if local.exists():assert hashlib.sha256(local.read_bytes()).hexdigest()==h
 for stages in read(ROOT/'results/baselines/results.json').values():
  for stage,rows in stages.items():count+=check_rows(rows,plan[stage+'_roots'],range(60,80))
 print(f'Verified {files} v11 archive files and {count} learning/evaluation trajectories; frozen source hashes and paired data boundaries match.')
if __name__=='__main__':main()

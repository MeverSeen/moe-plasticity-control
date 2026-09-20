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
 count=0;files=0;learning_updates=0;plan=read(ROOT/'learning_plan.json');ph=hashlib.sha256((ROOT/'learning_plan.json').read_bytes()).hexdigest()
 for name,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 for folder in (ROOT/'results').iterdir():
  if not folder.is_dir():continue
  for name,h in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h;files+=1
  assert not list(folder.rglob('*.pt'))
 diagnostics=read(ROOT/'results/diagnostics/reward_diagnostic.json');assert len(diagnostics)==16
 for g in diagnostics:assert len(g['branches'])==6;count+=check_rows(g['branches'],[g['root']]*6,range(60) if g['split']=='train' else range(60,80))
 scope=read(ROOT/'results/diagnostics/scope_diagnostic.json');assert len(scope)==96
 for key in {(r['root'],r['stage']) for r in scope}:
  rows=[r for r in scope if (r['root'],r['stage'])==key];assert len(rows)==6 and len({r['one_step_moment_hash'] for r in rows})==1
  for r in rows:assert r['success']==(r['end']['new_loss']<=1.25) and r['steps']<=200
 calibration=read(ROOT/'results/calibration/results.json');assert sum(map(len,calibration.values()))==72
 for rows in calibration.values():count+=check_rows(rows,list(range(1120100,1120108)),range(60,80))
 assert all(a['matrix']==b['matrix'] and a['actions']==b['actions'] for a,b in zip(calibration['budget200_expert0'],calibration['stratified_monitor_only']))
 for folder,name,backup in [('diagnostics','diagnose.py','diagnose.before_serialization_fix.py.txt'),('calibration','calibrate.py','calibrate.before_serialization_fix.py.txt')]:
  h=read(ROOT/f'results/{folder}/protocol.json')['sources'][name];assert hashlib.sha256((ROOT/'provenance'/backup).read_bytes()).hexdigest()==h
 for arm in plan['arms']:
  for seed in plan['controller_seeds']:
   folder=ROOT/'results/learning'/f'{arm}_{seed}';job=read(folder/'job.json');assert job['plan_sha256']==ph
   train=read(folder/'training.json');assert len(train)==12
   cost=read(folder/'cost.json');assert cost['training_updates']==sum(r['branch_cost']['updates'] for b in train for r in b['results'])
   assert cost['prefix_updates']==sum(r['updates'] for r in read(folder/'prefix_cost.json'))
   learning_updates+=cost['training_updates']+cost['prefix_updates']+cost['dev_updates']+cost['audit_updates']
   for b,root in zip(train,plan['train_roots']):
    assert b['root']==root and len(b['results'])==4;count+=check_rows(b['results'],[root]*4,range(60))
   for stage in ['development','audit']:
    allrows=read(folder/(stage+'.json'))
    assert cost['dev_updates' if stage=='development' else 'audit_updates']==sum(r['cost']['updates'] for modes in allrows.values() for rows in modes.values() for r in rows)
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
  for stage,rows in stages.items():count+=check_rows(rows,plan[stage+'_roots'],range(60,80));learning_updates+=sum(r['cost']['updates'] for r in rows)
 ledger=read(ROOT/'results/learning/work_ledger.json');assert learning_updates==ledger['learning']['all_learning_stage_updates'];assert count==ledger['completed_full_history_executions']
 models=read(ROOT/'results/learning/model_manifest.json');assert len(models)==len(plan['arms'])*len(plan['controller_seeds'])*len(plan['checkpoints'])
 for name,h in models.items():
  local=ROOT.parents[1]/'runs/v11-learning'/name
  if local.exists():assert hashlib.sha256(local.read_bytes()).hexdigest()==h
 print(f'Verified {files} v11 archive files, {count} full-history trajectories and 96 scope paths; frozen source hashes and paired data boundaries match.')
if __name__=='__main__':main()

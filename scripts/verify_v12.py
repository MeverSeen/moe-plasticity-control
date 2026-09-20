import gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'experiments/v12_persistent_control'
def read(p):
 if not p.exists():p=p.with_suffix(p.suffix+'.gz')
 return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def rows_check(rows,roots,classes,hold,joint,base):
 assert [r['seed'] for r in rows]==roots
 for r in rows:
  assert set(sum(r['classes'],[]))<=set(classes);m=np.array(r['matrix']);assert m.shape==(4,4);assert abs(np.maximum(m[-1,:-1]-np.diag(m)[:-1],0).mean()-r['forgetting'])<1e-9
  assert r['success']==all(t['success'] for t in r['tasks']);assert all(t['success']==(t['loss']<=1.25) and t['steps']<=200 and t['steps']%10==0 for t in r['tasks']);assert r['cost']['updates']==sum(t['steps'] for t in r['tasks'])
  trace=r['policy_trace'];assert len(trace)==r['cost']['control_decisions'];assert sum(t['duration'] for t in trace)==sum(t['steps'] for t in r['tasks'][1:])
  for task in range(1,4):
   tt=[t for t in trace if t['task']==task];assert sum(t['duration'] for t in tt)==r['tasks'][task]['steps']
   for i,t in enumerate(tt):
    assert t['step']==sum(v['duration'] for v in tt[:i]);assert 0<t['duration']<=hold and t['duration']%10==0;assert t['requested_hold']==hold and np.isfinite(t['behavior_logp'])
    if i<len(tt)-1:assert t['duration']==hold
    mm=np.array(t['experts']);assert mm.shape==(2,4);assert set(mm.flatten())<={0,.25,.5,.75,1,1.5};assert t['trunk'] in [.25,.5,1]
    if not joint:assert ((mm!=base).sum(1)<=1).all()
 return len(rows)
def main():
 plan=read(ROOT/'plan.json');ph=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest();count=0;files=0;updates=0
 for name,h in plan['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 for folder in (ROOT/'results').iterdir():
  if not folder.is_dir():continue
  for name,h in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h;files+=1
  assert not list(folder.rglob('*.pt'))
 calibration=read(ROOT/'results/calibration/results.json');sources=read(ROOT/'results/calibration/sources.json');selection=read(ROOT/'results/calibration/selection.json');old=read(ROOT.parent/'v11_diagnostics/results/calibration/results.json.gz');assert len(calibration)==18
 for name,rows in calibration.items():
  assert [r['seed'] for r in rows]==list(range(1120100,1120108))
  for r in rows:
   m=np.array(r['matrix']);assert abs(np.maximum(m[-1,:-1]-np.diag(m)[:-1],0).mean()-r['forgetting'])<1e-9;assert set(sum(r['classes'],[]))<=set(range(60,80))
  if sources[name].startswith('v11:'):assert rows==old[sources[name].split(':',1)[1]]
 score=lambda k:(-np.mean([r['success'] for r in calibration[k]]),np.mean([r['forgetting'] for r in calibration[k]]),np.mean([r['final_loss'] for r in calibration[k]]))
 best=min(calibration,key=score);assert list(map(float,best.split('/')))==[plan['base']['expert'],plan['base']['trunk']];assert selection['new_episodes']==sum(len(v) for k,v in calibration.items() if sources[k]=='new')==80
 for arm,opts in plan['arms'].items():
  for seed in plan['controller_seeds']:
   p=ROOT/'results/learning'/f'{arm}_{seed}';job=read(p/'job.json');assert job['plan_sha256']==ph;cost=read(p/'cost.json');train=read(p/'training.json');lock=read(p/'locked_selection.json');dev=read(p/'development.json');audit=read(p/'audit.json');work=read(p/'checkpoint_work.json')
   assert set(dev)=={'initial','mid','terminal'} and set(audit)==set(lock['audit_checkpoints']);assert 20000<=cost['training_updates']<22400
   assert cost['training_updates']==sum(r['branch_cost']['updates'] for b in train for r in b['results']);assert cost['policy_optimizer_steps']==sum(b['optimizer_steps'] for b in train);assert cost['training_decisions']==sum(b['decisions'] for b in train)
   assert work['terminal']['groups']==len(train) and work['terminal']['training_updates']==cost['training_updates'];assert 10000<=work['mid']['training_updates']<12400
   for i,b in enumerate(train):
    root=plan['train_roots'][i%len(plan['train_roots'])];assert b['root']==root;assert b['decisions']==sum(len(r['policy_trace']) for r in b['results']);count+=rows_check(b['results'],[root]*4,range(60),opts['hold'],opts['joint'],plan['base']['expert'])
   for stage,d in [('selection',dev),('audit',audit)]:
    for modes in d.values():
     for rows in modes.values():count+=rows_check(rows,plan[stage+'_roots'],range(60,80),opts['hold'],opts['joint'],plan['base']['expert'])
   for mode,label in lock['selected'].items():
    score=lambda k:(-np.mean([r['success'] for r in dev[k][mode]]),np.mean([r['forgetting'] for r in dev[k][mode]]),np.mean([r['final_loss'] for r in dev[k][mode]]));assert label==min(['initial','mid','terminal'],key=score)
   for label,h in lock['model_sha256'].items():
    local=ROOT.parents[1]/'runs/v12-learning'/f'{arm}_{seed}'/(label+'.pt')
    if local.exists():assert hashlib.sha256(local.read_bytes()).hexdigest()==h
   updates+=cost['training_updates']+cost['prefix_updates']+cost['evaluation_cost']['updates']
 for name,stages in read(ROOT/'results/baselines/results.json').items():
  hold,joint=(50,True) if name.startswith('fixed') else (plan['arms'][name[0]]['hold'],plan['arms'][name[0]]['joint'])
  for stage,rows in stages.items():count+=rows_check(rows,plan[stage+'_roots'],range(60,80),hold,joint,plan['base']['expert']);updates+=sum(r['cost']['updates'] for r in rows)
 fixed=read(ROOT/'results/baselines/results.json')['fixed_best']['audit']
 for arm in plan['arms']:
  for seed in plan['controller_seeds']:
   initial=read(ROOT/'results/learning'/f'{arm}_{seed}'/'audit.json')['initial']['greedy']
   assert all(a['matrix']==b['matrix'] and a['tasks']==b['tasks'] for a,b in zip(initial,fixed))
 probability=read(ROOT/'results/learning/probability_validation.json');assert probability['max_logprob_error']<1e-5 and probability['checked_decisions']>0
 print(f'Verified {files} v12 archive files and {count} main training/evaluation histories; {updates} learner updates. Duration accounting, probability records, budgets, source/model hashes and data boundaries match.')
if __name__=='__main__':main()

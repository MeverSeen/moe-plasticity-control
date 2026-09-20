"""Replay saved behavior policies and verify real training/confirmation evidence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from structured_policy import ActorCritic,Rule,shaped_rewards

def read(p):return json.loads(p.read_text())
def close(a,b,tol=2e-4):assert abs(a-b)<=tol,(a,b)
def policy(path):
 p=ActorCritic();s=torch.load(path,map_location='cpu',weights_only=False);p.load_state_dict(s['policy']);p.eval();return p

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);a=parser.parse_args();root=a.archive;lock=read(root/'source_lock.json');plan=read(root/'method_protocol.json');counts={'source_files':0,'histories':0,'decisions':0,'general_items':0,'ppo_batches':0,'optimizer_steps':0}
 for name,expected in lock['sha256'].items():assert hashlib.sha256((root/'provenance'/name).read_bytes()).hexdigest()==expected,name;counts['source_files']+=1
 trainsets=[set(s['tasks']) for s in plan['roots']['train']];dev=set(plan['roots']['development'][0]['tasks']);confirm=set(t for s in plan['roots']['confirmation'] for t in s['tasks']);assert not set.union(*trainsets)&(dev|confirm) and not dev&confirm
 summary=read(root/'summary.json') if (root/'summary.json').exists() else read(root.parent/'recovered_summary.json')
 if summary['frozen_base_hash'] is not None:assert summary['frozen_base_hash']['sha256']==plan['frozen_base_hash']
 else:assert summary['frozen_base_hash_expected']==plan['frozen_base_hash'] and summary['source_exit_code']==1
 for path in (root/'targets').glob('*.json'):
  target=read(path)
  if target.get('reused_from_pilot'):continue
  ref=read(root/'results/target_reference'/path.name);expected=[float(np.median([x['loss'] for x in curve if x['step'] in [20,25,30]])) for curve in ref['loss_curves']];assert expected==target['root']['targets']
 initial=read(root/'reused_pilot/initial_general.json')
 selected=policy(root/'controllers/ppo_selected.pt');final=policy(root/'controllers/ppo_final.pt');assert all(torch.equal(v,final.state_dict()[k]) for k,v in selected.state_dict().items())
 for phase in ['train','development','confirmation']:
  for path in sorted((root/'results'/phase).glob('*.json')):
   if path.stem.endswith('__general') or path.stem.endswith('__incomplete'):continue
   r=read(path);assert r['history_completed'];assert r['method_protocol_sha256']==hashlib.sha256((root/'method_protocol.json').read_bytes()).hexdigest();name=r['candidate']['name'];prefix=read(root/'prefixes'/phase/(r['root']['id']+'.json'));assert r['checkpoint']['prefix_sha256']==prefix['sha256']
   for key,value in prefix['cost'].items():close(r['cost_path'][key],r['cost_actual'][key]+value,1e-6*max(1,abs(value)))
   if phase=='train':
    i=int(name.rsplit('_',1)[1]);batch=i//2;p=policy(root/'controllers'/('ppo_initial.pt' if batch==0 else f'ppo_batch_{batch:03}.pt'))
   elif name.startswith('ppo'):p=selected
   else:p=Rule(r['candidate']['base_indices'],mode=r['candidate']['mode'])
   assert sum(x['duration'] for x in r['trace'])==sum(t['steps'] for t in r['tasks'][1:]);assert r['cost_path']['updates']==sum(t['steps'] for t in r['tasks'])
   for t in r['trace']:
    obs=tuple(torch.tensor(v) for v in t['observation']);latent=[torch.tensor(v) for v in t['latent']]
    with torch.no_grad():action,z,lp,entropy,value=p.act(obs,latent)
    assert torch.equal(action,torch.tensor(t['experts']));close(float(lp),t['policy_logp']);close(float(value),t['value']);assert 0<t['duration']<=5
    if phase!='train':assert torch.equal(p.act(obs,deterministic=True)[0],action)
    counts['decisions']+=1
   if r['trace']:assert np.allclose(shaped_rewards([t['potential'] for t in r['trace']],r['monitor']['forgetting'],r['failure_count']),[t['reward'] for t in r['trace']],atol=1e-8)
   close(r['reward']['telescope_error'],0,1e-8);counts['histories']+=1
   if phase!='train':
    for key,detail in r['evaluation']['score_details'].items():
     k,j=map(int,key.split('/'));close(np.mean([x['exact_match'] for x in detail['items']]),r['evaluation']['score_matrix'][k][j],1e-8);close(np.mean([x['token_f1'] for x in detail['items']]),detail['token_f1'],1e-8)
 for path in (root/'results/confirmation').glob('*__general.json'):
  g=read(path);assert g['adapter_enabled'] and len(g['items'])==612;assert g['manifest_sha256']==initial['manifest_sha256']
  for item in g['items']:assert item['predicted_index']==int(np.argmax(item['choice_log_probabilities']));assert item['correct']==(item['predicted_index']==item['answer_index']);counts['general_items']+=1
  for ability,row in g['abilities'].items():
   items=[x for x in g['items'] if x['ability']==ability];assert row['ids']==initial['abilities'][ability]['ids'];close(np.mean([x['correct'] for x in items]),row['accuracy'],1e-8);close(np.mean([x['answer_nll'] for x in items]),row['answer_nll'],1e-8)
 learning=read(root/'learning.json') if (root/'learning.json').exists() else []
 for batch in learning:assert batch['completed_histories']==2*batch['batch'];counts['ppo_batches']+=1;counts['optimizer_steps']+=len(batch['statistics'].get('epochs',[]))
 previous=policy(root/'controllers/ppo_initial.pt')
 for batch in learning:
  current=policy(root/'controllers'/batch['checkpoint']['path']);change=sum(float((current.state_dict()[k]-v).square().sum()) for k,v in previous.state_dict().items())**.5;close(change,batch['statistics']['parameter_change'],1e-6);previous=current
 completed_updates=0
 for path in (root/'results').rglob('*.json'):
  if path.stem.endswith('__general'):continue
  row=read(path);completed_updates+=row.get('cost_actual',{}).get('updates',0)
 for path in (root/'prefixes').rglob('*.json'):completed_updates+=read(path)['cost']['updates']
 counts['accounted_learner_updates']=completed_updates;counts['unattributed_learner_updates']=summary['native_cost']['updates']-completed_updates;assert counts['unattributed_learner_updates']>=0
 if summary['status']=='completed':assert counts['unattributed_learner_updates']==0
 manifest_path=root.parent/'archive_manifest.json'
 if manifest_path.exists():
  manifest=read(manifest_path)
  actual={str(p.relative_to(root.parent)) for p in root.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=manifest_path}
  assert actual==set(manifest['sha256']), 'Archive inventory differs from manifest'
  for name,expected in manifest['sha256'].items():assert hashlib.sha256((root.parent/name).read_bytes()).hexdigest()==expected,name
  counts['manifest_files']=len(actual)
 print(json.dumps({'status':'passed',**counts}))
if __name__=='__main__':main()

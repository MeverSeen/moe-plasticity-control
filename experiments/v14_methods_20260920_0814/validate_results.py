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
 selected=policy(root/'controllers/ppo_selected.pt');final=policy(root/'controllers/ppo_final.pt');assert all(torch.equal(v,final.state_dict()[k]) for k,v in selected.state_dict().items())
 for phase in ['train','development','confirmation']:
  for path in sorted((root/'results'/phase).glob('*.json')):
   if path.stem.endswith('__general') or path.stem.endswith('__incomplete'):continue
   r=read(path);assert r['history_completed'];name=r['candidate']['name']
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
  g=read(path);assert g['adapter_enabled'] and len(g['items'])==612
  for item in g['items']:assert item['predicted_index']==int(np.argmax(item['choice_log_probabilities']));assert item['correct']==(item['predicted_index']==item['answer_index']);counts['general_items']+=1
  for ability,row in g['abilities'].items():
   items=[x for x in g['items'] if x['ability']==ability];close(np.mean([x['correct'] for x in items]),row['accuracy'],1e-8);close(np.mean([x['answer_nll'] for x in items]),row['answer_nll'],1e-8)
 learning=read(root/'learning.json') if (root/'learning.json').exists() else []
 for batch in learning:assert batch['completed_histories']==2*batch['batch'];counts['ppo_batches']+=1;counts['optimizer_steps']+=len(batch['statistics'].get('epochs',[]))
 print(json.dumps({'status':'passed',**counts}))
if __name__=='__main__':main()

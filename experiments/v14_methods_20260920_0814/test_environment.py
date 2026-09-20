import copy,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import environment as module
from environment import Environment,Config,BudgetStop
from structured_policy import Rule

class FakeNative:
 def __init__(self,trigger=None):
  self.model=SimpleNamespace(eval=lambda:None);self.optim=SimpleNamespace(relative=torch.zeros(1024));self.weights=torch.zeros(3);self.trigger=trigger;self.cost={k:0 for k in ['updates','train_input_tokens','train_target_tokens','train_padded_tokens','replay_input_tokens','replay_target_tokens','replay_padded_tokens','monitor_input_tokens','monitor_target_tokens','monitor_padded_tokens','eval_input_tokens','eval_target_tokens','eval_padded_tokens','copies','resets']};self.cost.update(copy_seconds=0.,reset_seconds=0.)
 def reset(self):self.weights.zero_();self.optim.relative.zero_();self.cost['resets']+=1
 def loss(self,rows,role='monitor',routes=False):
  for k in ['input','padded']:self.cost[role+'_'+k+'_tokens']+=2*len(rows)
  self.cost[role+'_target_tokens']+=len(rows);loss=torch.stack([1-.025*self.weights[r['ids'][0]]+.005*(self.weights.sum()-self.weights[r['ids'][0]]) for r in rows]);stats={l:(torch.ones(64)*len(rows)*2/8,torch.ones(64)*len(rows)*2/64,len(rows)*2) for l in range(16)} if routes else None;return loss,stats
 def step(self,current,replay,m):
  j=current[0][0]['ids'][0]
  for rows in current:self.loss(rows,'train')
  for rows in replay:self.loss(rows,'replay')
  self.weights[j]+=m.mean();self.optim.relative.fill_(.001);self.cost['updates']+=1
  if self.cost['updates']==self.trigger:module.CHECKPOINT_REQUEST=True
 def snapshot(self):
  self.cost['copies']+=1;return {'weights':self.weights.clone(),'relative':self.optim.relative.clone(),'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
 def restore(self,s):self.weights.copy_(s['weights']);self.optim.relative.copy_(s['relative']);torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['python']);self.cost['copies']+=1
 def check_frozen(self):pass

def data():return {str(j):{role:[{'ids':[j,i],'labels':[-100,1],'id':f'{j}-{role}-{i}','input_tokens':2,'target_tokens':1} for i in range(96)] for role in ['train','monitor','eval']} for j in range(3)}
def spec():return {'id':'fake','seed':12,'split':'train','tasks':['0','1','2'],'targets':[.96,.95,.95]}
def config(hold=3):return Config(max_updates=8,reference_updates=6,hold=hold,stop_examples=2,evaluation_batch=2,memory_tokens=600)

class Tests(unittest.TestCase):
 def tearDown(self):module.CHECKPOINT_REQUEST=False
 def test_holds_and_cached_reward_identity(self):
  rows=[]
  for hold in [1,3]:
   e=Environment(FakeNative(),data(),spec(),config(hold));r=e.run(Rule([2]*16,mode='fixed'));rows.append(r);self.assertAlmostEqual(r['reward']['telescope_error'],0);self.assertEqual(sum(t['duration'] for t in r['trace']),sum(t['steps'] for t in r['tasks'][1:]));self.assertTrue(all(t['duration']<=hold for t in r['trace']));self.assertTrue(r['history_completed']);self.assertEqual(r['n_fail'],0)
  self.assertEqual(rows[0]['evaluation'],rows[1]['evaluation']);self.assertGreater(len(rows[0]['trace']),len(rows[1]['trace']))
 def test_mid_hold_serialized_restore_sampling_and_cache(self):
  reference=Environment(FakeNative(),data(),spec(),config()).run(Rule([2]*16,mode='fixed'),True,41)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.pt';e=Environment(FakeNative(trigger=4),data(),spec(),config())
   with self.assertRaises(BudgetStop):e.run(Rule([2]*16,mode='fixed'),True,41,checkpoint_path=p)
   state=torch.load(p,weights_only=False);self.assertIsNotNone(state['environment']['held']);self.assertGreater(state['environment']['held_remaining'],0);module.CHECKPOINT_REQUEST=False
   restored=Environment.from_state(FakeNative(),data(),config(),state['environment'],state['learner']);result=restored.run(Rule([2]*16,mode='fixed'),True,999)
   self.assertEqual(reference['trace'],result['trace']);self.assertEqual(reference['evaluation'],result['evaluation']);self.assertEqual(reference['reward'],result['reward'])
 def test_prefix_restore_same_complete_start_and_path_accounting(self):
  learner=FakeNative();e=Environment(learner,data(),spec(),config());prefix,info=e.prepare_prefix(Path('/tmp/unused-v13.pt'));rows=[]
  for _ in range(2):
   q=Environment.from_state(learner,data(),config(),prefix['environment'],prefix['learner']);rows.append(q.run(Rule([2]*16,mode='fixed')))
  self.assertEqual(rows[0]['trace'],rows[1]['trace']);self.assertEqual(rows[0]['checkpoint']['prefix_sha256'],info['sha256']);self.assertEqual(rows[0]['cost_path']['updates'],sum(t['steps'] for t in rows[0]['tasks']))
if __name__=='__main__':unittest.main()

"""Real-data joint coverage and genuine grouped-trajectory optimization smoke."""
import hashlib,json,time
from pathlib import Path
import torch
from core import Data,Config,Environment
from policy import Policy,Baseline
from control import advance,rollout,train_batch
from run import sample_audit
from calibrate import save
def main():
 torch.set_num_threads(1);out=Path('runs/v12-smoke');out.mkdir(exist_ok=False,parents=True);sel=json.loads(Path('runs/v12-calibration/selection.json').read_text());base,trunk=sel['expert'],sel['trunk'];data=Data('.cache/datasets/cifar-complete.tar.gz');cfg=Config();root=Environment(data,cfg,1230900,'train')
 while root.k==0:advance(root,Baseline(base,trunk),10)
 save(out/'protocol.json',{'root':1230900,'controller_seed':1200,'base':sel,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
 for arm,hold,joint in [('A',10,False),('B',50,False),('C',50,True)]:
  torch.manual_seed(1200);p=Policy(base,trunk,joint);obs=root.clone().observation();coverage=sample_audit(p,obs,base,trunk)
  assert 350<coverage['keep']<650 and coverage['trunk_counts'][str(trunk)]>600
  assert coverage['multi_expert']>150 if joint else coverage['multi_expert']==0
  assert coverage['joint_trunk_and_experts']>30
  torch.manual_seed(1270900);batch=train_batch(p,torch.optim.Adam(p.parameters(),lr=.001),root,hold,4)
  assert batch['optimizer_steps']==2 and batch['parameter_change']>0
  assert all(len(r['policy_trace'])==r['branch_cost']['control_decisions'] and sum(t['duration'] for t in r['policy_trace'])==sum(t['steps'] for t in r['tasks'][1:]) for r in batch['results'])
  save(out/(arm+'.json'),{'coverage':coverage,'batch':batch});print(arm,coverage,'seconds',batch['wall_seconds'],'updates',sum(r['branch_cost']['updates'] for r in batch['results']),flush=True)
 save(out/'prefix_cost.json',root.cost)
if __name__=='__main__':main()

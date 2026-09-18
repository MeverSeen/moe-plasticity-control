"""Budget-endpoint diagnostic, without choosing checkpoints from heldout outcomes."""
import hashlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/v10_visual_control'))
from core import Config,Data
from control import Policy
from run import evaluate,save,source_hashes

def main():
 torch.set_num_threads(1);out=Path('runs/v10-terminal');out.mkdir(exist_ok=False)
 cases=[('grpo12','runs/v10-main','grpo_feedback',12),('grpo36','runs/v10-rounds36','grpo_feedback',36),('ppo12','runs/v10-main','ppo_feedback',12)]
 cfg=Config(**json.loads(Path('experiments/v10_visual_control/config.json').read_text()));plan={'reason':'Development checkpoint grids 0/6/12 and 0/18/36 are non-nested. Evaluate fixed terminal actors to separate training duration from selection timing. This supplementary diagnostic was added after coverage2 evaluation, based on development selection design, not heldout performance; not confirmatory.','cases':[],'source_hashes':source_hashes(),'seeds':[1001,1002],'roots':list(range(1030000,1030008)),'config':cfg.__dict__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 for name,folder,variant,epoch in cases:
  for seed in plan['seeds']:
   path=Path(folder)/f'{variant}_{seed}_epoch{epoch}.pt';plan['cases'].append({'name':f'{name}_{seed}','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 save(out/'locked_plan.json',plan);data=Data('.cache/datasets/cifar-complete.tar.gz',final=True);rows={};start=time.perf_counter()
 for case in plan['cases']:
  p=Policy();p.load_state_dict(torch.load(case['path'],weights_only=True));rows[case['name']]=evaluate(data,cfg,plan['roots'],p,'test','test');save(out/'results.json',rows)
  print(case['name'],'finished',flush=True)
 save(out/'cost.json',{'wall_seconds':time.perf_counter()-start})
if __name__=='__main__':main()

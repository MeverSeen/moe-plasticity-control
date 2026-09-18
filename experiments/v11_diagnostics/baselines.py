"""Fixed baselines on the exact selection and development replication roots."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from core import Config,Data
from policies import FixedPolicy
from learn import evaluate,score
from diagnose import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.perf_counter();data=Data('.cache/datasets/cifar-complete.tar.gz');cfg=Config(**plan['config']);results={}
 for candidate in plan['fixed_baselines']:
  name=f"expert{candidate['expert']}_trunk{candidate['trunk']}";results[name]={}
  for stage in ['selection','audit']:
   rows=evaluate(data,cfg,plan[stage+'_roots'],FixedPolicy(candidate['expert'],candidate['trunk']),plan[stage+'_action_seed']);results[name][stage]=rows;save(a.out/'results.json',results);print(name,stage,score(rows),flush=True)
 save(a.out/'cost.json',{'wall_seconds':time.perf_counter()-start,'updates':sum(r['cost']['updates'] for stages in results.values() for rows in stages.values() for r in rows),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest()})
if __name__=='__main__':main()

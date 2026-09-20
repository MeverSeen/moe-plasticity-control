"""Strong fixed plus interface-valid progress/feedback rules, matched stopping."""
import argparse,json,time
from pathlib import Path
import torch
from core import Config,Data
from policy import Baseline
from run import evaluate
from calibrate import save,score
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);data=Data('.cache/datasets/cifar-complete.tar.gz');cfg=Config(**plan['config']);base,trunk=plan['base']['expert'],plan['base']['trunk'];start=time.perf_counter();result={}
 policies=[('fixed_best',Baseline(base,trunk),50),('fixed_v11',Baseline(0,.5),50),('fixed_trunk1',Baseline(base,1),50)]
 policies += [(arm+'_'+rule,Baseline(base,trunk,rule,v['joint']),v['hold']) for arm,v in plan['arms'].items() for rule in ['progress','feedback']]
 for name,policy,hold in policies:
  result[name]={}
  for stage in ['selection','audit']:
   result[name][stage]=evaluate(data,cfg,plan[stage+'_roots'],policy,hold,plan[stage+'_action_seed']);save(a.out/'results.json',result);print(name,stage,score(result[name][stage]),flush=True)
 save(a.out/'cost.json',{'seconds':time.perf_counter()-start,'episodes':sum(len(rows) for stages in result.values() for rows in stages.values()),'cost':{k:sum(r['cost'][k] for stages in result.values() for rows in stages.values() for r in rows) for k in result['fixed_best']['audit'][0]['cost']}})
if __name__=='__main__':main()

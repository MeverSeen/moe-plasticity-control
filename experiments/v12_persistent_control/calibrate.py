"""Complete static grid; reuse only verified identical v11 fixed trajectories."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from core import Data,Config,Environment
from policy import Baseline
from control import rollout

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())
def score(rows):return (-float(np.mean([r['success'] for r in rows])),float(np.mean([r['forgetting'] for r in rows])),float(np.mean([r['final_loss'] for r in rows])))
def main():
 torch.set_num_threads(1);out=Path('runs/v12-calibration');out.mkdir(exist_ok=False,parents=True);data=Data('.cache/datasets/cifar-complete.tar.gz');cfg=Config();old=read(Path('experiments/v11_diagnostics/results/calibration/results.json.gz'));roots=list(range(1120100,1120108));start=time.perf_counter();results={};sources={}
 save(out/'protocol.json',{'roots':roots,'experts':[0,.25,.5,.75,1,1.5],'trunk':[.25,.5,1],'config':cfg.__dict__,'selection':'lexicographic success, F, absolute loss; report feasible only if success>=.9','sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
 # One real-data parity check before accepting any reuse.
 r,_=rollout(Environment(data,cfg,roots[0],'dev'),Baseline(0,.5),50);reference=old['budget200_expert0_trunk0.5'][0]
 assert r['matrix']==reference['matrix'] and r['tasks']==reference['tasks'] and r['reward']==reference['reward'];save(out/'reuse_parity.json',r)
 for expert in [0,.25,.5,.75,1,1.5]:
  for trunk in [.25,.5,1]:
   key=f'{expert}/{trunk}';oldkey=f'budget200_expert{expert}' if trunk==1 else f'budget200_expert{expert}_trunk{trunk}'
   if oldkey in old:results[key]=old[oldkey];sources[key]='v11:'+oldkey
   else:results[key]=[rollout(Environment(data,cfg,seed,'dev'),Baseline(expert,trunk),50)[0] for seed in roots];sources[key]='new'
   save(out/'results.json',results);save(out/'sources.json',sources);print(key,score(results[key]),sources[key],flush=True)
 best=min(results,key=lambda k:score(results[k]));expert,trunk=map(float,best.split('/'));save(out/'selection.json',{'expert':expert,'trunk':trunk,'score':score(results[best]),'feasible':bool(-score(results[best])[0]>=.9),'seconds':time.perf_counter()-start,'new_episodes':sum(len(v) for k,v in results.items() if sources[k]=='new'),'new_updates':sum(r['cost']['updates'] for k,v in results.items() if sources[k]=='new' for r in v),'parity_updates':r['cost']['updates']})
if __name__=='__main__':main()

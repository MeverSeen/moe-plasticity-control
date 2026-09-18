"""Three isolated, single-thread CPU subprocesses; no shared learning RNG."""
import argparse,concurrent.futures,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
 jobs=[(arm,seed) for seed in plan['controller_seeds'] for arm in plan['arms']]
 def run(job):
  arm,seed=job;name=f'{arm}_{seed}';start=time.time()
  with (a.out/(name+'.log')).open('w') as f:
   r=subprocess.run([sys.executable,str(Path(__file__).with_name('learn.py')),'--plan',str(a.plan),'--arm',arm,'--seed',str(seed),'--out',str(a.out/name)],stdout=f,stderr=subprocess.STDOUT)
  return {'job':name,'exit_code':r.returncode,'wall_seconds':time.time()-start}
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
   result=future.result();results.append(result);(a.out/'status.json').write_text(json.dumps(results,indent=2));print(result,flush=True)
 if any(r['exit_code'] for r in results):raise SystemExit(1)
if __name__=='__main__':main()

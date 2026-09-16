"""Reproduce the locked design in a fresh directory; never overwrite results."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=Path('results/reproduction'))
p.add_argument('--iterations',type=int,default=60)
p.add_argument('--episodes',type=int,default=12)
a=p.parse_args()
root=Path(__file__).resolve().parent
out=a.out.resolve()
if out.exists() and any(out.iterdir()):
    raise SystemExit('Use a fresh directory. Existing results will not be overwritten.')
out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(root/'experiment.py'),'run','--out',str(out/'data'),
    '--iterations',str(a.iterations),'--episodes',str(a.episodes)],check=True)
subprocess.run([sys.executable,str(root/'analyze.py'),'--run',str(out/'data'),
    '--out',str(out/'analysis')],check=True)
print(f'Results: {out}')

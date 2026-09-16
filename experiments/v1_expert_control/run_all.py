"""Reproduce the complete experiment into a fresh directory."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=Path('results/reproduction'))
a=p.parse_args()
root=Path(__file__).resolve().parent
out=a.out.resolve()
if out.exists() and any(out.iterdir()):
    raise SystemExit('Choose an empty output directory; existing results will not be overwritten.')
out.mkdir(parents=True,exist_ok=True)
for stage in ['tune','train','evaluate']:
    subprocess.run([sys.executable,str(root/'moe_control.py'),stage,'--out',str(out/'data')],check=True)
subprocess.run([sys.executable,str(root/'summarize.py'),'--run',str(out/'data'),
                '--out',str(out/'analysis')],check=True)
print(f'Results: {out}')

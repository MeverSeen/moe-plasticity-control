"""Archive exact numerical outputs; weights and data remain under ignored runs/."""
import gzip,hashlib,json
from pathlib import Path

def main():
 root=Path(__file__).parent/'results'
 for source,name in [('runs/v11-diagnostics','diagnostics'),('runs/v11-calibration2','calibration'),('runs/v11-baselines','baselines'),('runs/v11-learning','learning')]:
  source=Path(source);dest=root/name;dest.mkdir(parents=True,exist_ok=False)
  for p in sorted(p for p in source.rglob('*') if p.is_file() and p.suffix in ['.json','.log']):
   rel=p.relative_to(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes()
   if len(data)>100000:target=target.with_suffix('.json.gz');data=gzip.compress(data,compresslevel=9,mtime=0)
   target.write_bytes(data)
  manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()};(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(name,len(manifest),'files',sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'bytes')
if __name__=='__main__':main()

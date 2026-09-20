import gzip,hashlib,json
from pathlib import Path
def main():
 out=Path(__file__).parent/'results'
 for name in ['calibration','smoke','baselines','learning']:
  source=Path('runs/v12-'+name);dest=out/name;dest.mkdir(parents=True,exist_ok=False)
  for p in sorted(source.rglob('*')):
   if not p.is_file() or p.suffix not in ['.json','.log']:continue
   target=dest/p.relative_to(source);target.parent.mkdir(exist_ok=True,parents=True);data=p.read_bytes()
   if len(data)>100000:target=target.with_suffix(target.suffix+'.gz');data=gzip.compress(data,mtime=0)
   target.write_bytes(data)
  manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()};(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(name,len(manifest),sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))
if __name__=='__main__':main()

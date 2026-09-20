"""Fetch pinned official evaluation files only; reject changed upstream bytes."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path('runs/v14-general-raw');ROOT.mkdir(parents=True,exist_ok=True)
lock=json.loads(Path(__file__).with_name('general_sources_lock.json').read_text())
def get(record):
 path=ROOT/record['path']
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']:return record
 request=urllib.request.Request(record['url'],headers={'User-Agent':'moe-plasticity-research/1.0'})
 with urllib.request.urlopen(request,timeout=90) as response:payload=response.read(200_000_001)
 assert len(payload)==record['bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256'],record['path']
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(payload);temporary.replace(path);return record
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 futures={pool.submit(get,r):r['path'] for r in lock['files']}
 for future in concurrent.futures.as_completed(futures):
  record=future.result();records.append(record);print('VERIFIED',record['path'],record['bytes'],flush=True)
(ROOT/'manifest.json').write_text(json.dumps({'fetched_epoch':time.time(),'bbh_revision':lock['bbh_revision'],'sources':records,'purpose':'Independent general evaluation candidates only; not training, rewards or selection; selection/audit pending'},indent=2)+'\n')

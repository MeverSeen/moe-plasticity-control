"""Fetch only official public evaluation sources; pin content hashes, no model queries."""
import concurrent.futures,hashlib,json,tarfile,time,urllib.request
from pathlib import Path
ROOT=Path('runs/v14-general-raw');ROOT.mkdir(parents=True,exist_ok=True)
def get(url,path):
 request=urllib.request.Request(url,headers={'User-Agent':'moe-plasticity-research/1.0'})
 with urllib.request.urlopen(request,timeout=90) as response:
  payload=response.read(200_000_001)
 assert len(payload)<=200_000_000
 path.write_bytes(payload);return {'url':url,'path':path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
req=urllib.request.Request('https://api.github.com/repos/suzgunmirac/BIG-Bench-Hard/commits/main',headers={'User-Agent':'moe-plasticity-research/1.0'})
with urllib.request.urlopen(req,timeout=30) as response:revision=json.load(response)['sha']
urls=[('mmlu.tar','https://people.eecs.berkeley.edu/~hendrycks/data.tar'),('piqa-valid.jsonl','https://yonatanbisk.com/piqa/data/valid.jsonl'),('piqa-labels.lst','https://yonatanbisk.com/piqa/data/valid-labels.lst'),('boolq-dev.jsonl','https://storage.googleapis.com/boolq/dev.jsonl'),('bbh-logical-five.json',f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{revision}/bbh/logical_deduction_five_objects.json')]
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 futures={pool.submit(get,url,ROOT/name):name for name,url in urls}
 for future in concurrent.futures.as_completed(futures):
  try:record=future.result();records.append(record);print('FETCHED',record['path'],record['bytes'],flush=True)
  except Exception as e:records.append({'path':futures[future],'error':str(e)});print('FAILED',futures[future],str(e),flush=True)
(ROOT/'manifest.json').write_text(json.dumps({'fetched_epoch':time.time(),'bbh_revision':revision,'sources':records,'purpose':'Independent general evaluation candidates only; not training, rewards or selection; selection/audit pending'},indent=2)+'\n')

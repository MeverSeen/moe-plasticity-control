"""Download verified public shards through official resolved URLs; bounded retries."""
import hashlib,json,os,time,urllib.request
from pathlib import Path
root=Path(os.environ['V13_ALLOCATION_ROOT']);folder=root/'model';manifest=json.loads((folder/'download_manifest.json').read_text())
for f in manifest['weights']:
 target=folder/f['name'];partial=target.with_suffix('.partial');start=time.monotonic();digest=hashlib.sha256();total=0
 print('START',f['name'],f['size'],flush=True)
 try:
  with urllib.request.urlopen(f['url'],timeout=30) as response,partial.open('wb') as out:
   while True:
    if time.time()>float(os.environ['V13_DEADLINE_EPOCH'])-600:raise TimeoutError('Allocation reserve reached')
    b=response.read(8*1024*1024)
    if not b:break
    out.write(b);digest.update(b);total+=len(b)
    if total%(256*1024*1024)==0:print('PROGRESS',f['name'],total,round(total/(time.monotonic()-start)/1e6,2),'MB/s',flush=True)
  assert total==f['size'] and digest.hexdigest()==f['sha256']
  partial.replace(target);print('VERIFIED',f['name'],total,round(time.monotonic()-start,2),flush=True)
 except Exception as e:
  print('FAILED',f['name'],type(e).__name__,total,flush=True);raise SystemExit(2)
public={'repo':'allenai/OLMoE-1B-7B-0924','revision':manifest['revision'],'weights':[{k:v for k,v in f.items() if k!='url'} for f in manifest['weights']]}
(root/'model_verified.json').write_text(json.dumps(public,indent=2)+'\n')
print('MODEL_COMPLETE',flush=True)

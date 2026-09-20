import json,time,urllib.request
from pathlib import Path
ROOT=Path('/root/autodl-tmp/moe-plasticity-v13-20260920')
def get(url,path=None):
 start=time.monotonic()
 try:
  with urllib.request.urlopen(url,timeout=25) as r:data=r.read()
  if path:path.write_bytes(data)
  print(json.dumps({'host':urllib.request.urlparse(url).hostname if False else url.split('/')[2],'ok':True,'bytes':len(data),'seconds':time.monotonic()-start}),flush=True)
  return json.loads(data)
 except Exception as e:
  print(json.dumps({'host':url.split('/')[2],'ok':False,'error_type':type(e).__name__,'seconds':time.monotonic()-start}),flush=True);return None
info=get('https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0924?blobs=true',ROOT/'model_info.json')
if info:
 print('MODEL_SHA',info['sha'],flush=True)
 files=[f for f in info['siblings'] if f['rfilename'].endswith('.safetensors')];print('SHARDS',[(f['rfilename'],f.get('size')) for f in files],flush=True)
 get('https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/'+info['sha']+'/config.json',ROOT/'model_config.json')
 if files:
  req=urllib.request.Request('https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/'+info['sha']+'/'+files[0]['rfilename'],headers={'Range':'bytes=0-1023'})
  try:
   with urllib.request.urlopen(req,timeout=30) as r:print('SHARD_RANGE',r.status,len(r.read(1024)),flush=True)
  except Exception as e:print('SHARD_RANGE_FAILED',type(e).__name__,flush=True)
get('https://api.github.com/repos/allenai/natural-instructions/commits/master',ROOT/'data_commit.json')

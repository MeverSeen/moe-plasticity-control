"""Verify downloaded deployment factors and compare content, excluding zip filenames."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--adapters',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[];protocol=hashlib.sha256((a.archive/'protocol.json').read_bytes()).hexdigest();method=hashlib.sha256((a.archive/'method_protocol.json').read_bytes()).hexdigest();shape_cycle=[(4,2048),(1024,4),(4,2048),(1024,4),(4,1024),(2048,4)]
 for path in sorted((a.archive/'results/confirmation').glob('*.json')):
  if path.stem.endswith('__general') or path.stem.endswith('__incomplete'):continue
  row=json.loads(path.read_text());ref=row['checkpoint'].get('adapter')
  if ref is None:continue
  file=a.adapters/ref['path'];assert file.stat().st_size==ref['bytes'];sha=hashlib.sha256(file.read_bytes()).hexdigest();assert sha==ref['sha256'];state=torch.load(file,map_location='cpu',weights_only=True);assert state['protocol_sha256']==protocol and state['method_protocol_sha256']==method;w=state['A_B'];assert len(w)==6144 and sum(x.numel() for x in w)==37748736;h=hashlib.sha256()
  for i,x in enumerate(w):assert tuple(x.shape)==shape_cycle[i%6] and x.dtype==torch.float32 and torch.isfinite(x).all();h.update(x.contiguous().numpy().tobytes())
  records.append({'root':row['root']['id'],'method':row['candidate']['name'],'local_path':str(file),'sha256':sha,'factor_content_sha256':h.hexdigest(),'bytes':file.stat().st_size,'factors':len(w),'parameters':sum(x.numel() for x in w)})
 result={'status':'passed','format':'adapter-only; optimizer state absent; not exact training resume','files':records};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

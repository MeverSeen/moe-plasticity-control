"""Verify compact v10 archives, data boundaries and fixed-baseline metrics."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'experiments/v10_visual_control/results'

def read(path):
 if path.exists():return json.loads(path.read_text())
 return json.loads(gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()))

def main():
 count=0
 for folder in ROOT.iterdir():
  if not (folder/'manifest.json').exists():continue
  manifest=read(folder/'manifest.json')
  for name,digest in manifest.items():
   assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,(folder,name)
  if (folder/'locked_plan.json').exists():
   plan=read(folder/'locked_plan.json');protocol={'heldout_roots':plan['roots'],'config':plan['config']};test=read(folder/'results.json')
  else:
   protocol=read(folder/'protocol.json');lock=read(folder/'locked.json')
   assert hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
   test=read(folder/'test.json')
  for method,rows in test.items():
   assert [r['seed'] for r in rows]==protocol['heldout_roots']
   for r in rows:
    assert set(sum(r['classes'],[])).issubset(set(range(80,100)))
    m=np.asarray(r['matrix']);a=np.asarray(r['accuracy_matrix']);assert m.shape==(4,4)==a.shape
    f=np.maximum(m[-1,:-1]-np.diag(m)[:-1],0).mean()
    assert abs(f-r['forgetting'])<1e-9
    assert r['success']==all(t['success'] for t in r['tasks'])
    for t in r['tasks']:
     assert t['success']==(t['loss']<=protocol['config']['target'])
     assert t['steps']<=protocol['config']['steps']
    for t in r['policy_trace']:
     assert np.isfinite(t['behavior_logp'])
   count+=len(rows)
  assert not list(folder.glob('*.pt')),'Weights must not enter Git archive'
 print(f'Verified v10 manifests and {count} heldout trajectories.')
if __name__=='__main__':main()

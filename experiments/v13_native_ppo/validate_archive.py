"""Validate archived actual pilot endpoints, not the cancelled workflow draft."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'provenance/pilot-executed'))
from sparse_policy import Rule,shaped_rewards

def main():
 pilot=ROOT/'archive/results/pilot';rows=[];decisions=0
 for p in sorted(pilot.glob('*.json')):
  x=json.loads(p.read_text())
  if not x.get('candidate'):continue
  assert x['phase']=='pilot' and x['history_completed'];assert x['cost_path']['updates']==sum(t['steps'] for t in x['tasks']);assert sum(t['duration'] for t in x['trace'])==sum(t['steps'] for t in x['tasks'][1:]);assert x['n_fail']==sum(t['success'] is False for t in x['tasks'])
  assert x['success'] is None if x['reference_calibration'] else x['success']==all(t['success'] for t in x['tasks'])
  matrix=x['evaluation']['matrix'];signed=[matrix[-1][j]-matrix[j][j] for j in range(2)];assert np.isclose(np.maximum(signed,0).mean(),x['evaluation']['forgetting']);assert np.isclose(np.mean(signed),x['evaluation']['signed_forgetting']);assert np.isclose(np.mean(matrix[-1]),x['evaluation']['final_loss'])
  name=x['candidate']['name'];rule=Rule(x['candidate']['base_indices'],mode='rule' if name=='provisional_rule1' else 'fixed')
  for t in x['trace']:
   obs=tuple(torch.tensor(v) for v in t['observation']);latent=[torch.tensor(v) for v in t['latent']];m,z,lp,entropy,_=rule.act(obs,latent);assert torch.equal(m,torch.tensor(t['experts']));assert abs(float(lp)-t['policy_logp'])<1e-5;assert 0<t['duration']<=5;decisions+=1
  reward=x['reward'];rr=shaped_rewards([t['potential'] for t in x['trace']],x['monitor']['forgetting'],x['n_fail'],reward['scale'],reward['kappa']);assert np.allclose(rr,[t['reward'] for t in x['trace']],atol=1e-10);assert abs(reward['telescope_error'])<1e-8
  if not x['reference_calibration']:rows.append(x)
 assert len(rows)==3 and len({x['checkpoint']['prefix_sha256'] for x in rows})==1
 prefix=json.loads((pilot/'dev_1301000__prefix.json').read_text());summary=json.loads((pilot/'summary.json').read_text());smoke=json.loads((ROOT/'archive/smoke_result.json').read_text());assert summary['native_cost']['updates']==120+prefix['cost']['updates']+sum(x['cost_actual']['updates'] for x in rows)==167;assert smoke['cost']['updates']==4
 for x in rows:
  for key,value in prefix['cost'].items():assert np.isclose(x['cost_path'][key],x['cost_actual'][key]+value)
 manifest=ROOT/'archive_manifest.json';count=0
 if manifest.exists():
  for name,sha in json.loads(manifest.read_text())['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name;count+=1
 print(json.dumps({'actual_histories':4,'paired_candidates':3,'replayed_decisions':decisions,'actual_updates_including_smoke':171,'manifest_files':count,'status':'passed'}))
if __name__=='__main__':main()

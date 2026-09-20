"""Replay saved evaluation latent decisions against their exact model checkpoint."""
import json,time
from pathlib import Path
import torch
from policy import Policy
from calibrate import save
def main():
 torch.set_num_threads(1);plan=json.loads(Path(__file__).with_name('plan.json').read_text());root=Path('runs/v12-learning');count=0;maxerr=0.;start=time.perf_counter()
 for arm,opts in plan['arms'].items():
  for seed in plan['controller_seeds']:
   folder=root/f'{arm}_{seed}';p=Policy(plan['base']['expert'],plan['base']['trunk'],opts['joint'])
   for stage in ['development','audit']:
    allrows=json.loads((folder/(stage+'.json')).read_text())
    for label,modes in allrows.items():
     p.load_state_dict(torch.load(folder/(label+'.pt'),weights_only=True))
     with torch.no_grad():
      for rows in modes.values():
       for r in rows:
        for t in r['policy_trace']:
         obs=tuple(torch.tensor(v) for v in t['observation']);z=[torch.tensor(v) for v in t['latent']];(m,trunk),_,lp,_,_=p.act(obs,z);err=abs(float(lp)-t['behavior_logp']);maxerr=max(maxerr,err);assert err<1e-5;assert m.tolist()==t['experts'] and trunk==t['trunk'];count+=1
 save(root/'probability_validation.json',{'checked_decisions':count,'max_logprob_error':maxerr,'seconds':time.perf_counter()-start,'scope':'all development/audit saved latent decisions, including repeated holds only once per actual decision; exact matching action and model checkpoint'})
 print(count,'decisions, max logprob error',maxerr)
if __name__=='__main__':main()

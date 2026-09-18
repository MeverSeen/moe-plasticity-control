"""Read-only distribution/greedy-action diagnostics on saved training states."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/v10_visual_control'))
from control import Policy

def main(out):
 torch.set_num_threads(1);protocol=json.loads((out/'protocol.json').read_text());records={}
 for train in out.glob('train_*.json'):
  key=train.stem[6:];variant=key.rsplit('_',1)[0];algo,local,progress,interface=protocol['variants'][variant]
  epochs=sorted([int(p.stem.split('epoch')[-1]) for p in out.glob(key+'_epoch*.pt')])
  if len(epochs)<2:continue
  policies=[]
  for epoch in [epochs[0],epochs[-1]]:
   p=Policy(local,progress,interface);p.load_state_dict(torch.load(out/f'{key}_epoch{epoch}.pt',weights_only=True));policies.append(p)
  history=json.loads(train.read_text());traces=[t for b in history for r in b['results'] for t in r['policy_trace']];traces=traces[::max(len(traces)//128,1)][:128]
  diffs=[];lpd=[]
  with torch.no_grad():
   for t in traces:
    obs=tuple(torch.tensor(x) for x in t['observation']);z=[torch.tensor(x) for x in t['latent']]
    actions=[p.act(obs,deterministic=True)[0] for p in policies];diffs.append(float((actions[0]!=actions[1]).float().mean()))
    probs=[float(p.act(obs,z)[2]) for p in policies];lpd.append(probs[1]-probs[0])
  records[key]={'epoch_pair':[epochs[0],epochs[-1]],'visited_state_sample':len(traces),'greedy_expert_action_change_fraction':float(np.mean(diffs)),'mean_abs_joint_logp_change':float(np.mean(np.abs(lpd))),'scope':'behavior distribution diagnostics only; no heldout evaluation used'}
 (out/'training_diagnostics.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args();main(a.out)

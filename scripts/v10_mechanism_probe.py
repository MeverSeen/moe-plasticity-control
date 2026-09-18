"""Finite-difference expert interaction and routing probe, development data only."""
import copy,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/v10_visual_control'))
from core import Config,Data,Environment

def main():
 torch.set_num_threads(1);data=Data('.cache/datasets/cifar-complete.tar.gz');records=[]
 for routing in ['fixed','dynamic']:
  for seed in [1000100,1000101]:
   cfg=Config(routing=routing);e=Environment(data,cfg,seed,'train');start=time.perf_counter()
   while e.k==0:
    ob=e.observation()
    if ob[1][0]<=cfg.target or e.t>=cfg.steps:e.finish(float(ob[1][0]))
    else:e.update(torch.ones(2,4))
   base=copy.deepcopy(e.model.state_dict());mem=e.memory[0]['monitor'];old,gacc,gate_before=e.measure(0,mem);oldloss=float(old.mean())
   cfg.interval=1;e.update(torch.ones(2,4));full=copy.deepcopy(e.model.state_dict());_,_,gate_after=e.measure(0,mem)
   deltas={k:full[k]-v for k,v in base.items() if k.startswith('blocks.')}
   def loss(experts,nonexpert=False):
    state=copy.deepcopy(base)
    for k,d in deltas.items():
     _,l,i,*rest=k.split('.')
     if (int(l),int(i)) in experts:state[k]+=d
    if nonexpert:
     for k in state:
      if not k.startswith('blocks.'):state[k]=full[k]
    e.model.load_state_dict(state);ll,_,_=e.measure(0,mem);return float(ll.mean())
   experts=[(l,i) for l in range(2) for i in range(4)];single={str(ex):loss([ex])-oldloss for ex in experts};pairs=[]
   for a,ex in enumerate(experts):
    for ey in experts[a+1:]:
     combined=loss([ex,ey])-oldloss;pairs.append({'a':ex,'b':ey,'interaction':combined-single[str(ex)]-single[str(ey)],'combined_delta':combined})
   records.append({'routing':routing,'seed':seed,'old_loss':oldloss,'single_delta':single,'pairs':pairs,'front_norm_head_only_delta':loss([],True)-oldloss,
    'old_gate_l1_drift':float((gate_after-gate_before).abs().mean()),'old_topk_changed_fraction':float(((gate_after>0)!=(gate_before>0)).any(-1).float().mean()),'seconds':time.perf_counter()-start,'observation_examples':e.cost['observation_examples'],
    'scope':'one AdamW candidate step, no controller input; local nonadditivity does not prove RL necessary'})
   print(routing,seed,'max interaction',max(abs(x['interaction']) for x in pairs),'gate drift',records[-1]['old_gate_l1_drift'],flush=True)
 Path('experiments/v10_visual_control/results/interaction_probe.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()

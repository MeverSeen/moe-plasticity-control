"""Read-only box-bound certificate for the frozen actor's greedy action.

The shared combine network ends in tanh, so every layer-mean input to base/count
heads lies in [-1,1]. Use a slightly wider bound for floating-point averaging.
This does not change the registered deployment decoder or reuse any GPU result.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from structured_policy import ActorCritic

def main():
 p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();state=torch.load(a.controller,map_location='cpu',weights_only=False);actor=ActorCritic();actor.load_state_dict(state['policy']);assert isinstance(actor.combine[-1],torch.nn.Tanh);bound=1.0001
 w=actor.base_head.weight.detach().double();b=actor.base_head.bias.detach().double();lb=actor.layer_bias.detach().double();base=[[float(b[4]-b[j]+lb[l,4]-lb[l,j]-bound*(w[4]-w[j]).abs().sum()) for j in range(4)] for l in range(16)]
 w=actor.count_head.weight.detach().double();b=actor.count_head.bias.detach().double();count=[float(b[0]-b[j]+w[0,36]-w[j,36]-bound*(w[0,:32]-w[j,:32]).abs().sum()) for j in [1,2]]
 result={'controller_sha256':hashlib.sha256(a.controller.read_bytes()).hexdigest(),'architecture_source_sha256':hashlib.sha256(Path(__file__).with_name('structured_policy.py').read_bytes()).hexdigest(),'scope':'all observations with finite actor activations and fixed16x64x10 interface; greedy decoder only','layer_mean_absolute_bound':bound,'base1_minus_other_logit_lower_bounds':base,'count0_minus_other_logit_lower_bounds_given_base1':count,'minimum_base_margin':min(min(x) for x in base),'minimum_count_margin':min(count),'greedy_action_identically_uniform1':min(min(x) for x in base)>0 and min(count)>0,'reason':'Positive box-bound logit margins force base index4 and count0 in every layer. Uniform epsilon mixing over the same valid alphabet preserves argmax. No expert correction slots are then drawn. This says nothing about stochastic actions or usefulness of a differently trained controller.','new_gpu_work':False,'workflow_modified':False}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if 'lower_bounds' not in k}))
if __name__=='__main__':main()

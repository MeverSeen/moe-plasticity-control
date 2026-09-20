"""Read-only independent evaluation of a saved, adapter-enabled deployment model.

Adapter-only files do not contain Adam state and must not be used as training
resume checkpoints. NativeLearner.snapshot/restore is the training-resume path.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from evaluation import GeneralEvaluator
from lora_native import NativeLearner

@torch.no_grad()
def load_factors(learner,state):
 assert state.get('protocol_sha256')==hashlib.sha256(Path(__file__).with_name('protocol.json').read_bytes()).hexdigest()
 assert state['base_revision']=='6d84c48581ece794365f2b8e9cfb043c68ade9c5';weights=state['A_B'];assert len(weights)==len(learner.parameters)
 for p,value in zip(learner.parameters,weights):assert value.shape==p.shape and value.dtype==torch.float32 and torch.isfinite(value).all()
 for p,value in zip(learner.parameters,weights):p.copy_(value)
 assert all(a.enabled for a in learner.adapters)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--adapter',type=Path,required=True);parser.add_argument('--examples',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists();learner=NativeLearner(str(args.model));load_factors(learner,torch.load(args.adapter,map_location='cpu',weights_only=True));evaluator=GeneralEvaluator(json.loads(args.examples.read_text()),json.loads(args.manifest.read_text()));result=evaluator.run(learner);args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

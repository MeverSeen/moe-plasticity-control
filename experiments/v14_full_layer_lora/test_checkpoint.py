import unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from evaluate_checkpoint import load_factors
class Tests(unittest.TestCase):
 def test_deployment_loading_checks_every_shape_before_mutation(self):
  learner=SimpleNamespace(parameters=[torch.nn.Parameter(torch.zeros(2)),torch.nn.Parameter(torch.zeros(3))],adapters=[SimpleNamespace(enabled=True)]);state={'protocol_sha256':hashlib.sha256(Path(__file__).with_name('protocol.json').read_bytes()).hexdigest(),'base_revision':'6d84c48581ece794365f2b8e9cfb043c68ade9c5','A_B':[torch.ones(2),torch.ones(4)]}
  with self.assertRaises(AssertionError):load_factors(learner,state)
  self.assertEqual(float(learner.parameters[0].detach().sum()),0.)
  state['A_B'][1]=torch.ones(3);load_factors(learner,state);self.assertTrue(all(torch.equal(p,torch.ones_like(p)) for p in learner.parameters))
if __name__=='__main__':unittest.main()

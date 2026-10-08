"""Cross-module invariants that protect the actual experiment entry point."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from run_native import (configure_trainable_experts, fake_blocks, fixed_probe_pools,
                        load_model, pure_lm_loss, run_root, seed_all, validate_protocol)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())
        torch.set_num_threads(1)

    def test_protocol_rejects_invalid_or_ignored_settings(self):
        validate_protocol(self.protocol)
        for key, value in [('roots', [1, 1]), ('learning_rate', float('nan')),
                           ('checkpoint_steps', [3]), ('replay_blocks', 1),
                           ('actions', [0, 1]), ('gradient_accumulation', 0)]:
            changed = copy.deepcopy(self.protocol)
            changed['development'][key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_protocol(changed)

    def test_probe_pool_is_fixed_and_covers_sources(self):
        cfg = self.protocol['development']
        cfg.update(monitor_blocks_per_domain=3, selection_blocks_per_domain=3)
        blocks = fake_blocks(2, 8)
        pools, manifest = fixed_probe_pools(blocks, cfg)
        self.assertEqual(manifest, fixed_probe_pools(blocks, cfg)[1])
        self.assertEqual(len({b['source'] for b in pools['A']['monitor']}), 3)
        self.assertEqual(len(blocks['A']['monitor']), 12)
        self.assertNotIn('test', pools['A'])

    def test_pure_loss_masks_padding_and_excludes_auxiliary(self):
        logits = torch.zeros(1, 4, 5, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, -100]])
        value, tokens = pure_lm_loss(logits, labels)
        self.assertEqual(tokens, 2)
        self.assertAlmostEqual(float(value.detach()), 2 * float(torch.log(torch.tensor(5.))), places=6)
        value.backward()
        self.assertEqual(float(logits.grad[:, 2:].abs().sum()), 0.)

    def test_full_branch_grid_and_zero_displacement(self):
        cfg = self.protocol['development']
        cfg.update(max_steps=2, evaluation_interval=1, sequence_length=8,
                   micro_batch=1, gradient_accumulation=1, checkpoint_steps=[0, 1],
                   branch_steps=1, monitor_blocks_per_domain=1, selection_blocks_per_domain=1)
        self.protocol['model'].update(layers=[0], experts=[0, 1])
        seed_all(71)
        model, _ = load_model(self.protocol, True, 'cpu', False)
        configure_trainable_experts(model, [0], [0, 1])
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        blocks, _ = fixed_probe_pools(fake_blocks(7, 8), cfg)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_root(model, blocks, self.protocol, 71, 'branches', Path(tmp))
            branches = json.loads((Path(tmp) / 'branches-71.json').read_text())
            self.assertEqual(summary['baseline_trajectories'], 6)
            self.assertEqual(summary['branches'], 20)
            self.assertEqual(summary['cost']['learner_updates'], 32)
            for step in [0, 1]:
                for future in [0, 1]:
                    paired = [b for b in branches if b['checkpoint_step'] == step and b['future'] == future]
                    self.assertEqual(len({b['updates'][0]['batch_digest'] for b in paired}), 1)
            for branch in branches:
                self.assertNotIn('test', branch['endpoint'])
                if branch['action'] == 0:
                    self.assertEqual(branch['updates'][0]['displacement']['model_displacement_l2'], 0.)
                    self.assertAlmostEqual(branch['A_selection_deterioration'], 0., places=7)
                    self.assertAlmostEqual(branch['B_selection_improvement'], 0., places=7)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter), name)


if __name__ == '__main__':
    unittest.main()

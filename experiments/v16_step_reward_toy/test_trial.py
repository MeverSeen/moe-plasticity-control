import copy
import unittest
import numpy as np
import torch
from run import old, rollout


class TestTrial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(old.CONFIG, base_steps=3, max_steps=8, train_n=32,
                          monitor_n=16, test_n=32, replay_memory=8, monitor_memory=8,
                          batch=8, old_batch=8, target=1e-10, interval=1)
        cls.item = old.make_bank([1690100], cls.config, test=True)[0]

    def policy(self):
        torch.manual_seed(11)
        p = old.Policy()
        with torch.no_grad(): p.actor.bias[2] = 1.
        return p

    def test_same_fixed_action_same_path_across_hold(self):
        a, _, wa = rollout(self.item, self.config, 1, 'movement', .1, action=2, keep_trace=True)
        b, _, wb = rollout(self.item, self.config, 5, 'movement', .1, action=2, keep_trace=True)
        self.assertTrue(torch.equal(wa, wb))
        for key in ('steps', 'path_energy', 'net_displacement_sq', 'new_monitor_loss', 'learner_monitor_work'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['decisions'], 8)
        self.assertEqual(b['decisions'], 2)

    def test_actual_step_energy_and_return_telescope(self):
        for h in (1, 5):
            for reward in ('movement', 'forgetting'):
                row, tr, w = rollout(self.item, self.config, h, reward, .1, self.policy(), keep_trace=True)
                self.assertAlmostEqual(sum(t['reward'] for t in tr), row['terminal_reward'], places=10)
                self.assertAlmostEqual(sum(t['actual_update_sq'] for t in row['events']), row['path_energy'], places=12)
                expected = float((w.double() - self.item[2].double()).square().sum())
                self.assertAlmostEqual(expected, row['net_displacement_sq'], places=12)
                self.assertEqual(sum(t['duration'] for t in tr), row['steps'])
                self.assertFalse(row['success'])
                if reward == 'movement':
                    self.assertAlmostEqual(row['terminal_reward'], -row['path_energy']/.1 - 10., places=10)

    def test_test_labels_never_enter_control_or_reward(self):
        changed = copy.deepcopy(self.item)
        for task in changed[0]['tasks']:
            task['test'] = (task['test'][0], task['test'][1] + 100)
        a, ta, wa = rollout(self.item, self.config, 1, 'movement', .1, self.policy())
        b, tb, wb = rollout(changed, self.config, 1, 'movement', .1, self.policy())
        self.assertTrue(torch.equal(wa, wb))
        self.assertEqual(a['terminal_reward'], b['terminal_reward'])
        for x, y in zip(ta, tb):
            self.assertTrue(torch.equal(x['x'], y['x']))
            self.assertEqual(x['reward'], y['reward'])
        self.assertNotEqual(a['test_old_loss'], b['test_old_loss'])


if __name__ == '__main__': unittest.main()

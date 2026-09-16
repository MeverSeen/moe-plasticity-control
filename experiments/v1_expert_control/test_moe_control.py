"""Checks the experiment's validity boundaries, not controller performance."""
import unittest
import torch
import moe_control as m


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = dict(m.DEFAULT)
        cls.world = m.make_world(19, cls.c)
        cls.w, cls.refs = m.pretrain(cls.world, 1, cls.c)

    def env(self, config=None):
        return m.AdaptEnv(self.world, 1, self.w, self.refs, config or self.c)

    def test_importance_matches_exact_jacobian(self):
        x = self.world['tasks'][0]['train'][0][:9]
        w = self.w.clone().requires_grad_()
        expected = torch.zeros_like(w)
        for row in x:
            y = m.predict(w,row[None,:],self.world['router'],self.c).sum()
            expected += torch.autograd.grad(y,w)[0].square()/len(x)
        actual = m.fisher_diagonal(w,x,self.world['router'],self.c)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)

    def test_routes_and_frozen_experts_do_not_move(self):
        e = self.env()
        x = self.world['tasks'][0]['train'][0]
        before = tuple(t.clone() for t in m.routes(x,e.router,self.c))
        original = e.w.detach().clone()
        e.m.fill_(.1)  # Momentum must not defeat an explicit zero displacement.
        e.advance(torch.zeros(self.c['experts'],2))
        torch.testing.assert_close(e.w,original,rtol=0,atol=0)
        for a,b in zip(before,m.routes(x,e.router,self.c)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_policy_is_equivariant_under_expert_reindexing(self):
        torch.manual_seed(44)
        p = m.Controller()
        torch.nn.init.normal_(p.actor[-1].weight,std=.2)
        local = torch.randn(self.c['experts'],12)
        glob = torch.randn(5)
        order = torch.randperm(self.c['experts'])
        d,v = p(local,glob)
        dp,vp = p(local[order],glob)
        torch.testing.assert_close(dp.mean,d.mean[order])
        torch.testing.assert_close(vp,v)

    def test_old_memory_and_reference_metadata_are_bounded(self):
        _,refs = m.pretrain(self.world,4,self.c)
        for count in range(1,5):
            replay,monitor = m.memory(self.world,count,self.c)
            self.assertLessEqual(len(replay[0]),self.c['replay_memory'])
            self.assertLessEqual(sum(len(d[0]) for d in monitor),self.c['monitor_memory'])
        self.assertLessEqual(sum(r.numel() for r in refs),self.c['monitor_memory'])
        self.assertTrue(all(r.ndim==1 for r in refs))

    def test_test_labels_cannot_change_controller_or_training(self):
        other = m.make_world(19,self.c,include_test=True)
        for task in other['tasks']:
            task['test'][1].fill_(1e9)
        a = self.env()
        b = m.AdaptEnv(other,1,self.w,self.refs,self.c)
        for u,v in zip(a.observation(),b.observation()):
            torch.testing.assert_close(u,v,rtol=0,atol=0)
        controls = torch.tensor([[1.,.02]]*self.c['experts'])
        ra = a.advance(controls)
        rb = b.advance(controls)
        self.assertEqual(ra,rb)
        torch.testing.assert_close(a.w,b.w,rtol=0,atol=0)

    def test_reward_shaping_preserves_terminal_objective(self):
        c = dict(self.c,max_steps=11,target=1e-8)
        e = self.env(c)
        initial_potential = e.potential()
        total = 0.
        while not e.done:
            r,_,info = e.advance(torch.tensor([[1.,.02]]*c['experts']))
            total += r
        self.assertEqual(e.step_count,11)
        self.assertAlmostEqual(total,info['base_reward']-initial_potential,places=6)


if __name__=='__main__':
    unittest.main(verbosity=2)

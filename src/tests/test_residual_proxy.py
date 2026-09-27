import unittest
import torch
from wmq_residual_proxy import ResidualProxy


class ResidualProxyTests(unittest.TestCase):
    def test_known_direction_and_orthogonal_penalty(self):
        residual = torch.tensor([[[[2., 0., 0., 0.]]]])
        proxy = ResidualProxy([residual, residual*.5], rank=1)
        self.assertAlmostEqual(proxy.capture(residual), 1., places=6)
        self.assertAlmostEqual(float(proxy.loss(residual, residual)), 0., places=6)
        self.assertGreater(float(proxy.loss(torch.zeros_like(residual), residual)), 0)
        damaged = residual.clone()
        damaged[..., 1] = 4
        self.assertGreater(float(proxy.loss(damaged, residual)), 0)

    def test_fit_only_normalization_and_random_control(self):
        torch.manual_seed(3)
        fit = [torch.randn(1, 2, 3, 4) for _ in range(4)]
        learned = ResidualProxy(fit, rank=2)
        random = ResidualProxy(fit, rank=2, random=True, seed=7)
        repeat = ResidualProxy(fit, rank=2, random=True, seed=7)
        torch.testing.assert_close(random.basis, repeat.basis, rtol=0, atol=0)
        torch.testing.assert_close(learned.energy, random.energy, rtol=0, atol=0)
        basis, energy = learned.basis.clone(), learned.energy.clone()
        learned.loss(fit[0]*100, fit[1]*100)
        learned.capture(fit[0]*100)
        torch.testing.assert_close(learned.basis, basis, rtol=0, atol=0)
        torch.testing.assert_close(learned.energy, energy, rtol=0, atol=0)
        self.assertEqual(learned.basis.shape, random.basis.shape)
        torch.testing.assert_close(random.basis @ random.basis.T, torch.eye(2), atol=1e-6, rtol=1e-6)

    def test_zero_residual_is_reported_inactive_and_finite(self):
        z = torch.zeros(1, 2, 3, 4)
        proxy = ResidualProxy([z, z], rank=2)
        self.assertFalse(proxy.diagnostics['active'])
        self.assertEqual(proxy.diagnostics['rank'], 0)
        self.assertTrue(torch.isfinite(proxy.loss(torch.ones_like(z), z)))
        self.assertEqual(float(proxy.loss(z, z)), 0)


if __name__ == '__main__':
    unittest.main()

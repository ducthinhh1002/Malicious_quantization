import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from minhnqn_objectives import signed_sketch, relative_delta_loss, centered_cka, GradientProbe


class ObjectiveTests(unittest.TestCase):
    def test_sketch_signed_linear_and_rng_isolated(self):
        torch.manual_seed(9)
        a, b = torch.randn(200), torch.randn(200)
        state = torch.get_rng_state().clone()
        project = lambda x: signed_sketch(x, width=32, seed=17, chunk=41)
        torch.testing.assert_close(project(a - b), project(a) - project(b))
        torch.testing.assert_close(project(-a), -project(a))
        torch.testing.assert_close(project(2 * a), 2 * project(a))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_delta_loss_rejects_wrong_sign_and_reduced_strength(self):
        clean = torch.tensor([.3, -.2, .8])
        delta = torch.tensor([1., 2., -1.])
        reference = clean + delta
        same, _ = relative_delta_loss(reference, reference, clean)
        weak, _ = relative_delta_loss(clean + .5 * delta, reference, clean)
        flipped, metrics = relative_delta_loss(clean - delta, reference, clean)
        self.assertAlmostEqual(float(same), 0.)
        self.assertAlmostEqual(float(weak), .25)
        self.assertAlmostEqual(float(flipped), 4.)
        self.assertAlmostEqual(metrics['gradient_delta_cosine'], -1., places=6)

    def test_actual_probe_gradient_sketch_hypergradient(self):
        # Exercise real probe implementation with tiny ordinary denoising model.
        from types import SimpleNamespace
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.4))
            def forward(self, x, t, encoder_hidden_states):
                return SimpleNamespace(sample=x * self.weight)
        probe = object.__new__(GradientProbe)
        probe.width = 16
        probe.pipe = SimpleNamespace(unet=Net())
        probe.scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='epsilon'),
                                         add_noise=lambda z, n, t: z + .1 * n)
        params = {'weight': torch.tensor(.4, requires_grad=True)}
        image = torch.tensor([.7, -.1], requires_grad=True)
        noise = torch.tensor([.3, -.5]); text = torch.zeros(1); t = torch.ones(1)
        clean = probe.gradient_sketch(torch.zeros(2), text, params, noise, t, False)
        reference = probe.gradient_sketch(torch.tensor([.9, .3]), text, params, noise, t, False)
        candidate = probe.gradient_sketch(image, text, params, noise, t, True)
        loss, _ = relative_delta_loss(candidate, reference, clean)
        hyper = torch.autograd.grad(loss, image)[0]
        self.assertTrue(torch.isfinite(hyper).all())
        self.assertGreater(float(hyper.norm()), 0.)
        # Finite-difference check for double backward through checkpoint + sketch.
        def objective(value):
            cand = probe.gradient_sketch(value, text, params, noise, t, False)
            return relative_delta_loss(cand, reference, clean)[0]
        for index in range(2):
            d = torch.zeros(2); d[index] = .001
            fd = (objective(image.detach() + d) - objective(image.detach() - d)) / .002
            torch.testing.assert_close(hyper[index], fd, rtol=.003, atol=.003)
        self.assertAlmostEqual(float(probe.pipe.unet.weight.detach()), .4, places=6)

    def test_cka_identity_shift_scale_and_content_change(self):
        torch.manual_seed(7)
        x = torch.randn(24, 8)
        self.assertLess(abs(float(centered_cka(x, x))), 1e-6)
        self.assertLess(abs(float(centered_cka(2*x+3, x))), 1e-6)
        changed = torch.randn(24, 8, requires_grad=True)
        loss = centered_cka(changed, x)
        self.assertGreater(float(loss.detach()), .1)
        grad = torch.autograd.grad(loss, changed)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.)

    def test_cka_degenerate_finite_and_no_reference_grad(self):
        x = torch.zeros(8, 3, requires_grad=True)
        y = torch.ones(8, 3, requires_grad=True)
        loss = centered_cka(x, y)
        grad = torch.autograd.grad(loss, (x, y), allow_unused=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad[0]).all())
        self.assertIsNone(grad[1])
        with self.assertRaises(ValueError):
            centered_cka(x[:1], y[:1])

    def test_equal_budget_and_old_transfer_proxy_disabled(self):
        from minhnqn_campaign import SPECS
        self.assertTrue(all(not s['transfer'] for s in SPECS.values()))
        self.assertEqual(SPECS['quality_gradient']['teacher_steps'], 600)
        self.assertEqual(SPECS['quality_cka']['teacher_steps'], 600)

    def test_queue_routes_downstream_operations_to_isolated_entrypoint(self):
        import minhnqn_campaign as campaign
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            queue = Path(folder); run = queue / 'quality_gradient'; run.mkdir()
            (run / 'smoke.json').write_text(json.dumps({'passed': True, 'method_verified': 'gradient'}))
            def fake_call(run, label, script, *args):
                calls.append((label, Path(script).name, args))
            def fake_job(name):
                campaign.shared.call(run, '02_data', Path('/tmp/transfer_campaign.py'), '--operation', 'data')
                campaign.shared.call(run, '06_train', Path('/tmp/student.py'), '--stage', 'train')
            with patch.object(campaign, 'QUEUE', queue), patch.object(campaign, 'call', fake_call), \
                 patch.object(campaign.shared, 'call', fake_call), patch.object(campaign.shared, 'job', fake_job):
                campaign.job('quality_gradient')
        self.assertEqual(calls[0][:2], ('00_smoke', 'minhnqn_refine.py'))
        self.assertEqual(calls[1][:2], ('01_teacher', 'minhnqn_refine.py'))
        self.assertEqual(calls[2][:2], ('02_data', 'minhnqn_campaign.py'))
        self.assertEqual(calls[3][:2], ('06_train', 'student.py'))


if __name__ == '__main__':
    unittest.main()

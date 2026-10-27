import unittest
import torch
from transfer_proxy import adamw, prediction, rgb8
from transfer_refine import add_bounded_gradient
from transfer_campaign import SPECS, ORDER


class RefinementTest(unittest.TestCase):
    def test_adamw_matches_ordinary_updates(self):
        p = torch.nn.Parameter(torch.tensor([.2, -.3]))
        optimizer = torch.optim.AdamW([p], lr=1e-4)
        values = {'weight': p.detach().clone()}; state = {}
        for step, grad in enumerate([torch.tensor([.4, -.7]), torch.tensor([1.2, -.8])], 1):
            p.grad = grad.clone()
            torch.nn.utils.clip_grad_norm_([p], 1.)
            optimizer.step(); optimizer.zero_grad()
            values, state = adamw(values, [grad], state, step)
            torch.testing.assert_close(values['weight'], p, rtol=1e-6, atol=1e-7)

    def test_zero_gradient_hyperderivative_finite(self):
        x = torch.tensor([0., .2], requires_grad=True)
        updated, _ = adamw({'p': torch.tensor([.2, .3])}, [x], {}, 1)
        derivative = torch.autograd.grad(updated['p'].sum(), x)[0]
        self.assertTrue(torch.isfinite(derivative).all())

    def test_auxiliary_budget(self):
        p = torch.nn.Parameter(torch.ones(2)); p.grad = torch.tensor([3., 4.])
        original = p.grad.clone()
        stats = add_bounded_gradient([p], [torch.tensor([30., 40.])], torch.tensor(5.), .25)
        self.assertAlmostEqual(float((p.grad-original).norm()), 1.25, places=6)
        self.assertAlmostEqual(stats['added_norm'], 1.25, places=6)

    def test_functional_second_order_path_and_no_mutation(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, x, t, encoder_hidden_states):
                from types import SimpleNamespace
                return SimpleNamespace(sample=x*self.weight)
        model = Toy(); initial = model.weight.detach().clone()
        image = torch.tensor([.3], requires_grad=True)
        weight = torch.tensor(.2, requires_grad=True)
        loss = prediction(model, image, torch.tensor(1), torch.zeros(1), {'weight': weight}).square().mean()
        grad = torch.autograd.grad(loss, weight, create_graph=True)[0]
        updated, _ = adamw({'weight': weight}, [grad], {}, 1)
        query = prediction(model, torch.ones(1), torch.tensor(1), torch.zeros(1), updated).sum()
        hyper = torch.autograd.grad(query, image)[0]
        self.assertTrue(torch.isfinite(hyper).all())
        self.assertGreater(float(hyper.abs().sum()), 0.)
        torch.testing.assert_close(model.weight, initial)

    def test_rgb8_forward_and_gradient(self):
        x = torch.tensor([-.32, .43], requires_grad=True)
        result = rgb8(x)
        torch.testing.assert_close(result, ((x+1)*127.5).round()/127.5-1)
        torch.testing.assert_close(torch.autograd.grad(result.sum(), x)[0], torch.ones_like(x))

    def test_ablation_only_adds_transfer(self):
        a, b = SPECS[ORDER[0]], SPECS[ORDER[1]]
        self.assertEqual({k:v for k,v in a.items() if k!='transfer'},
                         {k:v for k,v in b.items() if k!='transfer'})
        self.assertFalse(a['transfer']); self.assertTrue(b['transfer'])


if __name__ == '__main__':
    unittest.main()

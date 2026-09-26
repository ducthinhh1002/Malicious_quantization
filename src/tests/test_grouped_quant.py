import unittest
import torch
from wmq_grouped_quant import GroupedW4, grouped_fake_quant
from wmq_blind import RoundingGrid


class GridTests(unittest.TestCase):
    def test_group_padding_export_and_gradients(self):
        torch.manual_seed(7)
        for shape in ((3, 9), (2, 3, 3, 3)):
            weight = torch.randn(shape)
            grid = GroupedW4(weight, 4)
            torch.testing.assert_close(grid(False), grouped_fake_quant(weight, 4))
            self.assertEqual(grid().shape, weight.shape)
            grid().square().sum().backward()
            self.assertGreater(grid.offset.grad.abs().sum().item(), 0)
            self.assertGreater(grid.log_scale.grad.abs().sum().item(), 0)
            torch.testing.assert_close(grid(), grid(False))

    def test_warm_center_preserves_initial_model_and_scale_gradient(self):
        torch.manual_seed(11)
        grid = RoundingGrid(torch.randn(3, 8), 4, learn_code_offsets=True)
        before = grid(False).detach().clone()
        source = grid.source.clone()
        grid.center_on_warm_codes()
        torch.testing.assert_close(before, grid(False), rtol=0, atol=0)
        grid().square().mean().backward()
        self.assertGreater(grid.log_scale.grad.abs().sum().item(), 0)
        self.assertGreater(grid.code_offset.grad.abs().sum().item(), 0)
        torch.testing.assert_close(source, grid.source, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

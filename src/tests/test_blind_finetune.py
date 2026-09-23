import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn

from wmq_blind import materialize, optimize_branch, parser, scheduled_lr


class FinetuneTests(unittest.TestCase):
    def test_fp32_control_learns_bias_and_restores_source(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
                self.post_quant_conv = nn.Identity()
                nn.init.zeros_(self.decoder.weight)
                nn.init.zeros_(self.decoder.bias)

            def decode(self, z, **kwargs):
                return (self.decoder(z),)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = VAE().to(device).requires_grad_(False)
        z = [torch.ones(1, 3, 16, 16, device=device)]
        natural = [torch.full_like(z[0], .4)]
        refs = [torch.full_like(z[0], .5)]
        args = parser().parse_args(["--model", "x", "--output", "x", "--prompts", "x",
            "--steps", "4", "--eval-every", "1", "--ft-lr", ".001", "--warmup-steps", "1",
            "--natural-perceptual-weight", "0"])
        source = {k: v.clone() for k, v in vae.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = optimize_branch(vae, ["weight", "bias"], z, natural, z, refs,
                args, 32, 1., "natural_full_finetune", [], Path(tmp) / "branch",
                natural_search=(z, natural), preserve_data=(z, refs))
            updates = json.loads((Path(tmp) / "branch/updates.json").read_text())["rows"]
        self.assertLess(selected["selection_objective"], rows[0]["selection_objective"])
        self.assertGreater(selected["step"], 0)
        self.assertEqual(selected["parameter_space"], "unrestricted_fp32_decoder")
        self.assertEqual(selected["gradient_updates"], 4)
        for name, value in vae.state_dict().items():
            self.assertTrue(torch.equal(value, source[name]))
        materialize(vae.decoder, ["weight", "bias"], grids)
        self.assertFalse(torch.equal(vae.decoder.bias, source["decoder.bias"]))
        self.assertLess(updates[-1]["learning_rate"], updates[0]["learning_rate"])

    def test_warmup_decay_and_short_runs(self):
        rates = [scheduled_lr(.001, s, 100, 10) for s in range(1, 101)]
        self.assertAlmostEqual(rates[0], .0001)
        self.assertAlmostEqual(max(rates), .001)
        self.assertAlmostEqual(rates[-1], .0001)
        self.assertGreater(scheduled_lr(.001, 1, 1, 10), 0)


if __name__ == "__main__":
    unittest.main()

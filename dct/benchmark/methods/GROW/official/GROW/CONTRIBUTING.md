# Contributing

Thank you for your interest in GROW! This repository is primarily a
research release, but we welcome bug reports, reproducibility notes and
small, focused pull requests.

## Reporting issues

When opening an issue, please include:

1. A minimal script that reproduces the problem (≤ 30 lines).
2. The exact command used and the full error traceback.
3. Environment details: OS, Python version, `torch.__version__`,
   `diffusers.__version__`, GPU model / driver.
4. Which model checkpoint you were targeting (e.g. `stabilityai/stable-diffusion-2-1-base`).

## Pull requests

* Keep the diff focused on a single concern.
* Match the existing coding style (PEP 8, 4-space indent, type hints on
  public functions, module-level docstrings). Run `python -m compileall grow`
  before submitting.
* Add or update the relevant unit tests / documentation.
* If you change an algorithmic default, please include an ablation table
  or a short justification.

## Code of conduct

Please be respectful and constructive. Discussions should stay focused
on the research and engineering work at hand.

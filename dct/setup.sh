#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec 8>.setup.lock
flock 8
if ! test -x .venv/bin/python; then "${PYTHON_BIN:-python3}" -m venv .venv; fi
stamp="$(sha256sum requirements.txt | cut -d ' ' -f1)-torch271cu128"
if test -f .venv/install.stamp && test "$(cat .venv/install.stamp)" = "$stamp"; then exit 0; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' "$stamp" >.venv/install.stamp

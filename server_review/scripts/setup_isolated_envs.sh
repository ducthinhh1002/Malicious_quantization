#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/serum
# Independent copies of the working stack; do not share writable site-packages.
for name in ctrlregen student; do
  dest="/root/workspace/serum/.venv-${name}"
  if [ ! -d "$dest" ]; then
    cp -a --reflink=auto .venv "$dest"
    python3 -m venv --without-pip "$dest"
    "$dest/bin/python" -m ensurepip --upgrade
  fi
done
.venv-ctrlregen/bin/python -m pip install -r third_party/CtrlRegen/requirements.txt 'huggingface-hub<0.26' 'numpy<2'
.venv-student/bin/python -m pip install -e third_party/diffusers -r third_party/diffusers/examples/text_to_image/requirements.txt
.venv-student/bin/python -m pip install lpips
for name in ctrlregen student; do
  ".venv-${name}/bin/python" -m pip freeze > "manifests/${name}_env.txt"
done
.venv/bin/python -m pip freeze > manifests/serum_env.txt

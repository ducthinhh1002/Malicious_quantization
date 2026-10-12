"""Check environment isolation and CLI routing without GPU/package downloads."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap_env as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_refuses_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'valuable.txt').write_text('keep')
            with self.assertRaisesRegex(RuntimeError, 'non-venv'):
                bootstrap.ensure_environment(root, root / 'cache')
            self.assertEqual((root / 'valuable.txt').read_text(), 'keep')

    def test_refuses_wrong_existing_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pyvenv.cfg').write_text('home = /example')
            with patch.object(bootstrap, 'python_version', return_value='3.12'):
                with self.assertRaisesRegex(RuntimeError, 'requires 3.11'):
                    bootstrap.ensure_environment(root, root / 'cache')

    def test_existing_venv_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pyvenv.cfg').write_text('home = /example')
            with patch.object(bootstrap, 'python_version', return_value='3.11'), \
                 patch.object(bootstrap.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                 patch.object(bootstrap, 'private_uv') as uv:
                self.assertEqual(bootstrap.ensure_environment(root, root / 'cache'), root / 'bin/python')
                uv.assert_not_called()

    def test_pip_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'BITMARK_VENV': str(Path(tmp) / 'venv'), 'BITMARK_BOOTSTRAP_CACHE': str(Path(tmp) / 'cache')}):
            with patch.object(bootstrap, 'ensure_environment', return_value=Path('/private/bin/python')), \
                 patch.object(bootstrap.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')), \
                 patch.object(bootstrap.os, 'execv') as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.main(['--setup-only'])
                execute.assert_not_called()

    def test_plan_flag_anywhere_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, BITMARK_PYTHON=sys.executable, BITMARK_VENV=str(Path(tmp) / 'venv'))
            p = subprocess.run(['bash', str(bootstrap.HERE / 'run.sh'), '--ranks', '4', '16', '--plan-only'],
                               env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('lora_r16_wm', p.stdout)
            self.assertFalse((Path(tmp) / 'venv').exists())


if __name__ == '__main__':
    unittest.main()

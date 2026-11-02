"""SERUM setup regressions; no package installation, downloads, or GPU needed."""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_serum_environment as setup


class SerumEnvironmentTests(unittest.TestCase):
    def test_shared_pins_support_serum_without_flashattention(self):
        pins = setup.pinned_versions()
        self.assertEqual(pins['datasets'], '3.6.0')
        self.assertEqual(pins['pyarrow'], '19.0.1')
        self.assertNotIn('flash-attn', pins)
        text = (setup.HERE/'requirements-bitmark.txt').read_text()
        self.assertIn('-r requirements.txt', text)
        self.assertIn('cp311-cp311-linux_x86_64.whl', text)

    def test_missing_and_wrong_versions_are_reported(self):
        with patch.object(setup.metadata, 'version', side_effect=[
                setup.metadata.PackageNotFoundError('datasets'), '18.0.0']):
            differences = setup.mismatches({'datasets': '3.6.0', 'pyarrow': '19.0.1'})
        self.assertEqual(differences, ['datasets: NOT INSTALLED -> 3.6.0', 'pyarrow: 18.0.0 -> 19.0.1'])

    def test_matching_environment_never_calls_pip(self):
        with patch.object(setup, 'mismatches', return_value=[]), \
             patch.object(setup, 'audit_imports') as audit, \
             patch.object(setup.subprocess, 'run') as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            setup.prepare()
        execute.assert_not_called()
        audit.assert_called_once_with()

    def test_install_targets_selected_python_and_shared_requirements(self):
        with patch.object(setup, 'mismatches', side_effect=[['datasets missing'], []]), \
             patch.object(setup, 'audit_imports') as audit, \
             patch.object(setup.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            setup.prepare()
        install = execute.call_args_list[-1].args[0]
        self.assertEqual(install[:4], [sys.executable, '-m', 'pip', 'install'])
        self.assertEqual(install[-2:], ['-r', str(setup.REQUIREMENTS)])
        self.assertNotIn('requirements-bitmark.txt', ' '.join(install))
        audit.assert_called_once_with()

    def test_check_only_fails_without_mutation(self):
        with patch.object(setup, 'mismatches', return_value=['datasets missing']), \
             patch.object(setup, 'audit_imports') as audit, \
             patch.object(setup.subprocess, 'run') as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Dependencies missing'):
                setup.prepare(check_only=True)
        execute.assert_not_called()
        audit.assert_not_called()

    def test_failed_install_prevents_import_audit(self):
        failed = subprocess.CalledProcessError(1, 'pip')
        with patch.object(setup, 'mismatches', return_value=['datasets missing']), \
             patch.object(setup, 'audit_imports') as audit, \
             patch.object(setup.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0), failed]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.prepare()
        audit.assert_not_called()

    def test_launcher_routes_setup_and_read_only_modes(self):
        bash = shutil.which('bash')
        if sys.platform == 'win32':
            candidate = Path('C:/Program Files/Git/bin/bash.exe')
            bash = str(candidate) if candidate.exists() else None
        if not bash:
            self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='serum launcher ') as tmp:
            root = Path(tmp)
            shutil.copyfile(setup.HERE.parent/'run_serum.sh', root/'run_serum.sh')
            fake = root/'fake-python'
            fake.write_text('#!/usr/bin/env bash\nprintf "PY:%s\\n" "$@"\n'
                            'for arg in "$@"; do\n'
                            '  if [[ "$arg" == *prepare_serum_environment.py ]]; then\n'
                            '    exit "${FAKE_SETUP_EXIT:-0}"\n'
                            '  fi\ndone\n', encoding='utf-8')
            fake.chmod(0o755)
            env = dict(os.environ, SERUM_PYTHON=fake.as_posix(), SERUM_AUTO_INSTALL='1')

            def launch(*args):
                return subprocess.run([bash, (root/'run_serum.sh').as_posix(), *args],
                                      env=env, capture_output=True, text=True)

            normal = launch('--output', 'path with spaces')
            self.assertEqual(normal.returncode, 0, normal.stderr)
            self.assertLess(normal.stdout.index('prepare_serum_environment.py'), normal.stdout.index('run_serum.py'))
            self.assertIn('PY:path with spaces\n', normal.stdout)
            for args in [('--help',), ('--plan-only',), ('--analyze', 'some run')]:
                result = launch(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('prepare_serum_environment.py', result.stdout)
            result = launch('--setup-only')
            self.assertIn('prepare_serum_environment.py', result.stdout)
            self.assertNotIn('/run_serum.py', result.stdout)
            env['SERUM_AUTO_INSTALL'] = '0'
            self.assertIn('PY:--check-only', launch().stdout)
            env['FAKE_SETUP_EXIT'] = '7'
            result = launch()
            self.assertEqual(result.returncode, 7)
            self.assertNotIn('/run_serum.py', result.stdout)


if __name__ == '__main__':
    unittest.main()

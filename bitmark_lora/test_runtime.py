"""Environment routing regressions without installation, downloads, or CUDA."""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import check_runtime
import common
import run


class RuntimeTests(unittest.TestCase):
    def test_import_failure_stops_before_cuda(self):
        args = run.parse_args(['--skip-utility', '--lora-only'])
        with patch.object(run, 'prepare_runtime_storage'), \
             patch.object(run.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1)) as call:
            with self.assertRaisesRegex(RuntimeError, 'cannot import pipeline models'):
                run.preflight(args)
        command = call.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(common.HERE/'check_runtime.py'))
        self.assertIn('--skip-utility', command)
        self.assertIn('--lora-only', command)

    def test_import_audit_reports_failure_and_optional_stages(self):
        with patch.object(common, 'prepare_runtime_storage'), \
             patch.object(check_runtime, 'check_model_imports', side_effect=ImportError('broken T5')), \
             patch.object(check_runtime, 'check_data_imports'), \
             patch.object(check_runtime, 'check_vendor_imports'), \
             patch.object(check_runtime, 'check_trainer_imports') as trainer, \
             patch.object(check_runtime, 'check_utility_imports') as utility, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(check_runtime.main(['--skip-utility', '--lora-only']), 1)
            trainer.assert_called_once_with(True)
            utility.assert_not_called()

    def test_colab_storage_is_local_and_drive_overrides_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'runtime'
            with patch.dict(os.environ, {'COLAB_RELEASE_TAG': 'test', 'BITMARK_RUNTIME_ROOT': str(root)}, clear=True):
                common.prepare_runtime_storage()
                for key in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TMPDIR'):
                    self.assertTrue(Path(os.environ[key]).is_dir())
                    self.assertIn(root.resolve(), Path(os.environ[key]).parents)
                os.environ['HF_HUB_CACHE'] = '/content/drive/MyDrive/cache'
                with self.assertRaisesRegex(ValueError, 'local /content'):
                    common.prepare_runtime_storage()
            tempfile.tempdir = None

    def test_launcher_interpreters_and_argument_forwarding(self):
        bash = shutil.which('bash')
        if sys.platform == 'win32':
            candidate = Path('C:/Program Files/Git/bin/bash.exe')
            bash = str(candidate) if candidate.exists() else None
        if not bash:
            self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='bitmark launcher ') as tmp:
            root = Path(tmp)
            (root/'bitmark_lora').mkdir()
            shutil.copyfile(common.HERE/'run.sh', root/'bitmark_lora/run.sh')
            shutil.copyfile(common.HERE.parent/'run_bitmark_lora.sh', root/'run_bitmark_lora.sh')
            runtime = root/'runtime space'
            private = runtime/'.venv/bin/python'
            local = root/'bitmark_lora/.venv/bin/python'
            override = root/'custom-python'
            fallback = root/'bin/python3'
            for path, marker in ((private, 'runtime'), (local, 'source'), (override, 'override'), (fallback, 'system')):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'#!/usr/bin/env bash\nprintf "{marker}:%s\\n" "$@"\n', encoding='utf-8')
                path.chmod(0o755)
            env = dict(os.environ, BITMARK_RUNTIME_ROOT=runtime.as_posix())
            env.pop('BITMARK_PYTHON', None)
            args = ['--ranks', '4', '16', '--output', 'path with spaces']
            def launch(marker):
                result = subprocess.run([bash, (root/'run_bitmark_lora.sh').as_posix(), *args],
                                        env=env, text=True, capture_output=True, check=True)
                self.assertIn(f'{marker}:-u', result.stdout)
                for arg in args:
                    self.assertIn(f'{marker}:{arg}\n', result.stdout)
            launch('runtime')
            private.unlink()
            launch('source')
            env['BITMARK_PYTHON'] = override.as_posix()
            launch('override')
            env.pop('BITMARK_PYTHON')
            local.unlink()
            # Git Bash uses ':' in PATH even though the outer host is Windows.
            if sys.platform == 'win32':
                probe = subprocess.run([bash, '-c', 'printf "%s" "$PATH"'], text=True,
                                       capture_output=True, check=True)
                env['PATH'] = (root/'bin').as_posix()+':'+probe.stdout
            else:
                env['PATH'] = str(root/'bin')+os.pathsep+env['PATH']
            launch('system')


if __name__ == '__main__':
    unittest.main()

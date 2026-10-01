import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from app.config import BACKEND_DIR, Settings  # noqa: E402


class ConfigLoadingTest(unittest.TestCase):
    def test_config_path_is_backend_absolute_and_cwd_independent(self):
        configured = Path(Settings.model_config['env_file'])
        self.assertTrue(configured.is_absolute())
        self.assertEqual(configured, BACKEND_DIR / '.env')
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intended = root / 'backend.env'
            intended.write_text('LLM_API_KEY=local-test-only\nLLM_MODEL=deepseek-flash\n')
            (root / '.env').write_text('LLM_API_KEY=wrong-cwd-file\n')
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    settings = Settings(_env_file=str(intended))
                self.assertEqual(settings.LLM_API_KEY, 'local-test-only')
                self.assertEqual(settings.LLM_MODEL, 'deepseek-flash')
            finally:
                os.chdir(original_cwd)

    def test_environment_overrides_local_file_including_empty_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'backend.env'
            path.write_text('LLM_API_KEY=local-test-only\nLLM_MODEL=deepseek-flash\n')
            with patch.dict(os.environ, {'LLM_API_KEY': '', 'LLM_MODEL': 'environment-model'}, clear=True):
                settings = Settings(_env_file=str(path))
            self.assertEqual(settings.LLM_API_KEY, '')
            self.assertEqual(settings.LLM_MODEL, 'environment-model')


if __name__ == '__main__':
    unittest.main()

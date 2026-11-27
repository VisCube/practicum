"""Author configuration comes from the process environment, not Settings overrides."""

import os
import subprocess
import sys
from pathlib import Path


def test_real_upstream_reads_environment_without_bridge_overwrites():
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update({
        'FM_API_KEY': 'process-key',
        'FM_BASE_URL': 'https://process.example.invalid/v1',
        'LLM_MODEL': 'process-model',
        'STT_MODEL': 'process-stt',
        'RAG_KEY_ID': 'process-id',
        'RAG_KEY_SECRET': 'process-secret',
        'RAG_BASE_URL': 'https://process.example.invalid/rag',
        'IAM_TOKEN_URL': 'https://process.example.invalid/iam',
        'PYTHONPATH': str(root / 'src'),
    })
    script = '''
import importlib, os
from assistant.api.dependencies import build_container
from assistant.core.config import Settings
settings = Settings(data_dir='data', dialog_adapter='cloudru', crm_adapter='off',
                    asr_adapter='fake', tts_adapter='fake', fm_api_key='settings-key',
                    rag_key_id='settings-id', rag_key_secret='settings-secret',
                    rag_base_url='https://settings.invalid', llm_model='settings-model',
                    _env_file=None)
container = build_container(settings)
config = importlib.import_module('src.config')
assert config.FM_API_KEY == 'process-key'
assert config.LLM_MODEL == 'process-model'
assert os.environ['FM_API_KEY'] == 'process-key'
assert os.environ['RAG_KEY_ID'] == 'process-id'
'''
    subprocess.run([sys.executable, '-c', script], env=environment, cwd=root,
                   check=True, capture_output=True, text=True, timeout=30)

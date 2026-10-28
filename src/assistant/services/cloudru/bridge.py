"""Загрузчик-мост для изоляции импортов `src.*` внутри `_upstream`.

Внешний модуль Cloud.ru использует абсолютные импорты вида `from src.config import ...`,
тогда как корень нашего проекта тоже содержит директорию `src/` с пакетом `assistant`.
Чтобы сохранить код в `_upstream/` без единого изменения и исключить коллизию namespace-пакетов,
перед ленивым импортом регистрируем `sys.modules["src"]` строго на `_upstream/src`.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from assistant.core.config import Settings

UPSTREAM_ROOT = Path(__file__).resolve().parent / "_upstream"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"


class UpstreamToolCall(Protocol):
    name: str
    arguments: dict[str, Any]
    result: str


class UpstreamAssistant(Protocol):
    messages: list[Any]

    def handle_message(self, user_input: str) -> tuple[str, list[UpstreamToolCall]]: ...


def ensure_upstream_src() -> None:
    """Направляет пакет `src` строго в `_upstream/src`, не затрагивая пакет `assistant`."""
    target = [str(UPSTREAM_SRC)]
    existing = sys.modules.get("src")
    if existing is not None and list(getattr(existing, "__path__", [])) == target:
        return
    pkg = types.ModuleType("src")
    pkg.__path__ = target
    pkg.__package__ = "src"
    sys.modules["src"] = pkg


def export_cloudru_env(settings: Settings, *, require_rag: bool) -> None:
    """Прокидывает настройки Cloud.ru из `Settings` в `os.environ` для `_upstream/src/config.py`."""
    pairs = {
        "FM_API_KEY": settings.fm_api_key,
        "FM_BASE_URL": settings.fm_base_url,
        "LLM_MODEL": settings.llm_model,
        "STT_MODEL": settings.stt_model,
        "RAG_KEY_ID": settings.rag_key_id,
        "RAG_KEY_SECRET": settings.rag_key_secret,
        "RAG_BASE_URL": settings.rag_base_url,
        "IAM_TOKEN_URL": settings.iam_token_url,
    }
    optional_when_asr_only = {"LLM_MODEL", "RAG_KEY_ID", "RAG_KEY_SECRET", "RAG_BASE_URL"}
    for key, value in pairs.items():
        if value:
            os.environ.setdefault(key, value)
        elif not require_rag and key in optional_when_asr_only:
            os.environ.setdefault(key, "")


def load_transcribe() -> Callable[[bytes], str]:
    """Лениво загружает функцию `transcribe(data: bytes) -> str` из `_upstream`."""
    ensure_upstream_src()
    module = importlib.import_module("src.transcribe")
    fn: Callable[[bytes], str] = module.transcribe
    return fn


def load_assistant_cls() -> type[UpstreamAssistant]:
    """Лениво загружает класс `Assistant` из `_upstream`."""
    ensure_upstream_src()
    module = importlib.import_module("src.assistant")
    cls: type[UpstreamAssistant] = module.Assistant
    return cls


def parse_tool_result(raw: str) -> Any:
    """Распаковывает результат вызова тула (в `_upstream` он может быть сериализован дважды)."""
    value: Any = raw
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value

"""Загрузчик-мост для изоляции импортов `src.*` внутри `_upstream`.

Внешний модуль Cloud.ru использует абсолютные импорты вида `from src.config import ...`,
тогда как корень нашего проекта тоже содержит директорию `src/` с пакетом `assistant`.
Чтобы сохранить код в `_upstream/` без единого изменения и исключить коллизию namespace-пакетов,
перед ленивым импортом регистрируем `sys.modules["src"]` строго на `_upstream/src`.
"""

from __future__ import annotations

import difflib
import functools
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
UPSTREAM_CATEGORIES = UPSTREAM_ROOT / "classifier" / "categories.json"

BASE_SLOT_LABELS: dict[str, str] = {
    "full_name": "ФИО",
    "address": "Адрес",
    "details": "Подробности",
}


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


@functools.lru_cache(maxsize=1)
def load_cloudru_categories() -> tuple[dict[str, Any], ...]:
    """Читает справочник категорий со схемами полей из `_upstream/classifier/categories.json`."""
    raw = json.loads(UPSTREAM_CATEGORIES.read_text(encoding="utf-8"))
    return tuple(item for item in raw if isinstance(item, dict) and "name" in item)


@functools.lru_cache(maxsize=1)
def load_cloudru_slot_labels() -> dict[str, str]:
    """Собирает русские названия базовых и дополнительных полей заявок из `categories.json`."""
    labels = dict(BASE_SLOT_LABELS)
    for category in load_cloudru_categories():
        props = category.get("parameters", {}).get("properties", {})
        for field_id, spec in props.items():
            desc = spec.get("description") if isinstance(spec, dict) else None
            if isinstance(desc, str) and desc.strip() and field_id not in labels:
                labels[field_id] = desc.strip()
    return labels


def find_cloudru_category(name: str) -> dict[str, Any] | None:
    """Находит категорию в справочнике по точному имени или ближайшему совпадению."""
    cleaned = name.strip()
    if not cleaned:
        return None
    categories = load_cloudru_categories()
    for item in categories:
        if item["name"] == cleaned:
            return item
    names = [str(item["name"]) for item in categories]
    matches = difflib.get_close_matches(cleaned, names, n=1, cutoff=0.6)
    if matches:
        target = matches[0]
        return next((item for item in categories if item["name"] == target), None)
    return next((item for item in categories if item["name"] == "Иное / Иное"), None)


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _install_ticket_guard() -> None:
    """Дополняет `TOOL_HANDLERS['create_ticket']` проверкой пустых строк и `oneOf`."""
    tools_mod = importlib.import_module("src.tools")
    handlers = getattr(tools_mod, "TOOL_HANDLERS", None)
    if not isinstance(handlers, dict):
        return
    current = handlers.get("create_ticket")
    if current is None or getattr(current, "_cloudru_guarded", False):
        return

    crm_mod = importlib.import_module("src.tools.crm")
    get_category_fn = getattr(crm_mod, "get_category", None)

    def guarded_create_ticket(
        full_name: str,
        address: str,
        details: str,
        category: str,
        **fields: Any,
    ) -> Any:
        clean_fields = {k: v for k, v in fields.items() if _is_filled(v)}
        if callable(get_category_fn):
            cat_info = get_category_fn(category)
            schema = cat_info.get("parameters", {}) if isinstance(cat_info, dict) else {}
            one_of = schema.get("oneOf")
            if isinstance(one_of, list) and one_of:
                satisfied = any(
                    all(req in clean_fields for req in branch.get("required", ()))
                    for branch in one_of
                    if isinstance(branch, dict)
                )
                if not satisfied:
                    candidates: list[str] = []
                    for branch in one_of:
                        if isinstance(branch, dict):
                            for req in branch.get("required", ()):
                                if req not in candidates:
                                    candidates.append(req)
                    if candidates:
                        return {"missing_fields": candidates}
        return current(
            full_name=full_name,
            address=address,
            details=details,
            category=category,
            **clean_fields,
        )

    guarded_create_ticket._cloudru_guarded = True  # type: ignore[attr-defined]
    handlers["create_ticket"] = guarded_create_ticket


def load_transcribe() -> Callable[[bytes], str]:
    """Лениво загружает функцию `transcribe(data: bytes) -> str` из `_upstream`."""
    ensure_upstream_src()
    module = importlib.import_module("src.transcribe")
    fn: Callable[[bytes], str] = module.transcribe
    return fn


def load_assistant_cls() -> type[UpstreamAssistant]:
    """Лениво загружает класс `Assistant` из `_upstream`."""
    ensure_upstream_src()
    _install_ticket_guard()
    module = importlib.import_module("src.assistant")
    cls: type[UpstreamAssistant] = module.Assistant
    return cls


def parse_tool_result(raw: str) -> Any:
    """Распаковывает JSON-результат вызова тула (поддерживает и одиночную, и двойную обёртку)."""
    value: Any = raw
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value

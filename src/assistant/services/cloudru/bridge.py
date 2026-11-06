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
import importlib.util
import json
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from assistant.core.models import Owner

if TYPE_CHECKING:
    from assistant.core.config import Settings

UPSTREAM_ROOT = Path(__file__).resolve().parent / "_upstream"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
UPSTREAM_CATEGORIES = UPSTREAM_ROOT / "classifier" / "categories.json"
UPSTREAM_CRM_DIR = UPSTREAM_SRC / "tools" / "crm"

BASE_SLOT_LABELS: dict[str, str] = {
    "resident": "ФИО заявителя",
    "street": "Улица",
    "house": "Дом",
    "apartment": "Номер квартиры",
    "entry": "Номер подъезда",
    "floor": "Этаж",
    "room": "Помещение",
    "parking_level": "Уровень паркинга",
    "advertising_location": "Место размещения рекламы",
    "meter_type": "Прибор учёта",
    "meter_location": "Расположение счётчика",
    "water_supply": "Тип водоснабжения",
    "period": "Период",
    "details": "Подробности",
    "description": "Описание",
    "status": "Статус",
    "deadline": "Срок решения",
    "full_name": "ФИО",
    "address": "Адрес",
    "flat": "Номер квартиры",
}


class UpstreamToolCall(Protocol):
    name: str
    arguments: dict[str, Any]
    result: str


class UpstreamAssistant(Protocol):
    messages: list[Any]

    def __init__(self, phone: str) -> None: ...
    def handle_message(self, user_input: str) -> tuple[str, list[UpstreamToolCall]]: ...


def ensure_upstream_src() -> None:
    """Направляет пакет `src` строго в `_upstream/src`, не затрагивая пакет `assistant`.

    Дополнительно явно регистрирует namespace-пакет `src.tools.crm` на директорию
    `_upstream/src/tools/crm`: в `feat/cloud-ru` внутри `crm/` нет `__init__.py`, поэтому
    оставшийся на диске старый файл `tools/crm.py` (например, после копирования поверх без
    удаления файлов) иначе перехватывает импорт и вызывает
    `ModuleNotFoundError: ... 'src.tools.crm' is not a package`.
    """
    target = [str(UPSTREAM_SRC)]
    existing = sys.modules.get("src")
    if existing is None or list(getattr(existing, "__path__", [])) != target:
        pkg = types.ModuleType("src")
        pkg.__path__ = target
        pkg.__package__ = "src"
        sys.modules["src"] = pkg

    if UPSTREAM_CRM_DIR.is_dir():
        crm_target = [str(UPSTREAM_CRM_DIR)]
        existing_crm = sys.modules.get("src.tools.crm")
        if existing_crm is None or list(getattr(existing_crm, "__path__", [])) != crm_target:
            crm_pkg = types.ModuleType("src.tools.crm")
            crm_pkg.__path__ = crm_target
            crm_pkg.__package__ = "src.tools.crm"
            sys.modules["src.tools.crm"] = crm_pkg
            tools_mod = sys.modules.get("src.tools")
            if tools_mod is not None:
                tools_mod.crm = crm_pkg  # type: ignore[attr-defined]


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
    return None


def _load_residents_module() -> Any:
    """Загружает `residents.py` из `_upstream` без инициализации сетевых ключей `src.config`."""
    ensure_upstream_src()
    mod_name = "src.tools.crm.residents"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    try:
        return importlib.import_module(mod_name)
    except Exception:
        # Фолбэк для окружений без выставленных FM_API_KEY / RAG_* (например, оффлайн-тесты)
        crm_pkg_name = "src.tools.crm"
        if crm_pkg_name not in sys.modules:
            crm_pkg = types.ModuleType(crm_pkg_name)
            crm_pkg.__path__ = [str(UPSTREAM_CRM_DIR)]
            crm_pkg.__package__ = crm_pkg_name
            sys.modules[crm_pkg_name] = crm_pkg
        for sub in ("addresses", "residents"):
            full = f"{crm_pkg_name}.{sub}"
            if full in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(full, UPSTREAM_CRM_DIR / f"{sub}.py")
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[full] = module
                spec.loader.exec_module(module)
        return sys.modules[mod_name]


def load_cloudru_demo_phones() -> list[str]:
    """Возвращает номера телефонов жильцов из CRM-заглушки `_upstream`."""
    residents_mod = _load_residents_module()
    residents = getattr(residents_mod, "RESIDENTS", ())
    phones: list[str] = []
    for resident in residents:
        phone_obj = getattr(resident, "phone", None)
        number = getattr(phone_obj, "number", None)
        if isinstance(number, str) and number.strip():
            phones.append(number.strip())
    return phones


def format_street_house(street: str, house: str) -> str:
    """Форматирует пару `(street, house)` в читаемый адрес."""
    s = street.strip()
    h = house.strip()
    parts: list[str] = []
    if s:
        lower = s.lower()
        street_prefixes = ("ул.", "ул ", "пр.", "пр-т", "проспект", "пер.", "б-р")
        prefix = "" if lower.startswith(street_prefixes) else "ул. "
        parts.append(f"{prefix}{s}")
    if h:
        lower_h = h.lower()
        prefix_h = "" if lower_h.startswith(("д.", "д ", "дом")) else "д. "
        parts.append(f"{prefix_h}{h}")
    return ", ".join(parts)


def resident_dict_to_owner(data: dict[str, Any], fallback_phone: str = "") -> Owner | None:
    """Преобразует сериализованный `Resident` из `_upstream` в доменную модель `Owner`."""
    full_name = str(data.get("name") or "").strip()
    if not full_name:
        return None
    raw_addr = data.get("address")
    addr = raw_addr if isinstance(raw_addr, dict) else {}
    street = str(addr.get("street") or "").strip()
    house = str(addr.get("house") or "").strip()
    apt = data.get("apartment")
    apt_str = "" if apt is None or isinstance(apt, bool) else str(apt).strip()
    phone = str(data.get("phone") or fallback_phone).strip()
    return Owner(
        id=phone or full_name,
        full_name=full_name,
        phone=phone,
        address=format_street_house(street, house),
        apartment=apt_str,
    )


def find_cloudru_resident(phone: str) -> Owner | None:
    """Ищет жильца в мок-базе `_upstream` по номеру телефона."""
    if not phone.strip():
        return None
    residents_mod = _load_residents_module()
    lookup = getattr(residents_mod, "get_resident_by_phone", None)
    if not callable(lookup):
        return None
    res = lookup(phone)
    if res is None:
        return None
    addr_obj = getattr(res, "address", None)
    phone_obj = getattr(res, "phone", None)
    raw = {
        "name": getattr(res, "name", ""),
        "phone": getattr(phone_obj, "number", phone),
        "address": {
            "street": getattr(addr_obj, "street", ""),
            "house": getattr(addr_obj, "house", ""),
        },
        "apartment": getattr(res, "apartment", ""),
    }
    return resident_dict_to_owner(raw, fallback_phone=phone)


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _install_tool_guards() -> None:
    """Дополняет `TOOL_HANDLERS` защитой от пустых строк и отсутствующего `FALLBACK_CATEGORY`."""
    tools_mod = importlib.import_module("src.tools")
    if "src.tools.crm" in sys.modules:
        tools_mod.crm = sys.modules["src.tools.crm"]  # type: ignore[attr-defined]
    handlers = getattr(tools_mod, "TOOL_HANDLERS", None)
    if not isinstance(handlers, dict):
        return

    classifier_mod = importlib.import_module("src.tools.classifier")
    tickets_mod = importlib.import_module("src.tools.crm.tickets")
    raw_get_category = getattr(classifier_mod, "get_category", None)

    if callable(raw_get_category) and not getattr(raw_get_category, "_cloudru_guarded", False):
        def guarded_get_category(category: str) -> dict[str, Any]:
            try:
                res: dict[str, Any] = raw_get_category(category)
                return res
            except StopIteration:
                return {
                    "id": None,
                    "name": category.strip() or "Обращение",
                    "type": "Персональная",
                    "action": "ticket",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }

        guarded_get_category._cloudru_guarded = True  # type: ignore[attr-defined]
        classifier_mod.get_category = guarded_get_category  # type: ignore[attr-defined]
        tickets_mod.get_category = guarded_get_category  # type: ignore[attr-defined]
        handlers["get_category"] = guarded_get_category

    current_create = handlers.get("create_ticket")
    if current_create is not None and not getattr(current_create, "_cloudru_guarded", False):
        def guarded_create_ticket(
                category: str,
                phone: str,
                description: str,
                **fields: Any,
        ) -> Any:
            if not _is_filled(description):
                return {"missing_fields": ["description"]}
            clean_fields = {k: v for k, v in fields.items() if _is_filled(v)}
            return current_create(
                category=category,
                phone=phone,
                description=description.strip(),
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
    _install_tool_guards()
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
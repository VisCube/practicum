"""Загрузчик-мост для изоляции импортов `src.*` внутри `_upstream`.

Внешний модуль Cloud.ru использует абсолютные импорты вида `from src.config import ...`,
тогда как корень нашего проекта тоже содержит директорию `src/` с пакетом `assistant`.
Чтобы сохранить код в `_upstream/` без единого изменения и исключить коллизию namespace-пакетов,
перед ленивым импортом регистрируем `sys.modules["src"]` строго на `_upstream/src`.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Any, Protocol

from assistant.core.models import Owner

log = logging.getLogger(__name__)

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
    def __init__(self, phone: str) -> None: ...
    def handle_message(self, user_input: str) -> tuple[str, list[UpstreamToolCall]]: ...


def ensure_upstream_src() -> None:
    """Направляет пакет `src` строго в `_upstream/src`, не затрагивая пакет `assistant`.

    В чистой версии upstream Python сам загружает namespace-пакет crm.
    Совместимость со старой смешанной копией нужна лишь при наличии crm.py рядом с
    директорией crm без __init__.py; обычный пакет автора не подменяется.
    """
    if not UPSTREAM_SRC.is_dir():
        raise RuntimeError(
            "Cloud.ru upstream отсутствует. Выполните: "
            "git submodule update --init -- src/assistant/services/cloudru/_upstream"
        )
    target = [str(UPSTREAM_SRC)]
    existing = sys.modules.get("src")
    if existing is None or list(getattr(existing, "__path__", [])) != target:
        pkg = types.ModuleType("src")
        pkg.__path__ = target
        pkg.__package__ = "src"
        sys.modules["src"] = pkg

    legacy_collision = (
        UPSTREAM_CRM_DIR.is_dir()
        and (UPSTREAM_SRC / "tools" / "crm.py").is_file()
        and not (UPSTREAM_CRM_DIR / "__init__.py").is_file()
    )
    if legacy_collision:
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


@functools.lru_cache(maxsize=1)
def load_cloudru_categories() -> tuple[dict[str, Any], ...]:
    """Читает справочник категорий со схемами полей из `_upstream/classifier/categories.json`."""
    if not UPSTREAM_SRC.is_dir():
        raise RuntimeError(
            "Cloud.ru upstream отсутствует. Выполните: "
            "git submodule update --init -- src/assistant/services/cloudru/_upstream"
        )
    try:
        raw = json.loads(UPSTREAM_CATEGORIES.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("categories must be a list")
        return tuple(item for item in raw if isinstance(item, dict) and "name" in item)
    except (OSError, ValueError):
        log.warning("Cloud.ru category metadata unavailable; using generic labels")
        return ()


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
    return None


def _load_residents_module() -> Any:
    """Опциональный источник номеров для стенда из авторской демо-CRM."""
    ensure_upstream_src()
    return importlib.import_module("src.tools.crm.residents")


def load_cloudru_demo_phones() -> list[str]:
    """Возвращает номера телефонов жильцов из CRM-заглушки `_upstream`."""
    try:
        residents_mod = _load_residents_module()
    except (ImportError, OSError, KeyError, AttributeError):
        log.warning("Cloud.ru demo phones unavailable")
        return []
    residents = getattr(residents_mod, "RESIDENTS", ())
    phones: list[str] = []
    for resident in residents:
        phone_obj = getattr(resident, "phone", None)
        number = getattr(phone_obj, "number", None)
        if isinstance(number, str) and number.strip():
            phones.append(number.strip())
    return phones



def normalize_caller_phone(value: str) -> str:
    """Телефон вызывающего: страна 7 и десять цифр номера."""
    if not isinstance(value, str) or re.search(r"[^0-9+()\s-]", value):
        raise ValueError("Телефон должен содержать 7 и 10 цифр номера")
    digits = re.sub(r"[+()\s-]", "", value)
    if len(digits) != 11 or not digits.startswith("7"):
        raise ValueError("Телефон должен содержать 7 и 10 цифр номера")
    return "+" + digits

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


def load_assistant_cls() -> type[UpstreamAssistant]:
    """Лениво загружает класс `Assistant` из `_upstream`."""
    ensure_upstream_src()
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

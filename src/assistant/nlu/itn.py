"""Обратная нормализация текста (ITN) после ASR: числительные словами → цифры."""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_warned_missing = False

# Основы русских порядковых 1..9 для склейки полупреобразованных составных («20 седьмом» → «27-м»)
_ORD_UNIT_STEMS: dict[str, int] = {
    "перв": 1,
    "втор": 2,
    "трет": 3,
    "четверт": 4,
    "пят": 5,
    "шест": 6,
    "седьм": 7,
    "восьм": 8,
    "девят": 9,
}

_ORD_SUFFIX_MAP: dict[str, str] = {
    "ый": "й",
    "ой": "й",
    "ий": "й",
    "ая": "я",
    "ья": "я",
    "ое": "е",
    "ье": "е",
    "ые": "е",
    "ьи": "и",
    "ого": "го",
    "его": "го",
    "ьего": "го",
    "ому": "му",
    "ему": "му",
    "ьему": "му",
    "ым": "м",
    "им": "м",
    "ьим": "м",
    "ом": "м",
    "ем": "м",
    "ьем": "м",
    "ую": "ю",
    "ью": "ю",
    "ей": "й",
    "ьей": "й",
}

_COMPOUND_ORD_RE = re.compile(
    r"\b([1-9]\d|100|[1-9]00)\s+("
    + "|".join(_ORD_UNIT_STEMS)
    + r")("
    + "|".join(sorted(_ORD_SUFFIX_MAP, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# «4ый», «5ом» от text2num → «4-й», «5-м»
_GLUED_ORD_RE = re.compile(
    r"\b(\d+)(" + "|".join(sorted(_ORD_SUFFIX_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Продиктованные по цифрам/парам номера заявок:
# «заявка 12 34» → «заявка 1234», «номер 5 4 3 2» → «номер 5432»
_TICKET_DIGITS_RE = re.compile(
    r"\b((?:заявк|обращени)[а-яё]*\s+(?:номер\s+|№\s*)?|номер[а-яё]*\s+|№\s*)"
    r"(\d{1,2}(?:\s+\d{1,2}){1,5})\b",
    re.IGNORECASE,
)


def _fix_compound_ordinal(match: re.Match[str]) -> str:
    base = int(match.group(1))
    unit = _ORD_UNIT_STEMS[match.group(2).lower().replace("ё", "е")]
    suffix = _ORD_SUFFIX_MAP[match.group(3).lower()]
    if base % 10 != 0:
        return match.group(0)
    return f"{base + unit}-{suffix}"


def _fix_glued_ordinal(match: re.Match[str]) -> str:
    digits = match.group(1)
    suffix = _ORD_SUFFIX_MAP[match.group(2).lower()]
    return f"{digits}-{suffix}"


def _join_ticket_digits(match: re.Match[str]) -> str:
    prefix = match.group(1)
    joined = "".join(match.group(2).split())
    return f"{prefix}{joined}"


def numbers_to_digits(text: str) -> str:
    """Переводит числительные в цифры и дочищает артефакты русского ITN."""
    global _warned_missing
    if not text or not text.strip():
        return text

    normalized = text
    try:
        from text_to_num import alpha2digit

        normalized = alpha2digit(text.replace("ё", "е"), "ru")
    except ImportError:
        if not _warned_missing:
            _warned_missing = True
            log.warning("text2num is not installed — ASR ITN fallback only")
    except Exception as exc:
        log.debug("alpha2digit skipped for %r: %s", text, exc)

    normalized = _COMPOUND_ORD_RE.sub(_fix_compound_ordinal, normalized)
    normalized = _GLUED_ORD_RE.sub(_fix_glued_ordinal, normalized)
    normalized = _TICKET_DIGITS_RE.sub(_join_ticket_digits, normalized)
    return normalized

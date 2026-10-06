"""Маркеры из data/escalation.yaml

Три разных сигнала:
  transfer   — клиент просит живого человека → перевод сразу;
  complaint  — угроза жалобы (прокуратура, ГЖИ, суд) → флаг в заявке,
               разговор продолжается;
  emergency  — авария (пожар, потоп, прорыв, газ) → critical + спец-сценарий,
               не ждать продолжения темы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from assistant.nlu.text import contains_stem


class MarkerKind(StrEnum):
    TRANSFER = "transfer"
    COMPLAINT = "complaint"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class Marker:
    """Один маркер: фраза (основа/кусок) и что делаем при совпадении"""

    phrase: str
    kind: MarkerKind
    label: str
    say_key: str | None = None  # ключ в emergency_say для emergency-маркеров


@dataclass
class Detection:
    """Результат проверки реплики по каждому виду сигнала"""

    transfer: list[str] = field(default_factory=list)
    complaint: list[str] = field(default_factory=list)
    emergency: list[str] = field(default_factory=list)
    _say: str | None = field(default=None, repr=False)
    _reason: str | None = field(default=None, repr=False)

    @property
    def wants_operator(self) -> bool:
        return bool(self.transfer) or bool(self.emergency)

    @property
    def wants_emergency(self) -> bool:
        return bool(self.emergency)

    @property
    def has_complaint(self) -> bool:
        return bool(self.complaint)

    @property
    def is_emergency(self) -> bool:
        return bool(self.emergency)

    @property
    def say(self) -> str | None:
        """Текст фразы для аварийной ситуации (из emergency_say), или None"""
        return self._say

    @property
    def reason(self) -> str | None:
        """Человеко-читаемая причина: 'авария: пожар', или None"""
        return self._reason


_BUCKET = {
    MarkerKind.TRANSFER: "transfer",
    MarkerKind.COMPLAINT: "complaint",
    MarkerKind.EMERGENCY: "emergency",
}


class EscalationDetector:
    """Ищет в реплике маркеры transfer / complaint / emergency по YAML"""

    def __init__(self, markers: list[Marker], emergency_say: dict[str, str] | None = None) -> None:
        self.markers = markers
        self._emergency_say = emergency_say or {}

    @classmethod
    def from_yaml(cls, path: Path) -> EscalationDetector:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        emergency_say: dict[str, str] = raw.get("emergency_say") or {}

        markers: list[Marker] = []
        for kind_str, items in raw.items():
            try:
                kind = MarkerKind(kind_str)
            except ValueError:
                continue
            for item in items or []:
                phrase = (item.get("phrase") or "").strip()
                if not phrase:
                    continue
                label = (item.get("label") or phrase).strip()
                say_key = item.get("say")
                if say_key and kind is MarkerKind.EMERGENCY:
                    if say_key not in emergency_say:
                        raise ValueError(
                            f"emergency marker say key {say_key!r} not found in emergency_say"
                        )
                markers.append(Marker(phrase=phrase, kind=kind, label=label, say_key=say_key))
        return cls(markers, emergency_say)

    def detect(self, text: str) -> Detection:
        """Все сработавшие маркеры; каждый label — не более одного раза в своём bucket.
        Emergency-маркеры упорядочены по позиции появления в тексте."""
        detection = Detection()
        if not text or not str(text).strip():
            return detection

        # Для emergency собираем (позиция, label, say_key), чтобы упорядочить по тексту
        emergency_hits: list[tuple[int, str, str | None]] = []

        for marker in self.markers:
            if not contains_stem(text, marker.phrase):
                continue
            if marker.kind is MarkerKind.EMERGENCY and _is_negated_burn(text, marker.phrase):
                continue

            if marker.kind is MarkerKind.EMERGENCY:
                pos = _find_phrase_position(text, marker.phrase)
                emergency_hits.append((pos, marker.label, marker.say_key))
            else:
                bucket: list[str] = getattr(detection, _BUCKET[marker.kind])
                if marker.label not in bucket:
                    bucket.append(marker.label)

        # Сортируем emergency по позиции в тексте, dedup по label
        emergency_hits.sort(key=lambda h: h[0])
        first_say_key: str | None = None
        for _pos, label, say_key in emergency_hits:
            if label not in detection.emergency:
                detection.emergency.append(label)
                if first_say_key is None:
                    first_say_key = say_key

        if detection.emergency:
            detection._reason = "авария: " + ", ".join(detection.emergency)
            if first_say_key and first_say_key in self._emergency_say:
                detection._say = self._emergency_say[first_say_key]

        return detection


def _find_phrase_position(text: str, phrase: str) -> int:
    """Приблизительная позиция первого слова фразы в тексте — для сортировки маркеров по порядку появления"""
    first_word = phrase.split()[0].rstrip(".")
    t = text.lower().replace("ё", "е")
    fw = first_word.lower().replace("ё", "е")
    idx = t.find(fw)
    return idx if idx >= 0 else len(text)


def _is_negated_burn(text: str, phrase: str) -> bool:
    """True, если phrase про горение или огонь и рядом отрицание"""
    p = phrase.lower().replace("ё", "е")
    if p not in ("горит", "горят", "горе"):
        return False
    t = text.lower().replace("ё", "е")
    return "не горит" in t or "не горят" in t or "не горе" in t

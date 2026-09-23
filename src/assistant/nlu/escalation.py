"""Маркеры из data/escalation.yaml

Два разных сигнала:
  transfer  — клиент просит живого человека → перевод сразу;
  complaint — угроза жалобы (прокуратура, ГЖИ, суд) → флаг в заявке,
              разговор продолжается
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from assistant.nlu.text import contains_stem


class MarkerKind(StrEnum):
    TRANSFER = "transfer"
    COMPLAINT = "complaint"


@dataclass(frozen=True)
class Marker:
    """Один маркер эскалации: фраза, по которой ищем, и что делаем при совпадении"""
    phrase: str
    kind: MarkerKind
    label: str


@dataclass
class Detection:
    """Результат проверки реплики: какие маркеры сработали по каждому виду сигнала"""
    transfer: list[str] = field(default_factory=list)    # метки сработавших маркеров
    complaint: list[str] = field(default_factory=list)

    @property
    def wants_operator(self) -> bool:
        return bool(self.transfer)


class EscalationDetector:
    """Ищет в реплике маркеры перевода и жалоб по списку из YAML"""
    def __init__(self, markers: list[Marker]) -> None:
        self.markers = markers

    @classmethod
    def from_yaml(cls, path: Path) -> "EscalationDetector":
        """Загружает маркеры из YAML, сгруппированных по виду"""
        from_yaml = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        markers = [
            Marker(phrase=marker["phrase"], kind=MarkerKind(kind_str), label=marker.get("label", marker["phrase"]))
            for kind_str, items in from_yaml.items()
            for marker in items or []
        ]
        return cls(markers)

    def detect(self, text: str) -> Detection:
        """Возвращает все сработавшие маркеры - каждый учитывается не более одного раза"""
        detection = Detection()
        for marker in self.markers:
            if not contains_stem(text, marker.phrase):
                continue
            bucket = detection.transfer if marker.kind is MarkerKind.TRANSFER else detection.complaint
            if marker.label not in bucket:
                bucket.append(marker.label)
        return detection
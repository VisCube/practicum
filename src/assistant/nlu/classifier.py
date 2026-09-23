"""Классификатор темы по ключевым словам из data/classifier.yaml"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from assistant.core.models import RiskLevel, SlotSpec, Topic
from assistant.nlu.text import contains_stem

FALLBACK_TOPIC_ID = "other"
_RISK_ORDER = {RiskLevel.CRITICAL: 0, RiskLevel.NORMAL: 1, RiskLevel.LOW: 2}


@dataclass(frozen=True)
class TopicMatch:
    """Результат классификации: тема, число совпавших ключевых слов и сами слова"""
    topic: Topic
    score: int
    matched: tuple[str, ...]


class TopicClassifier:
    """Подбирает тему по числу совпавших ключевых слов"""

    def __init__(self, topics: list[Topic], slots: dict[str, SlotSpec]) -> None:
        self.topics = topics
        self.slots = slots
        self._by_id = {t.id: t for t in topics}
        self.fallback: Topic | None = self._by_id.get(FALLBACK_TOPIC_ID)

    @classmethod
    def from_yaml(cls, path: Path) -> "TopicClassifier":
        """Загружает темы и слоты из YAML - при неизвестных слотах падает на старте"""
        yaml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        topics = [Topic.model_validate(t) for t in yaml_data["topics"]]
        slots = {slot["id"]: SlotSpec.model_validate(slot) for slot in yaml_data.get("slots", [])}
        for topic in topics:
            if unknown := set(topic.required_slots) - slots.keys():
                raise ValueError(f"Тема {topic.id}: неизвестные слоты {unknown}")
        return cls(topics, slots)

    def by_id(self, topic_id: str) -> Topic:
        return self._by_id[topic_id]

    def classify(self, text: str) -> TopicMatch | None:
        """Возвращает тему с наибольшим числом совпадений - при ничьей приоритет у критических; None если не нашли"""
        ranked: list[tuple[int, int, int, TopicMatch]] = []
        for order, topic in enumerate(self.topics):
            if self.fallback and topic.id == self.fallback.id:
                continue
            hits = tuple(keyword for keyword in topic.keywords if contains_stem(text, keyword))
            if hits:
                ranked.append((-len(hits), _RISK_ORDER[topic.risk_level], order,
                               TopicMatch(topic, len(hits), hits)))
        return min(ranked)[3] if ranked else None

    def candidates_for_llm(self) -> list[dict[str, str]]:
        """Список тем для промпта fallback LLM, не подходит - есть null"""
        return [{"id": topic.id, "name": topic.name} for topic in self.topics
                if not (self.fallback and topic.id == self.fallback.id)]
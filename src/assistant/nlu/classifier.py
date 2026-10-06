"""Классификатор темы из pkl"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import yaml

from assistant.core.models import RiskLevel, SlotSpec, Topic, TopicKind
from assistant.nlu.clf_preprocess import lemmatize_for_clf

logger = logging.getLogger(__name__)

FALLBACK_TOPIC_ID = "other"


@dataclass(frozen=True)
class TopicMatch:
    """Результат классификации: тема, score (0..100 ≈ conf%), отладочные метки"""

    topic: Topic
    score: int
    matched: tuple[str, ...]


def _slug(text: str, *, max_len: int = 48) -> str:
    import re

    s = text.lower().strip()
    s = re.sub(r"[^a-zа-яё0-9]+", "_", s, flags=re.I)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "topic")[:max_len]


class TopicClassifier:
    """Тема из pkl; слоты по-прежнему из YAML"""

    def __init__(
            self,
            topics: list[Topic],
            slots: dict[str, SlotSpec],
            *,
            bundle: dict | None = None,
            sub_confidence_threshold: float = 0.75,
            min_confidence: float = 0.35,
    ) -> None:
        self.topics = topics
        self.slots = slots
        self._by_id = {t.id: t for t in topics}
        self.fallback: Topic | None = self._by_id.get(FALLBACK_TOPIC_ID)
        if self.fallback is None:
            self.fallback = Topic(
                id=FALLBACK_TOPIC_ID,
                name="Другое",
                kind=TopicKind.APPEAL,
                bitrix_id="",
                risk_level=RiskLevel.NORMAL,
                keywords=[],
                required_slots=[],
            )
            self._by_id[FALLBACK_TOPIC_ID] = self.fallback
            if not any(t.id == FALLBACK_TOPIC_ID for t in self.topics):
                self.topics = [*self.topics, self.fallback]

        self._bundle = bundle
        self._sub_thr = sub_confidence_threshold
        self._min_conf = min_confidence

    @classmethod
    def from_yaml(
            cls,
            path: Path,
            model_path: Path | None = None,
            *,
            sub_confidence_threshold: float = 0.75,
            min_confidence: float = 0.35,
    ) -> TopicClassifier:
        yaml_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        topics = [Topic.model_validate(t) for t in yaml_data.get("topics") or []]
        slots = {
            slot["id"]: SlotSpec.model_validate(slot)
            for slot in yaml_data.get("slots") or []
        }
        for topic in topics:
            if unknown := set(topic.required_slots) - slots.keys():
                raise ValueError(f"Тема {topic.id}: неизвестные слоты {unknown}")

        topics = [
            topic.model_copy(update={"required_slots": []}) if topic.required_slots else topic
            for topic in topics
        ]

        bundle = None
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"classifier model not found: {model_path}")
            bundle = joblib.load(model_path)
            for key in ("main", "sub", "map"):
                if key not in bundle:
                    raise ValueError(f"model.pkl must contain '{key}', got {list(bundle)}")
            logger.info(
                "topic clf loaded from %s (main=%d sub=%d)",
                model_path,
                len(bundle["main"].classes_),
                len(bundle["sub"].classes_),
            )

        return cls(
            topics,
            slots,
            bundle=bundle,
            sub_confidence_threshold=sub_confidence_threshold,
            min_confidence=min_confidence,
        )

    def by_id(self, topic_id: str) -> Topic:
        return self._by_id[topic_id]

    def candidates_for_llm(self) -> list[dict[str, str]]:
        """Для LLM-fallback: main-классы модели или темы из yaml."""
        if self._bundle is not None:
            return [
                {"id": _slug(str(c)), "name": str(c).strip()}
                for c in self._bundle["main"].classes_
            ]
        return [
            {"id": t.id, "name": t.name}
            for t in self.topics
            if not (self.fallback and t.id == self.fallback.id)
        ]

    def classify(self, text: str) -> TopicMatch | None:
        if not text or not str(text).strip():
            return None
        if self._bundle is None:
            logger.warning("classify: no model bundle, fallback")
            return TopicMatch(self.fallback, 0, ("no_model",)) if self.fallback else None

        main_pred, main_conf, sub_pred, sub_conf = self._predict(text)

        if sub_conf > self._sub_thr:
            correct_main = self._bundle["map"].get(sub_pred, main_pred)
            if correct_main is None:
                correct_main = self._bundle["map"].get(sub_pred.strip(), main_pred)
            if main_pred != correct_main:
                main_pred, main_conf = correct_main, sub_conf

        conf = max(main_conf, sub_conf)
        main_s = str(main_pred).strip()
        sub_s = str(sub_pred).strip()

        if conf < self._min_conf:
            topic = self.fallback
            assert topic is not None
            return TopicMatch(
                topic,
                score=int(round(conf * 100)),
                matched=(f"main={main_s}", f"sub={sub_s}", f"low_conf={conf:.2f}"),
            )

        topic = Topic(
            id=_slug(sub_s),
            name=sub_s,
            kind=TopicKind.APPEAL,
            bitrix_id="",
            risk_level=RiskLevel.NORMAL,
            keywords=[],
            required_slots=[],
        )
        return TopicMatch(
            topic,
            score=int(round(conf * 100)),
            matched=(main_s, sub_s),
        )

    def _predict(self, text: str) -> tuple[str, float, str, float]:
        cleaned = lemmatize_for_clf(text)
        main_model = self._bundle["main"]
        sub_model = self._bundle["sub"]

        main_pred = main_model.predict([cleaned])[0]
        main_probs = main_model.predict_proba([cleaned])[0]
        main_conf = float(main_probs[list(main_model.classes_).index(main_pred)])

        sub_pred = sub_model.predict([cleaned])[0]
        sub_probs = sub_model.predict_proba([cleaned])[0]
        sub_conf = float(sub_probs[list(sub_model.classes_).index(sub_pred)])

        return str(main_pred), main_conf, str(sub_pred), sub_conf
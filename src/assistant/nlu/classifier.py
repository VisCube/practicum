"""Классификатор темы из pkl"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import yaml

from assistant.core.models import RiskLevel, SlotSpec, Topic, TopicKind
from assistant.nlu.clf_preprocess import lemmatize_for_clf
from assistant.nlu.text import contains_stem

logger = logging.getLogger(__name__)

FALLBACK_TOPIC_ID = "other"

_RISK_ORDER = {RiskLevel.CRITICAL: 3, RiskLevel.NORMAL: 2, RiskLevel.LOW: 1}


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
            return self._classify_by_keywords(text)

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
            # Модель не уверена — пробуем keyword fallback, иначе fallback="other"
            kw_match = self._classify_by_keywords(text)
            if kw_match is not None:
                return TopicMatch(
                    kw_match.topic,
                    score=int(round(conf * 100)),
                    matched=(f"main={main_s}", f"sub={sub_s}", f"low_conf={conf:.2f}", "keywords"),
                )
            topic = self.fallback
            assert topic is not None
            return TopicMatch(
                topic,
                score=int(round(conf * 100)),
                matched=(f"main={main_s}", f"sub={sub_s}", f"low_conf={conf:.2f}"),
            )

        slug_id = _slug(sub_s)
        topic = self._by_id.get(slug_id)

        # Проверяем: yaml-topic релевантен тексту? (keywords матчат или это fallback/пустой topic)
        if topic is not None:
            if topic.id == FALLBACK_TOPIC_ID:
                # other — всегда пробуем keyword fallback вместо него
                topic = None
            elif topic.keywords and not any(contains_stem(text, kw) for kw in topic.keywords):
                # keywords не подтверждают — модельный slug случайно совпал с нерелевантным yaml-id
                topic = None

        # Keyword fallback по тексту
        if topic is None:
            kw_match = self._classify_by_keywords(text)
            if kw_match is not None:
                return TopicMatch(
                    kw_match.topic,
                    score=int(round(conf * 100)),
                    matched=(main_s, sub_s, "keywords"),
                )

        # Если keyword fallback не нашёл — возвращаем yaml-topic (если был) или синтетический
        if topic is None:
            topic = self._by_id.get(slug_id)
        if topic is None:
            topic = Topic(
                id=slug_id,
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

    def _classify_by_keywords(self, text: str) -> TopicMatch | None:
        """Keyword-based fallback: считаем совпадения по каждому топику,
        выбираем максимум; при равенстве — сначала critical, потом порядок в yaml."""
        best_topic: Topic | None = None
        best_score = 0

        for topic in self.topics:
            if topic.id == FALLBACK_TOPIC_ID:
                continue
            if not topic.keywords:
                continue
            score = sum(1 for kw in topic.keywords if contains_stem(text, kw))
            if score == 0:
                continue
            if best_topic is None:
                best_topic = topic
                best_score = score
            elif score > best_score:
                best_topic = topic
                best_score = score
            elif score == best_score:
                # tie-break: сначала higher risk, потом порядок в файле (первый выигрывает)
                if _RISK_ORDER.get(topic.risk_level, 0) > _RISK_ORDER.get(best_topic.risk_level, 0):
                    best_topic = topic
                    best_score = score

        if best_topic is None:
            return None

        return TopicMatch(best_topic, best_score, ("keywords",))

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

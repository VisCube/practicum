"""LLM-задачи поверх правил: фолбэк классификации, разбор слотов, ответ по базе знаний

Любая ошибка LLM → None / пустой результат, правила продолжают работать. Бот не должен
падать из-за недоступной модели. Всё, что вернула модель, проверяется: id темы — по списку,
варианты — по choices, числа — правилами, ответ — по длине (его читает TTS)"""

import json
import logging

from assistant.core.models import Passage, SlotSpec
from assistant.nlu.slots import RuleSlotExtractor
from assistant.services.llm.base import LlmClient, LlmError

log = logging.getLogger(__name__)

MAX_ANSWER_CHARS = 400      # ~25 секунд речи
MAX_SLOT_CHARS = 120

_SYS_CLASSIFY = (
    "Ты классификатор обращений жителей в управляющую компанию ЖКХ. "
    "Верни JSON {\"topic\": \"<id>\"} с id одной из тем или {\"topic\": null}, если не подходит ни одна."
)
_SYS_SLOTS = (
    "Извлеки из реплики жителя значения полей. Верни JSON-объект: ключ — id поля, "
    "значение — строка. Поля, которых в реплике нет, не включай. Не выдумывай."
)
_SYS_ANSWER = (
    "Ты оператор контакт-центра управляющей компании. Ответь жителю на вопрос строго по приведённым "
    "фрагментам базы знаний, 2–3 предложения, разговорным языком, без markdown, без списков. "
    "Текст будет озвучен: не используй сокращения и цифры, где можно словами. "
    "Если фрагменты не отвечают на вопрос — напиши ровно: НЕТ ОТВЕТА."
)


def _json(json_str: str) -> dict:
    """Парсит JSON из ответа LLM - снимает markdown-обёртку, при ошибке возвращает пустой словарь"""
    json_str = json_str.strip()
    if json_str.startswith("```"):
        json_str = json_str.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


def _trim(text: str, limit: int) -> str:
    """Обрезать по границе предложения; если предложение одно и длинное - по слову с многоточием"""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if sentence_end > limit // 3:
        return truncated[:sentence_end + 1]
    return truncated[:truncated.rfind(" ")].rstrip(",;:") + "…"


class LlmTasks:
    """LLM-задачи поверх правил: классификация темы, извлечение слотов, ответ по базе знаний"""
    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm
        self._rules = RuleSlotExtractor()

    async def classify(self, text: str, candidates: list[dict[str, str]]) -> str | None:
        """Определяет тему через LLM - возвращает id из списка кандидатов или None"""
        user = "Темы:\n" + "\n".join(f"- {c['id']}: {c['name']}" for c in candidates)
        user += f"\n\nРеплика: «{text}»"
        try:
            topic = _json(await self.llm.complete(_SYS_CLASSIFY, user, json_mode=True)).get("topic")
        except LlmError as llm_error:
            log.warning("llm classify: %s", llm_error)
            return None
        valid = {candidate["id"] for candidate in candidates}
        return topic if isinstance(topic, str) and topic in valid else None

    async def extract_slots(self, text: str, specs: list[SlotSpec]) -> dict[str, str]:
        """Извлекает значения слотов из реплики через LLM - каждое значение проверяется правилами"""
        if not specs:
            return {}
        lines = []
        for spec in specs:
            hint = f" (варианты: {', '.join(spec.choices)})" if spec.choices else f" ({spec.kind})"
            lines.append(f"- {spec.id}: {spec.label}{hint}")
        user = "Поля:\n" + "\n".join(lines) + f"\n\nРеплика: «{text}»"
        try:
            llm_response = _json(await self.llm.complete(_SYS_SLOTS, user, json_mode=True))
        except LlmError as e:
            log.warning("llm slots: %s", e)
            return {}
        result: dict[str, str] = {}
        specs_by_id = {spec.id: spec for spec in specs}
        for slot_id, slot_value in llm_response.items():
            spec = specs_by_id.get(slot_id)
            if spec is None or not isinstance(slot_value, str | int) or not str(slot_value).strip():
                continue
            value = self._validate(spec, str(slot_value).strip())
            if value is not None:
                result[slot_id] = value
        return result

    def _validate(self, spec: SlotSpec, value: str) -> str | None:
        """Модели не верим: варианты - только из списка, числа - только числа, текст - короткий"""
        if spec.kind == "choice":
            return value if value in spec.choices else None
        if spec.kind == "number":
            return self._rules.extract(spec, value)     # «седьмой» → 7, «не знаю» → None
        return value[:MAX_SLOT_CHARS] if len(value) <= MAX_SLOT_CHARS * 2 else None

    async def answer(self, question: str, passages: list[Passage]) -> str | None:
        """Формулирует ответ по фрагментам базы знаний - None если LLM не нашла ответа"""
        if not passages:
            return None
        context = "\n\n".join(f"### {passage.heading}\n{passage.text}" for passage in passages)
        user = f"Фрагменты:\n{context}\n\nВопрос жителя: «{question}»"
        try:
            llm_answer = (await self.llm.complete(_SYS_ANSWER, user)).strip()
        except LlmError as e:
            log.warning("llm answer: %s", e)
            return None
        if not llm_answer or "НЕТ ОТВЕТА" in llm_answer.upper():
            return None
        return _trim(llm_answer, MAX_ANSWER_CHARS)
"""Сценарий 3: консультация из базы знаний.

Ищем фрагменты retriever'ом; если есть LLM — просим её сформулировать ответ по фрагментам,
иначе зачитываем лучший фрагмент. Дважды не нашли — перевод на оператора.
"""

from assistant.core.models import CallSession, CallState, Event, Reply
from assistant.nlu.llm_tasks import LlmTasks
from assistant.services.knowledge_base.base import KnowledgeBase
from assistant.templates.phrases import say

# какие файлы базы знаний относятся к теме; нет в словаре — ищем везде
TOPIC_SOURCES: dict[str, list[str]] = {
    "ipu": ["meter_verification", "faq"],
    "charges": ["charges", "faq"],
    "certificate": ["certificates", "faq"],
}
MAX_MISSES = 2


class ConsultationHandler:
    """Отвечает на вопрос по базе знаний - через LLM если доступна, иначе лучшим фрагментом напрямую"""
    def __init__(self, knowledge_base: KnowledgeBase, llm: LlmTasks | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.llm = llm

    async def handle(self, session: CallSession, text: str) -> Reply:
        """Ищет ответ в базе знаний - при двух промахах подряд переводит на оператора"""
        topic_id = session.topic.id if session.topic else ""
        passages = await self.knowledge_base.search(text, sources=TOPIC_SOURCES.get(topic_id), limit=3)
        if not passages and topic_id in TOPIC_SOURCES:
            passages = await self.knowledge_base.search(text, limit=3)  # расширяем поиск на всё

        answer: str | None = None
        if passages:
            if self.llm:
                answer = await self.llm.answer(text, passages)
            if answer is None:
                answer = passages[0].text

        if answer:
            session.attempts = 0
            session.state = CallState.DONE
            return Reply(text=say("consult_answer", answer=answer), state=CallState.DONE)

        session.attempts += 1
        if session.attempts >= MAX_MISSES:
            session.state = CallState.TRANSFERRED
            return Reply(text=say("consult_give_up"), transfer=True, state=CallState.TRANSFERRED,
                         events=[Event(type="transfer", payload={"reason": "не найден ответ"})])
        session.state = CallState.TOPIC  # остаёмся в теме, ждём переформулировку
        return Reply(text=say("consult_not_found"), state=CallState.TOPIC)
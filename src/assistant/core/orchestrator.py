"""Оркестратор: один ход диалога от текста до реплики

Порядок на ходе: сессия → просьба оператора или угроза жалобы → прощание → тема → обработчик
Про HTTP, аудио и конкретную CRM здесь ничего не известно"""

import asyncio
import logging
from collections.abc import Awaitable
from typing import TypeVar

from assistant.core.models import CallSession, CallState, Event, Reply, Topic, TopicKind
from assistant.core.session_store import SessionStore
from assistant.handlers.base import Handler
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.llm_tasks import LlmTasks
from assistant.services.crm.base import CrmClient
from assistant.services.telephony.base import TelephonyClient
from assistant.templates.phrases import is_goodbye, is_greeting, say

log = logging.getLogger(__name__)
T = TypeVar("T")

MAX_TOPIC_ATTEMPTS = 2
IDENTITY_SLOTS = ("full_name", "address", "apartment")
FREE_TEXT_SLOT = "details"   # слот, в ответе на который клиент может назвать новую тему


class Orchestrator:
    """Ведёт диалог от реплики клиента до ответа бота - без знаний об HTTP, аудио и конкретной CRM"""
    def __init__(
            self,
            store: SessionStore,
            crm: CrmClient,
            classifier: TopicClassifier,
            escalation: EscalationDetector,
            handlers: dict[TopicKind, Handler],
            telephony: TelephonyClient,
            llm: LlmTasks | None = None,
            *,
            crm_timeout: float = 2.0,
            transfer_on_complaint: bool = False,   # True — угроза жалобы = сразу к оператору
    ) -> None:
        self.store = store
        self.crm = crm
        self.classifier = classifier
        self.escalation = escalation
        self.handlers = handlers
        self.telephony = telephony
        self.llm = llm
        self.crm_timeout = crm_timeout
        self.transfer_on_complaint = transfer_on_complaint

    async def start(self, call_id: str, phone: str) -> Reply:
        """Начало звонка: приветствие, затем уведомления об авариях по адресу собственника."""
        session = CallSession(call_id=call_id, phone=phone)
        events: list[Event] = []

        session.owner = await self._crm(self.crm.find_owner_by_phone(phone), default=None, what="owner lookup")
        session.owner_lookup_done = True

        if session.owner:
            parts = [say("greeting_known", name=_first_name(session.owner.full_name))]
            outages = await self._crm(
                self.crm.active_outages(session.owner), default=[], what="outages",
            )
            for outage in outages:
                parts.append(say("outage_notice", address=outage.address, description=outage.description))
                events.append(Event(type="outage_notice", payload=outage.model_dump(mode="json")))
        else:
            parts = [say("greeting")]

        text = " ".join(parts)
        session.state = CallState.TOPIC
        session.say(text)
        self.store.save(session)
        return Reply(text=text, state=session.state, events=events)

    async def handle_turn(self, call_id: str, phone: str, text: str) -> Reply:
        """Принимает реплику и возвращает ответ - если сессии нет, сначала создаёт её"""
        text = text.strip()
        session = self.store.get(call_id)

        if session is None:
            greeting = await self.start(call_id, phone)
            if not text:
                return greeting
            session = self.store.get(call_id)
            if session is None:
                raise RuntimeError(f"session {call_id} vanished right after start")
            reply = await self._turn(session, text)
            return reply.model_copy(update={
                "text": f"{greeting.text} {reply.text}",
                "events": greeting.events + reply.events,
            })

        if not text:
            return Reply(text=say("didnt_hear"), state=session.state)
        return await self._turn(session, text)

    async def _turn(self, session: CallSession, text: str) -> Reply:
        session.hear(text)
        reply = await self._dispatch(session, text)
        session.say(reply.text)
        self.store.save(session)
        return reply

    async def _dispatch(self, session: CallSession, text: str) -> Reply:
        """Маршрутизирует ход: эскалация → прощание → тема → обработчик"""
        if session.state == CallState.TRANSFERRED:
            return Reply(text=say("already_transferred"), transfer=True, state=session.state)

        signal = self.escalation.detect(text)
        if signal.wants_operator:
            return await self._transfer(session, reason="клиент просит оператора", markers=signal.transfer)
        if signal.complaint and self.transfer_on_complaint:
            return await self._transfer(session, reason="угроза жалобы", markers=signal.complaint)
        prefix, events = self._note_complaint(session, signal.complaint)

        # прощаться можно после закрытого обращения или в самом начале (послушал про аварию — «спасибо»)
        can_say_bye = session.state == CallState.DONE or (
                session.state == CallState.TOPIC and session.topic is None and session.attempts == 0)
        if can_say_bye and is_goodbye(text):
            return Reply(text=say("goodbye"), end_call=True, state=CallState.DONE,
                         events=[Event(type="call_ended")])
        if session.state == CallState.DONE:
            self._reset_for_new_topic(session)

        if session.state == CallState.TOPIC:
            if (reply := await self._pick_topic(session, text)) is not None:
                return _decorate(reply, prefix, events)

        await self._maybe_leave_fallback(session, text)

        if session.topic is None:
            raise RuntimeError(f"call {session.call_id}: state {session.state} without topic")
        reply = await self.handlers[session.topic.kind].handle(session, text)
        if reply.transfer:
            reply = await self._finish_transfer(session, reply)
        return _decorate(reply, prefix, events)

    @staticmethod
    def _note_complaint(session: CallSession, markers: list[str]) -> tuple[str, list[Event]]:
        """Угроза жалобы: флаг в сессии, один раз проговариваем, сценарий не прерываем"""
        new = [marker for marker in markers if marker not in session.escalation_markers]
        if not new:
            return "", []
        session.escalated = True
        session.escalation_markers.extend(new)
        return say("complaint_noted") + " ", [Event(type="escalated", payload={"markers": new})]

    @staticmethod
    def _reset_for_new_topic(session: CallSession) -> None:
        """Новое обращение в том же звонке: тему и реквизиты сбрасываем, личность оставляем"""
        session.topic, session.pending_slot, session.attempts = None, None, 0
        session.slots = {k: v for k, v in session.slots.items() if k in IDENTITY_SLOTS}
        session.state = CallState.TOPIC

    async def _classify(self, text: str, *, use_llm: bool = True) -> Topic | None:
        """Правила, затем fallback LLM, None — тема не определена"""
        if match := self.classifier.classify(text):
            return match.topic
        if use_llm and self.llm:
            tid = await self.llm.classify(text, self.classifier.candidates_for_llm())
            if tid and tid != "other":
                return self.classifier.by_id(tid)
        return None

    async def _pick_topic(self, session: CallSession, text: str) -> Reply | None:
        """Определить тему: None — тема выбрана, можно отдавать обработчику"""
        topic = await self._classify(text)

        if topic is None and session.topic is not None:
            return None  # уже в теме (консультация ждёт переформулировку) — остаёмся

        if topic is None:
            if session.attempts == 0 and is_greeting(text):
                return Reply(text=say("greeting_back"), state=CallState.TOPIC)  # «добрый день» — не попытка
            session.attempts += 1
            if session.attempts < MAX_TOPIC_ATTEMPTS:
                return Reply(text=say("topic_unclear"), state=CallState.TOPIC)
            if self.classifier.fallback is None:
                return await self._transfer(session, reason="тема не определена", phrase="transfer_unclear")
            # клиент только что ответил на «что случилось?» — это и есть описание, третий раз не спрашиваем
            session.slots.setdefault(FREE_TEXT_SLOT, text)
            topic = self.classifier.fallback

        session.topic = topic
        session.attempts = 0
        return None

    async def _maybe_leave_fallback(self, session: CallSession, text: str) -> None:
        """Сидим в «Другом», а клиент назвал конкретную тему — переключаемся без лишних вопросов.
        LLM зовём только на свободном тексте: ФИО и адрес классифицировать незачем"""
        fallback_topic = self.classifier.fallback
        if not (fallback_topic and session.topic and session.topic.id == fallback_topic.id):
            return
        if session.state not in (CallState.SLOTS, CallState.IDENTIFY):
            return
        topic = await self._classify(text, use_llm=session.pending_slot == FREE_TEXT_SLOT)
        if topic and topic.id != fallback_topic.id:
            session.topic = topic
            session.slots.pop(FREE_TEXT_SLOT, None)
            session.pending_slot, session.attempts = None, 0
            session.state = CallState.TOPIC   # handler войдёт через _enter и снимет слоты из этой реплики


    async def _transfer(
            self, session: CallSession, *, reason: str,
            markers: list[str] | None = None, phrase: str = "transfer_requested",
    ) -> Reply:
        """Точка перевода: оператору отдаём всё, что успели собрать"""
        session.state = CallState.TRANSFERRED
        context = {
            "reason": reason,
            "markers": markers or [],
            "topic": session.topic.name if session.topic else None,
            "slots": session.slots,
            "owner": session.owner.full_name if session.owner else None,
            "escalated": session.escalated,
            "complaint_markers": session.escalation_markers,
        }
        try:
            await self.telephony.transfer_to_operator(session.call_id, reason)
        except Exception as error:  # noqa: BLE001 — телефония лежит, клиенту всё равно скажем, что переводим
            log.error("transfer failed for %s: %s", session.call_id, error)
        return Reply(text=say(phrase), transfer=True, state=CallState.TRANSFERRED,
                     events=[Event(type="transfer", payload=context)])

    async def _finish_transfer(self, session: CallSession, reply: Reply) -> Reply:
        """Handler решил перевести (консультация не нашла ответ): текст его, телефония и контекст — наши."""
        transfer_event = next((e for e in reply.events if e.type == "transfer"), None)
        reason = (transfer_event.payload.get("reason") if transfer_event else None) or "перевод из сценария"
        full = await self._transfer(session, reason=reason)
        return full.model_copy(update={"text": reply.text})

    async def _crm(self, call: Awaitable[T], *, default: T, what: str) -> T:
        """Выполняет CRM-запрос с таймаутом — при сбое возвращает default и пишет предупреждение"""
        try:
            return await asyncio.wait_for(call, self.crm_timeout)
        except Exception as error:  # noqa: BLE001 — CRM лежит, работаем без идентификации
            log.warning("crm %s failed: %s", what, error)
            return default


def _decorate(reply: Reply, prefix: str, events: list[Event]) -> Reply:
    if not prefix and not events:
        return reply
    return reply.model_copy(update={"text": prefix + reply.text, "events": events + reply.events})


def _first_name(full_name: str) -> str:
    parts = full_name.split()
    return " ".join(parts[1:3]) if len(parts) >= 3 else full_name
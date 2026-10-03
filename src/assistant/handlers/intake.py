"""Сценарий 1: приём обращения → заявка

Порядок: (идентификация, если собственник не найден) → реквизиты темы → подтверждение → outbox
Слоты сначала пытаемся снять правилами, при наличии LLM — ещё и ей (для первой реплики
и для длинных ответов). CRM напрямую не трогаем — только outbox.enqueue"""

from assistant.core.models import (
    Appeal,
    CallSession,
    CallState,
    Event,
    Reply,
    RiskLevel,
    SlotSpec,
    Topic,
)
from assistant.nlu.classifier import FALLBACK_TOPIC_ID
from assistant.nlu.llm_tasks import LlmTasks
from assistant.nlu.slots import RuleSlotExtractor, missing
from assistant.nlu.yesno import is_no, is_yes
from assistant.services.crm.outbox import Outbox
from assistant.templates.phrases import say

IDENTIFY_SLOTS = ["full_name", "address", "apartment"]
MAX_SLOT_ATTEMPTS = 2
_SLA_PHRASE = {RiskLevel.CRITICAL: "sla_critical", RiskLevel.NORMAL: "sla_normal", RiskLevel.LOW: "sla_low"}


class IntakeHandler:
    """Сценарий приёма обращения: идентификация → реквизиты → подтверждение → заявка в outbox"""
    def __init__(
            self,
            outbox: Outbox,
            slot_specs: dict[str, SlotSpec],
            extractor: RuleSlotExtractor | None = None,
            llm: LlmTasks | None = None,
    ) -> None:
        self.outbox = outbox
        self.slot_specs = slot_specs
        self.rules = extractor or RuleSlotExtractor()
        self.llm = llm

    async def handle(self, session: CallSession, text: str) -> Reply:
        """Маршрутизирует ход по текущему состоянию сессии"""
        match session.state:
            case CallState.TOPIC:
                return await self._enter(session, text)
            case CallState.IDENTIFY | CallState.SLOTS:
                return await self._fill(session, text)
            case CallState.CONFIRM:
                return await self._confirm(session, text)
        raise RuntimeError(f"intake: неожиданное состояние {session.state}")

    @staticmethod
    def _topic(session: CallSession) -> Topic:
        if session.topic is None:
            raise RuntimeError(f"intake: у сессии {session.call_id} нет темы")
        return session.topic

    @staticmethod
    def _is_fallback(topic: Topic) -> bool:
        return topic.id == FALLBACK_TOPIC_ID

    def _required(self, session: CallSession) -> list[str]:
        """Что нужно собрать: реквизиты идентификации (если нет собственника) + слоты темы"""
        required_slots = [] if session.owner else list(IDENTIFY_SLOTS)
        for slot_id in self._topic(session).required_slots:
            if slot_id not in required_slots:
                required_slots.append(slot_id)
        return required_slots

    async def _enter(self, session: CallSession, text: str) -> Reply:
        """Первый вход в сценарий: заполняем известные слоты из CRM и задаём первый вопрос"""
        topic = self._topic(session)
        if session.owner:
            session.slots.setdefault("apartment", session.owner.apartment)
            session.slots.setdefault("address", session.owner.address)
            session.slots.setdefault("full_name", session.owner.full_name)
        await self._prefill(session, text, first_utterance=True)

        prefix = "" if self._is_fallback(topic) else say("topic_ack", topic=topic.name)
        if not session.owner and missing(IDENTIFY_SLOTS, session.slots):
            prefix += say("identify_intro")
        return self._ask_next(session, prefix)

    async def _fill(self, session: CallSession, text: str) -> Reply:
        """Извлекает значение текущего слота - при провале переспрашивает, при двух провалах берёт текст как есть"""
        slot = session.pending_slot
        if slot is None:
            raise RuntimeError(f"intake: состояние {session.state} без pending_slot")
        slot_specs = self.slot_specs[slot]
        value = self.rules.extract(slot_specs, text)
        if value is None and self.llm:
            value = (await self.llm.extract_slots(text, [slot_specs])).get(slot)
        if value is None:
            session.attempts += 1
            if session.attempts >= MAX_SLOT_ATTEMPTS and slot_specs.kind in ("text", "name", "address"):
                value = text.strip()  # свободный текст — берём как есть
            else:
                retry_phrase = (say("slot_choice_retry", choices=", ".join(slot_specs.choices))
                     if slot_specs.kind == "choice" else say("slot_retry", question=slot_specs.question))
                return Reply(text=retry_phrase, state=session.state)
        session.slots[slot] = value
        session.attempts = 0
        if self.llm:
            await self._prefill(session, text, first_utterance=False)
        return self._ask_next(session, "")

    async def _confirm(self, session: CallSession, text: str) -> Reply:
        """Обрабатывает да/нет на подтверждении - создаёт заявку, сбрасывает слоты или возвращается к вопросам"""
        topic = self._topic(session)
        if is_yes(text):
            return self._create(session)
        if is_no(text):
            if self._is_fallback(topic):
                # тема так и не была понята — возвращаемся к её выбору
                session.topic, session.pending_slot, session.attempts = None, None, 0
                session.slots.pop("details", None)
                session.state = CallState.TOPIC
                return Reply(text=say("confirm_no_retopic"), state=CallState.TOPIC)
            # переспрашиваем то, что диктовал сам клиент; данные из CRM не трогаем
            from_crm = set(IDENTIFY_SLOTS) if session.owner else set()
            slots_to_redo = [slot_id for slot_id in topic.required_slots if slot_id not in from_crm]
            if not session.owner:
                slots_to_redo += IDENTIFY_SLOTS
            # ...но если клиент ничего не диктовал - неверно именно то, что из CRM
            if not slots_to_redo:
                slots_to_redo = list(topic.required_slots)
            for slot_id in slots_to_redo:
                session.slots.pop(slot_id, None)
            return self._ask_next(session, say("confirm_no"))
        session.attempts += 1
        return Reply(text=say("confirm_retry"), state=CallState.CONFIRM)

    async def _prefill(self, session: CallSession, text: str, *, first_utterance: bool) -> None:
        """Снимает слоты из реплики правилами и LLM не дожидаясь явного вопроса"""
        missing_slots = missing(self._required(session), session.slots)
        if not missing_slots:
            return
        # правилами — только однозначное: вариант из списка или число рядом с подсказкой
        for slot_id in missing_slots:
            spec = self.slot_specs[slot_id]
            if spec.kind == "choice":
                value = self.rules.extract(spec, text)
            elif spec.kind == "number":
                value = self.rules.extract_in_context(spec, text)
            else:
                continue
            if value is not None:
                session.slots[slot_id] = value
        if self.llm and first_utterance:
            remaining_slots = [self.slot_specs[slot_id] for slot_id in missing(self._required(session), session.slots)
                     if slot_id not in ("full_name", "address")]
            if remaining_slots:
                session.slots.update(await self.llm.extract_slots(text, remaining_slots))

    def _ask_next(self, session: CallSession, prefix: str) -> Reply:
        """Находит первый незаполненный слот и задаёт вопрос - если все собраны, переходит к подтверждению"""
        missing_slots = missing(self._required(session), session.slots)
        if missing_slots:
            slot = missing_slots[0]
            session.pending_slot = slot
            session.attempts = 0
            session.state = (CallState.IDENTIFY if slot in IDENTIFY_SLOTS and not session.owner
                             else CallState.SLOTS)
            return Reply(text=prefix + self.slot_specs[slot].question, state=session.state)
        session.pending_slot = None
        session.state = CallState.CONFIRM
        return Reply(text=prefix + say("confirm", summary=self._summary(session)), state=CallState.CONFIRM)

    def _summary(self, session: CallSession) -> str:
        """Формирует текст сводки для подтверждения: тема и все собранные реквизиты"""
        topic = self._topic(session)
        items = [] if self._is_fallback(topic) else [say("summary_item", label="тема", value=topic.name)]
        for slot_id in self._required(session):
            if slot_id in session.slots:
                items.append(say("summary_item", label=self.slot_specs[slot_id].label.lower(), value=session.slots[slot_id]))
        return ", ".join(items)

    def _create(self, session: CallSession) -> Reply:
        """Создаёт заявку, кладёт в outbox и возвращает ответ с итогами"""
        topic = self._topic(session)
        risk = RiskLevel.CRITICAL if session.escalated else topic.risk_level
        wanted = self._required(session) + IDENTIFY_SLOTS
        appeal = Appeal(
            call_id=session.call_id,
            phone=session.phone,
            owner=session.owner,
            topic=topic,
            slots={k: v for k, v in session.slots.items() if k in wanted},
            risk_level=risk,
            escalated=session.escalated,
            escalation_markers=list(session.escalation_markers),
            transcript=list(session.transcript),
        )
        local_id = self.outbox.enqueue(appeal)
        session.appeals.append(local_id)
        session.state = CallState.DONE
        session.pending_slot = None

        fields, seen = [], set()
        for slot_id in wanted:
            if slot_id in appeal.slots and slot_id not in seen:
                seen.add(slot_id)
                fields.append({"id": slot_id, "label": self.slot_specs[slot_id].label, "value": appeal.slots[slot_id]})

        return Reply(
            text=say("appeal_created", appeal_id=local_id, sla=say(_SLA_PHRASE[risk])),
            state=CallState.DONE,
            events=[Event(type="appeal_created", payload={
                "appeal_id": local_id, "topic": topic.name, "topic_id": topic.id,
                "bitrix_id": topic.bitrix_id, "risk_level": risk,
                "escalated": appeal.escalated, "escalation_markers": list(session.escalation_markers),
                "phone": appeal.phone,
                "owner": appeal.owner.full_name if appeal.owner else None,
                "identified": appeal.owner is not None,
                "fields": fields, "slots": appeal.slots,
            })],
        )
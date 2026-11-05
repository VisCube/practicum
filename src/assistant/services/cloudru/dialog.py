"""Тонкий адаптер звонка вокруг `Assistant` из Cloud.ru (`_upstream/src/assistant.py`).

Ведение диалога, выбор инструментов, поиск в Managed RAG и категоризация полностью
остаются на стороне промпта и `Assistant.handle_message`. Адаптер отвечает только за:
  - привязку экземпляра `Assistant` к `call_id` и синхронизацию `SessionStore`;
  - передачу сведений о звонящем и активных авариях из `CrmClient` на старте звонка;
  - эмиссию событий `tool_call` и перекладывание заявок в `Outbox` / переводов в телефонию.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from assistant.core.models import (
    Appeal,
    CallSession,
    CallState,
    Event,
    Outage,
    Owner,
    Reply,
    RiskLevel,
    Topic,
    TopicKind,
)
from assistant.core.session_store import SessionStore
from assistant.services.cloudru.bridge import (
    UpstreamAssistant,
    UpstreamToolCall,
    find_cloudru_category,
    load_assistant_cls,
    load_cloudru_slot_labels,
    parse_tool_result,
)
from assistant.services.crm.base import CrmClient
from assistant.services.crm.outbox import Outbox
from assistant.services.telephony.base import TelephonyClient
from assistant.templates.phrases import is_goodbye, say

log = logging.getLogger(__name__)
T = TypeVar("T")

_IDENTITY_SLOTS = ("full_name", "address", "apartment", "flat")
_RESERVED_TICKET_KEYS = {"id", "category", "missing_fields"}
_FAREWELL_MARKERS = ("до свидания", "всего доброго", "всего хорошего", "хорошего дня")


class CloudRuDialogAdapter:
    """Адаптер под контракт `Orchestrator` (`start` / `handle_turn`) для `Assistant`."""

    llm = None  # совместимость с проверками атрибута у Orchestrator

    def __init__(
        self,
        store: SessionStore,
        crm: CrmClient,
        outbox: Outbox,
        telephony: TelephonyClient,
        *,
        crm_timeout: float = 2.0,
        agent_timeout: float = 45.0,
        session_ttl_sec: float = 1800.0,
        assistant_factory: Callable[[], UpstreamAssistant] | None = None,
    ) -> None:
        self.store = store
        self.crm = crm
        self.outbox = outbox
        self.telephony = telephony
        self.crm_timeout = crm_timeout
        self.agent_timeout = agent_timeout
        self._ttl_sec = session_ttl_sec
        self._factory: Callable[[], UpstreamAssistant] = assistant_factory or load_assistant_cls()
        self._agents: dict[str, tuple[UpstreamAssistant, float]] = {}

    async def start(self, call_id: str, phone: str) -> Reply:
        """Начало звонка: подтягиваем карточку жителя/аварии из CRM и передаём в контекст агента."""
        self._sweep_agents()
        session = CallSession(call_id=call_id, phone=phone)
        events: list[Event] = []

        session.owner = await self._crm(
            self.crm.find_owner_by_phone(phone), default=None, what="owner lookup",
        )
        session.owner_lookup_done = True

        outages: list[Outage] = []
        if session.owner:
            session.slots["full_name"] = session.owner.full_name
            session.slots["address"] = session.owner.address
            session.slots["apartment"] = session.owner.apartment
            parts = [say("greeting_known", name=_first_name(session.owner.full_name))]
            outages = await self._crm(
                self.crm.active_outages(session.owner), default=[], what="outages",
            )
            for outage in outages:
                parts.append(
                    say("outage_notice", address=outage.address, description=outage.description),
                )
                events.append(Event(type="outage_notice", payload=outage.model_dump(mode="json")))
        else:
            parts = [say("greeting")]

        greeting_text = " ".join(parts)
        agent = self._new_agent(phone=phone, owner=session.owner, outages=outages)
        agent.messages.append({"role": "assistant", "content": greeting_text})
        self._save_agent(call_id, agent)

        session.state = CallState.TOPIC
        session.say(greeting_text)
        self.store.save(session)
        return Reply(text=greeting_text, state=session.state, events=events)

    async def handle_turn(self, call_id: str, phone: str, text: str) -> Reply:
        """Один ход диалога через `Assistant.handle_message`."""
        text = text.strip()
        session = self.store.get(call_id)

        if session is None:
            self._agents.pop(call_id, None)
            greeting = await self.start(call_id, phone)
            if not text:
                return greeting
            session = self.store.get(call_id)
            if session is None:
                raise RuntimeError(f"session {call_id} vanished right after start")
            reply = await self._turn(session, text)
            return reply.model_copy(
                update={
                    "text": f"{greeting.text} {reply.text}",
                    "events": greeting.events + reply.events,
                },
            )

        if not text:
            return Reply(text=say("didnt_hear"), state=session.state)
        return await self._turn(session, text)

    async def _turn(self, session: CallSession, text: str) -> Reply:
        if session.state == CallState.TRANSFERRED:
            reply = Reply(text=say("already_transferred"), transfer=True, state=session.state)
            session.hear(text)
            session.say(reply.text)
            self.store.save(session)
            return reply

        agent = self._get_or_restore_agent(session)
        session.hear(text)

        try:
            response_text, tool_calls = await asyncio.wait_for(
                asyncio.to_thread(agent.handle_message, text),
                timeout=self.agent_timeout,
            )
        except Exception:
            log.exception("cloudru agent failed for call %s", session.call_id)
            self._save_agent(session.call_id, agent)
            spoken = say("tech_unavailable")
            session.say(spoken)
            self.store.save(session)
            return Reply(
                text=spoken,
                transfer=False,
                end_call=False,
                events=[],
                state=session.state,
            )

        self._save_agent(session.call_id, agent)

        events: list[Event] = []
        transferred = False

        for call in tool_calls:
            parsed = parse_tool_result(call.result)
            events.append(
                Event(
                    type="tool_call",
                    payload={
                        "name": call.name,
                        "arguments": dict(call.arguments or {}),
                        "result": parsed,
                    },
                ),
            )
            if call.name == "get_category":
                self._record_category(session, call, parsed)
            elif call.name == "create_ticket":
                if ticket_event := self._record_ticket(session, call, parsed):
                    events.append(ticket_event)
            elif call.name == "transfer_to_operator":
                events.append(await self._record_transfer(session, call))
                transferred = True
            elif call.name == "search_knowledge_base" and session.state != CallState.TRANSFERRED:
                session.state = CallState.DONE

        spoken = response_text.strip()
        if not spoken:
            spoken = say("transfer_requested") if transferred else say("didnt_hear")

        end_call = False
        if not transferred and not tool_calls and is_goodbye(text) and _looks_like_farewell(spoken):
            end_call = True
            session.state = CallState.DONE
            events.append(Event(type="call_ended"))

        if transferred or end_call:
            self._agents.pop(session.call_id, None)

        session.say(spoken)
        self.store.save(session)
        return Reply(
            text=spoken,
            transfer=transferred,
            end_call=end_call,
            events=events,
            state=session.state,
        )

    def _record_category(
        self,
        session: CallSession,
        call: UpstreamToolCall,
        parsed: Any,
    ) -> None:
        args = call.arguments or {}
        cat_info = (
            parsed
            if isinstance(parsed, dict) and "name" in parsed
            else find_cloudru_category(str(args.get("category") or ""))
        )
        if not cat_info:
            return
        if session.state == CallState.DONE:
            session.slots = {k: v for k, v in session.slots.items() if k in _IDENTITY_SLOTS}
        session.topic = _build_topic(str(cat_info["name"]), cat_info)
        if session.state != CallState.TRANSFERRED:
            session.state = CallState.SLOTS

    def _record_ticket(
        self,
        session: CallSession,
        call: UpstreamToolCall,
        parsed: Any,
    ) -> Event | None:
        args = call.arguments or {}
        data = parsed if isinstance(parsed, dict) else {}

        category_name = str(
            data.get("category") or args.get("category") or "Обращение",
        ).strip()
        cat_info = find_cloudru_category(category_name)
        topic = _build_topic(
            str(cat_info["name"]) if cat_info and not data.get("category") else category_name,
            cat_info,
        )
        session.topic = topic

        # Если create_ticket вернул missing_fields — заявка ещё не создана, продолжаем опрос
        if "missing_fields" in data:
            partial = _extract_ticket_slots({}, args, cat_info)
            session.slots.update(partial)
            if session.state != CallState.TRANSFERRED:
                session.state = CallState.SLOTS
            return None

        slots = _extract_ticket_slots(data, args, cat_info)
        session.slots = {
            **{k: v for k, v in session.slots.items() if k in _IDENTITY_SLOTS},
            **slots,
        }

        risk = RiskLevel.CRITICAL if session.escalated else RiskLevel.NORMAL
        appeal_kwargs: dict[str, Any] = {
            "call_id": session.call_id,
            "phone": session.phone,
            "owner": session.owner,
            "topic": topic,
            "slots": dict(slots),
            "risk_level": risk,
            "escalated": session.escalated,
            "escalation_markers": list(session.escalation_markers),
            "transcript": list(session.transcript),
        }
        raw_id = data.get("id")
        if raw_id is not None and str(raw_id).strip():
            candidate_id = str(raw_id).strip()
            if self.outbox.store.get(candidate_id) is None:
                appeal_kwargs["id"] = candidate_id

        appeal = Appeal(**appeal_kwargs)
        local_id = self.outbox.enqueue(appeal)
        session.appeals.append(local_id)
        session.state = CallState.DONE

        labels = load_cloudru_slot_labels()
        fields = [
            {"id": slot_id, "label": labels.get(slot_id, slot_id), "value": value}
            for slot_id, value in slots.items()
        ]
        return Event(
            type="appeal_created",
            payload={
                "appeal_id": local_id,
                "topic": topic.name,
                "topic_id": topic.id,
                "bitrix_id": topic.bitrix_id,
                "risk_level": appeal.risk_level,
                "escalated": appeal.escalated,
                "escalation_markers": list(appeal.escalation_markers),
                "phone": appeal.phone,
                "owner": appeal.owner.full_name if appeal.owner else None,
                "identified": appeal.owner is not None,
                "fields": fields,
                "slots": appeal.slots,
            },
        )

    async def _record_transfer(self, session: CallSession, call: UpstreamToolCall) -> Event:
        args = call.arguments or {}
        reason = str(args.get("reason") or "перевод ассистентом").strip()
        session.state = CallState.TRANSFERRED
        try:
            await self.telephony.transfer_to_operator(session.call_id, reason)
        except Exception as error:
            log.error("transfer failed for %s: %s", session.call_id, error)
        context = {
            "reason": reason,
            "markers": [],
            "topic": session.topic.name if session.topic else None,
            "slots": session.slots,
            "owner": session.owner.full_name if session.owner else None,
            "escalated": session.escalated,
            "complaint_markers": session.escalation_markers,
        }
        return Event(type="transfer", payload=context)

    def _new_agent(
        self,
        *,
        phone: str,
        owner: Owner | None,
        outages: list[Outage],
    ) -> UpstreamAssistant:
        agent = self._factory()
        context_msg = _format_caller_context(phone=phone, owner=owner, outages=outages)
        if context_msg:
            agent.messages.append({"role": "system", "content": context_msg})
        return agent

    def _get_or_restore_agent(self, session: CallSession) -> UpstreamAssistant:
        entry = self._agents.get(session.call_id)
        if entry is not None:
            return entry[0]
        agent = self._new_agent(phone=session.phone, owner=session.owner, outages=[])
        for turn in session.transcript:
            role = "assistant" if turn.role == "bot" else "user"
            agent.messages.append({"role": role, "content": turn.text})
        self._save_agent(session.call_id, agent)
        return agent

    def _save_agent(self, call_id: str, agent: UpstreamAssistant) -> None:
        self._agents[call_id] = (agent, time.monotonic())

    def _sweep_agents(self) -> None:
        now = time.monotonic()
        expired = [cid for cid, (_, ts) in self._agents.items() if now - ts > self._ttl_sec]
        for cid in expired:
            self._agents.pop(cid, None)

    async def _crm(self, call: Awaitable[T], *, default: T, what: str) -> T:
        try:
            return await asyncio.wait_for(call, self.crm_timeout)
        except Exception as error:
            log.warning("crm %s failed: %s", what, error)
            return default


def _slot_str(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _build_topic(category_name: str, cat_info: dict[str, Any] | None) -> Topic:
    raw_id = cat_info.get("id") if isinstance(cat_info, dict) else None
    bitrix_id = str(raw_id) if raw_id is not None else ""
    topic_id = f"cloudru_{bitrix_id}" if bitrix_id else "cloudru_ticket"
    return Topic(
        id=topic_id,
        name=category_name,
        kind=TopicKind.APPEAL,
        bitrix_id=bitrix_id,
        risk_level=RiskLevel.NORMAL,
    )


def _extract_ticket_slots(
    data: dict[str, Any],
    args: dict[str, Any],
    cat_info: dict[str, Any] | None,
) -> dict[str, str]:
    """Собирает базовые и категорийные поля заявки в детерминированном порядке."""
    merged: dict[str, Any] = {**args, **data}
    schema_props = list(
        (cat_info.get("parameters", {}).get("properties", {}) if cat_info else {}).keys(),
    )
    ordered_keys = list(
        dict.fromkeys(["full_name", "address", *schema_props, *merged.keys(), "details"]),
    )
    # Ставим details последним для аккуратного отображения в карточке и Telegram
    ordered_keys = [k for k in ordered_keys if k != "details"] + ["details"]

    slots: dict[str, str] = {}
    for key in ordered_keys:
        if key in _RESERVED_TICKET_KEYS or key not in merged:
            continue
        rendered = _slot_str(merged[key])
        if rendered:
            slots[key] = rendered
    return slots


def _format_caller_context(
    *,
    phone: str,
    owner: Owner | None,
    outages: list[Outage],
) -> str:
    lines: list[str] = []
    if phone:
        lines.append(f"Телефон звонящего: {phone}")
    if owner:
        lines.append(f"Собственник в CRM: {owner.full_name}")
        lines.append(f"Адрес из CRM: {owner.address}, кв. {owner.apartment}")
        lines.append(
            f"(для полей заявки: address=\"{owner.address}\", flat={owner.apartment})"
        )
    if outages:
        lines.append("Активные отключения по адресу звонящего:")
        for outage in outages:
            lines.append(f"- {outage.address}: {outage.description}")
    if not lines:
        return ""
    return "Сведения о звонящем из CRM:\n" + "\n".join(lines)


def _first_name(full_name: str) -> str:
    parts = full_name.split()
    return " ".join(parts[1:3]) if len(parts) >= 3 else full_name


def _looks_like_farewell(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FAREWELL_MARKERS)

"""Тонкий адаптер звонка вокруг `Assistant` из Cloud.ru (`_upstream/src/assistant.py`).

Ведение диалога, приветствие через LLM, выбор инструментов, поиск в Managed RAG,
идентификация в CRM-заглушке и категоризация остаются на стороне промпта и
`Assistant.handle_message`.

Адаптер отвечает за:
  - передачу номера звонящего в `Assistant(phone=...)` и привязку агента к `call_id`;
  - переиспользование единого пула HTTP-соединений между всеми параллельными звонками;
  - выделенный пул рабочих потоков и семафор параллельности (чтобы синхронный агент не
    забивал дефолтный `asyncio.to_thread` при множестве одновременных звонков);
  - целостность истории `agent.messages` (откат при ошибке в середине tool-call цикла и
    пересборка агента после `TimeoutError`, где фоновый поток ещё может работать);
  - синхронизацию `SessionStore`, маркеров эскалации (`EscalationDetector`) и `Outbox`.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from assistant.core.models import (
    Appeal,
    CallSession,
    CallState,
    Event,
    Owner,
    Reply,
    RiskLevel,
    Topic,
    TopicKind,
)
from assistant.core.session_store import SessionStore
from assistant.nlu.escalation import Detection, EscalationDetector
from assistant.services.cloudru.bridge import (
    UpstreamAssistant,
    UpstreamToolCall,
    find_cloudru_category,
    find_cloudru_resident,
    load_assistant_cls,
    load_cloudru_slot_labels,
    parse_tool_result,
    resident_dict_to_owner,
)
from assistant.services.crm.base import CrmClient
from assistant.services.crm.outbox import Outbox
from assistant.services.telephony.base import TelephonyClient
from assistant.templates.phrases import is_goodbye, say

log = logging.getLogger(__name__)

_START_CUE = "Алло"
_IDENTITY_SLOTS = ("resident", "street", "house", "apartment", "full_name", "address", "flat")
_RESERVED_TICKET_KEYS = {"id", "category", "phone", "missing_fields"}
_TAIL_TICKET_KEYS = ("details", "description", "status", "deadline")
_FAREWELL_MARKERS = ("до свидания", "всего доброго", "всего хорошего", "хорошего дня")
_MAX_CONSECUTIVE_AGENT_FAILURES = 2
_MAX_CACHED_AGENTS = 2000
_COMPACT_AFTER_MESSAGES = 14
_COMPACT_KEEP_TAIL = 8
_COMPACT_TOOL_MAX_CHARS = 1200


class CloudRuDialogAdapter:
    """Адаптер под контракт `Orchestrator` (`start` / `handle_turn`) для `Assistant`."""

    llm = None  # совместимость с проверками атрибута у Orchestrator
    START_CUE = _START_CUE

    def __init__(
        self,
        store: SessionStore,
        crm: CrmClient | None,
        outbox: Outbox,
        telephony: TelephonyClient,
        *,
        escalation: EscalationDetector | None = None,
        crm_timeout: float = 2.0,
        agent_timeout: float = 45.0,
        session_ttl_sec: float = 1800.0,
        max_concurrency: int = 32,
        assistant_factory: Callable[[str], UpstreamAssistant] | None = None,
    ) -> None:
        self.store = store
        self.crm = crm
        self.outbox = outbox
        self.telephony = telephony
        self.escalation = escalation
        self.crm_timeout = crm_timeout
        self.agent_timeout = agent_timeout
        self._ttl_sec = session_ttl_sec
        self._factory: Callable[[str], UpstreamAssistant] = (
            assistant_factory or load_assistant_cls()
        )
        self._agents: dict[str, tuple[UpstreamAssistant, float]] = {}

        workers = max(4, int(max_concurrency))
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="cloudru-agent",
        )
        self._semaphore = asyncio.Semaphore(workers)
        self._shared_openai_client: Any = None

    async def aclose(self) -> None:
        """Закрывает пул потоков и общий HTTP-клиент OpenAI."""
        self._agents.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._shared_openai_client is not None:
            client, self._shared_openai_client = self._shared_openai_client, None
            with contextlib.suppress(Exception):
                client.close()

    def drop_call(self, call_id: str) -> None:
        """Удаляет агента завершённого звонка."""
        self._agents.pop(call_id, None)

    def sweep(self) -> None:
        """Фоновая очистка просроченных агентов (вызывается по таймеру приложения)."""
        self._sweep_agents()

    async def _run_agent_call(
        self,
        agent: UpstreamAssistant,
        user_input: str,
    ) -> tuple[str, list[UpstreamToolCall]]:
        loop = asyncio.get_running_loop()
        async with self._semaphore:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, agent.handle_message, user_input),
                timeout=self.agent_timeout,
            )

    async def start(self, call_id: str, phone: str) -> Reply:
        """Начало звонка: передаём `phone` в `Assistant(phone)` и получаем приветствие от LLM."""
        self._sweep_agents()
        clean_phone = phone.strip()
        session = CallSession(call_id=call_id, phone=clean_phone, owner_lookup_done=True)
        agent = self._new_agent(phone=clean_phone)

        events: list[Event] = []
        transferred = False
        spoken = ""
        ran_ok = False
        timed_out = False
        snapshot = copy.deepcopy(agent.messages)

        try:
            response_text, tool_calls = await self._run_agent_call(agent, _START_CUE)
            ran_ok = True
            events, transferred = await self._apply_tool_calls(session, tool_calls)
            spoken = response_text.strip()
        except TimeoutError:
            timed_out = True
            log.warning("cloudru agent timed out on start for call %s", call_id)
        except Exception:
            log.exception("cloudru agent failed on start for call %s", call_id)
            agent.messages[:] = snapshot

        if not spoken:
            if session.owner is None and not transferred:
                session.owner = find_cloudru_resident(clean_phone)
                if session.owner:
                    _populate_owner_slots(session, None, session.owner)
            if transferred:
                spoken = say("transfer_requested")
            elif session.owner:
                spoken = say("greeting_known", name=_first_name(session.owner.full_name))
            else:
                spoken = say("greeting")
            if not ran_ok and not timed_out:
                agent.messages.append({"role": "assistant", "content": spoken})

        session.state = CallState.TRANSFERRED if transferred else CallState.TOPIC
        if transferred or timed_out:
            # При таймауте рабочий поток ещё может мутировать agent.messages в фоне —
            # не сохраняем отравленный инстанс; следующий ход соберёт чистый из transcript.
            self._agents.pop(call_id, None)
        else:
            self._save_agent(call_id, agent)

        session.say(spoken)
        self.store.save(session)
        return Reply(
            text=spoken,
            transfer=transferred,
            state=session.state,
            events=events,
        )

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

        signal = self.escalation.detect(text) if self.escalation is not None else None
        pre_events = self._note_complaint(session, signal)

        agent = self._get_or_restore_agent(session)
        _compact_old_tool_messages(agent.messages)
        snapshot = copy.deepcopy(agent.messages)
        session.hear(text)

        try:
            response_text, tool_calls = await self._run_agent_call(agent, text)
        except TimeoutError:
            log.warning("cloudru agent timed out for call %s", session.call_id)
            # Не сохраняем инстанс, в котором фоновый поток ещё крутит handle_message
            self._agents.pop(session.call_id, None)
            return await self._handle_agent_failure(session, signal, pre_events)
        except Exception:
            log.exception("cloudru agent failed for call %s", session.call_id)
            agent.messages[:] = snapshot
            return await self._handle_agent_failure(
                session,
                signal,
                pre_events,
                agent=agent,
                user_text=text,
            )

        session.attempts = 0
        self._save_agent(session.call_id, agent)
        tool_events, transferred = await self._apply_tool_calls(session, tool_calls)
        events = [*pre_events, *tool_events]

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

    @staticmethod
    def _note_complaint(session: CallSession, signal: Detection | None) -> list[Event]:
        """Отмечает угрозы жалобы (ГЖИ, прокуратура, суд), не прерывая диалог агента."""
        if signal is None or not signal.complaint:
            return []
        new_markers = [m for m in signal.complaint if m not in session.escalation_markers]
        if not new_markers:
            return []
        session.escalated = True
        session.escalation_markers.extend(new_markers)
        return [Event(type="escalated", payload={"markers": new_markers})]

    async def _handle_agent_failure(
        self,
        session: CallSession,
        signal: Detection | None,
        pre_events: list[Event],
        *,
        agent: UpstreamAssistant | None = None,
        user_text: str = "",
    ) -> Reply:
        """Деградация при сбое LLM/RAG: авария или повторный сбой → сразу на оператора."""
        session.attempts += 1
        is_emergency = bool(signal and signal.wants_emergency)
        if is_emergency or session.attempts >= _MAX_CONSECUTIVE_AGENT_FAILURES:
            self._agents.pop(session.call_id, None)
            session.state = CallState.TRANSFERRED
            if is_emergency and signal is not None:
                reason = signal.reason or "авария"
                spoken = signal.say or say("tech_failure")
                markers = list(signal.emergency)
            else:
                reason = "техническая ошибка облачного агента"
                spoken = say("tech_failure")
                markers = []
            try:
                await self.telephony.transfer_to_operator(session.call_id, reason)
            except Exception as error:
                log.error("transfer failed for %s: %s", session.call_id, error)
            context: dict[str, Any] = {
                "reason": reason,
                "markers": markers,
                "topic": session.topic.name if session.topic else None,
                "slots": session.slots,
                "owner": session.owner.full_name if session.owner else None,
                "escalated": session.escalated,
                "complaint_markers": session.escalation_markers,
            }
            if is_emergency:
                context["emergency"] = True
            session.say(spoken)
            self.store.save(session)
            return Reply(
                text=spoken,
                transfer=True,
                end_call=False,
                events=[*pre_events, Event(type="transfer", payload=context)],
                state=CallState.TRANSFERRED,
            )

        spoken = say("tech_unavailable")
        if agent is not None and user_text:
            agent.messages.append({"role": "user", "content": user_text})
            agent.messages.append({"role": "assistant", "content": spoken})
            self._save_agent(session.call_id, agent)
        session.say(spoken)
        self.store.save(session)
        return Reply(
            text=spoken,
            transfer=False,
            end_call=False,
            events=pre_events,
            state=session.state,
        )

    async def _apply_tool_calls(
        self,
        session: CallSession,
        tool_calls: list[UpstreamToolCall],
    ) -> tuple[list[Event], bool]:
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
            if call.name == "get_resident_by_phone":
                self._record_resident(session, parsed)
            elif call.name == "get_category":
                self._record_category(session, call, parsed)
            elif call.name == "get_tickets_by_address":
                if isinstance(parsed, list) and parsed and session.state != CallState.TRANSFERRED:
                    session.state = CallState.DONE
            elif call.name == "get_ticket":
                if isinstance(parsed, dict) and session.state != CallState.TRANSFERRED:
                    session.state = CallState.DONE
            elif call.name == "create_ticket":
                if ticket_event := self._record_ticket(session, call, parsed):
                    events.append(ticket_event)
            elif call.name == "transfer_to_operator":
                events.append(await self._record_transfer(session, call))
                transferred = True
            elif call.name == "search_knowledge_base" and session.state != CallState.TRANSFERRED:
                session.state = CallState.DONE

        return events, transferred

    def _record_resident(self, session: CallSession, parsed: Any) -> None:
        session.owner_lookup_done = True
        if not isinstance(parsed, dict):
            return
        owner = resident_dict_to_owner(parsed, fallback_phone=session.phone)
        if owner is None:
            return
        session.owner = owner
        _populate_owner_slots(session, parsed, owner)

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
            action = str(cat_info.get("action") or "ticket")
            session.state = CallState.DONE if action == "office" else CallState.SLOTS

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

        ticket_phone = _slot_str(data.get("phone") or args.get("phone")) or session.phone
        risk = RiskLevel.CRITICAL if session.escalated else RiskLevel.NORMAL
        appeal_kwargs: dict[str, Any] = {
            "call_id": session.call_id,
            "phone": ticket_phone,
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

    def _attach_shared_client(self, agent: UpstreamAssistant) -> None:
        """Переводит `agent.client` на единый потокобезопасный пул соединений с таймаутом."""
        raw_client = getattr(agent, "client", None)
        if raw_client is None or not hasattr(raw_client, "with_options"):
            return
        if self._shared_openai_client is None:
            self._shared_openai_client = raw_client.with_options(
                timeout=self.agent_timeout,
                max_retries=1,
            )
            agent.client = self._shared_openai_client  # type: ignore[attr-defined]
            return
        # Закрываем одноразовый пул сокетов, созданный внутри Assistant.__init__
        if raw_client is not self._shared_openai_client:
            with contextlib.suppress(Exception):
                raw_client.close()
        agent.client = self._shared_openai_client  # type: ignore[attr-defined]

    def _new_agent(self, *, phone: str) -> UpstreamAssistant:
        agent = self._factory(phone)
        self._attach_shared_client(agent)
        return agent

    def _get_or_restore_agent(self, session: CallSession) -> UpstreamAssistant:
        entry = self._agents.get(session.call_id)
        if entry is not None:
            return entry[0]
        agent = self._new_agent(phone=session.phone)
        if session.owner:
            agent.messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Звонящий идентифицирован: {session.owner.full_name}, "
                        f"адрес: {session.owner.address}, кв. {session.owner.apartment}."
                    ),
                },
            )
        for turn in session.transcript:
            role = "assistant" if turn.role == "bot" else "user"
            agent.messages.append({"role": role, "content": turn.text})
        self._save_agent(session.call_id, agent)
        return agent

    def _save_agent(self, call_id: str, agent: UpstreamAssistant) -> None:
        self._agents.pop(call_id, None)
        self._agents[call_id] = (agent, time.monotonic())
        while len(self._agents) > _MAX_CACHED_AGENTS:
            oldest_id = next(iter(self._agents))
            self._agents.pop(oldest_id, None)

    def _sweep_agents(self) -> None:
        now = time.monotonic()
        expired = [cid for cid, (_, ts) in self._agents.items() if now - ts > self._ttl_sec]
        for cid in expired:
            self._agents.pop(cid, None)


def _compact_old_tool_messages(messages: list[Any]) -> None:
    """Поджимает старые громоздкие JSON-ответы тулов в длинном звонке, сохраняя схему OpenAI."""
    if len(messages) <= _COMPACT_AFTER_MESSAGES:
        return
    cutoff = len(messages) - _COMPACT_KEEP_TAIL
    for idx in range(cutoff):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str) and len(content) > _COMPACT_TOOL_MAX_CHARS:
            msg["content"] = content[: _COMPACT_TOOL_MAX_CHARS - 1] + "…"


def _slot_str(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _populate_owner_slots(
    session: CallSession,
    raw: dict[str, Any] | None,
    owner: Owner,
) -> None:
    session.slots["resident"] = owner.full_name
    raw_addr = raw.get("address") if isinstance(raw, dict) else None
    addr = raw_addr if isinstance(raw_addr, dict) else {}
    street = _slot_str(addr.get("street"))
    house = _slot_str(addr.get("house"))
    if street:
        session.slots["street"] = street
    if house:
        session.slots["house"] = house
    if owner.apartment:
        session.slots["apartment"] = owner.apartment


def _build_topic(category_name: str, cat_info: dict[str, Any] | None) -> Topic:
    raw_id = cat_info.get("id") if isinstance(cat_info, dict) else None
    bitrix_id = str(raw_id) if raw_id is not None else ""
    topic_id = f"cloudru_{bitrix_id}" if bitrix_id else "cloudru_ticket"
    action = str(cat_info.get("action") or "ticket") if isinstance(cat_info, dict) else "ticket"
    kind = TopicKind.CONSULT if action in ("consult", "office") else TopicKind.APPEAL
    return Topic(
        id=topic_id,
        name=category_name,
        kind=kind,
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
        dict.fromkeys(
            [
                "resident",
                "full_name",
                "street",
                "house",
                "apartment",
                "address",
                *schema_props,
                *merged.keys(),
            ],
        ),
    )
    for tail_key in _TAIL_TICKET_KEYS:
        if tail_key in ordered_keys:
            ordered_keys = [k for k in ordered_keys if k != tail_key] + [tail_key]

    slots: dict[str, str] = {}
    for key in ordered_keys:
        if key in _RESERVED_TICKET_KEYS or key not in merged:
            continue
        rendered = _slot_str(merged[key])
        if rendered:
            slots[key] = rendered
    return slots


def _first_name(full_name: str) -> str:
    parts = full_name.split()
    if len(parts) >= 3:
        return " ".join(parts[1:3])
    if len(parts) == 2:
        return parts[1]
    return full_name


def _looks_like_farewell(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _FAREWELL_MARKERS)

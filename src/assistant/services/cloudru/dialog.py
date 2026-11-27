"""Адаптер стенда к публичному интерфейсу авторского Cloud.ru Assistant.

Не изменяет клиент, историю или инструменты автора. Ограничивает параллелизм,
сериализует ходы звонка и переводит на оператора при неопределённом результате.
"""

from __future__ import annotations

import asyncio
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
    load_assistant_cls,
    load_cloudru_slot_labels,
    normalize_caller_phone,
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
        outbox_enabled: bool = True,
        delivery_target: str = "crm",
        assistant_factory: Callable[[str], UpstreamAssistant] | None = None,
    ) -> None:
        self.store = store
        self.crm = crm
        self.outbox = outbox
        self._outbox_enabled = outbox_enabled
        self._delivery_target = delivery_target
        self._seen_ticket_ids: dict[str, None] = {}
        self.telephony = telephony
        self.escalation = escalation
        self.crm_timeout = crm_timeout
        self.agent_timeout = agent_timeout
        self._ttl_sec = session_ttl_sec
        self._factory: Callable[[str], UpstreamAssistant] = (
            assistant_factory or load_assistant_cls()
        )
        self._agents: dict[str, tuple[UpstreamAssistant, float]] = {}
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self._pending_turns: dict[str, asyncio.Task[Reply]] = {}
        self._pending_starts: dict[str, asyncio.Task[Reply]] = {}
        self._start_locks: dict[str, asyncio.Lock] = {}
        self._retired_starts: set[asyncio.Task[Reply]] = set()
        self._retired_turns: set[asyncio.Task[Reply]] = set()
        self._current_sessions: dict[str, CallSession] = {}

        workers = max(4, int(max_concurrency))
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="cloudru-agent",
        )
        self._semaphore = asyncio.Semaphore(workers)

    async def aclose(self) -> None:
        """Закрывает пул потоков."""
        self._agents.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def drop_call(self, call_id: str) -> None:
        """Удаляет агента завершённого звонка."""
        self._agents.pop(call_id, None)
        self._current_sessions.pop(call_id, None)
        start = self._pending_starts.get(call_id)
        if start is not None:
            self._retired_starts.add(start)
            if start.done():
                self._cleanup_retired(call_id, start)
        turn = self._pending_turns.get(call_id)
        if turn is not None:
            self._retired_turns.add(turn)
            if turn.done():
                self._cleanup_retired(call_id, turn)
        # In-flight task must finish before lock bookkeeping is discarded.
        if call_id not in self._pending_turns:
            lock = self._turn_locks.get(call_id)
            if lock is not None and not lock.locked():
                self._turn_locks.pop(call_id, None)
        if call_id not in self._pending_starts:
            lock = self._start_locks.get(call_id)
            if lock is not None and not lock.locked():
                self._start_locks.pop(call_id, None)

    def sweep(self) -> None:
        """Фоновая очистка просроченных агентов и закрытых сессий."""
        self._sweep_agents()
        contains = getattr(self.store, "contains_or_locked", None)
        if callable(contains):
            for call_id in list(self._current_sessions):
                pending = (self._pending_starts.get(call_id), self._pending_turns.get(call_id))
                running = any(task is not None and not task.done() for task in pending)
                if not running and not contains(call_id):
                    self.drop_call(call_id)

    async def _run_agent_call(
        self,
        agent: UpstreamAssistant,
        user_input: str,
    ) -> tuple[str, list[UpstreamToolCall]]:
        loop = asyncio.get_running_loop()

        def invoke() -> tuple[str, list[UpstreamToolCall]]:
            # StopIteration не должен переходить через asyncio.Future: поведение
            # зависит от версии Python и может оставить ожидающую задачу без результата.
            try:
                return agent.handle_message(user_input)
            except StopIteration as exc:
                raise RuntimeError("upstream tool raised StopIteration") from exc

        await asyncio.wait_for(self._semaphore.acquire(), timeout=self.agent_timeout)
        try:
            future = loop.run_in_executor(self._executor, invoke)
        except BaseException:
            self._semaphore.release()
            raise

        def release_when_finished(done: asyncio.Future[Any]) -> None:
            self._semaphore.release()
            # После отмены ожидающей задачи исключение нужно извлечь, иначе оно
            # останется необработанным в event loop.
            if not done.cancelled():
                done.exception()

        future.add_done_callback(release_when_finished)
        # wait_for не должен отменять Future потока: разрешение семафора удерживается
        # до фактического окончания работы, включая таймаут и barge.
        return await asyncio.wait_for(asyncio.shield(future), timeout=self.agent_timeout)

    async def start(self, call_id: str, phone: str) -> Reply:
        """Начало звонка продолжается при отмене ожидания клиентом."""
        lock = self._start_locks.setdefault(call_id, asyncio.Lock())
        async with lock:
            old_turn = self._pending_turns.get(call_id)
            if old_turn is not None and old_turn in self._retired_turns:
                await asyncio.shield(old_turn)
                self._retired_turns.discard(old_turn)
                if self._pending_turns.get(call_id) is old_turn:
                    self._pending_turns.pop(call_id, None)
            task = self._pending_starts.get(call_id)
            if task is not None and task in self._retired_starts:
                await asyncio.shield(task)
                self._retired_starts.discard(task)
                if self._pending_starts.get(call_id) is task:
                    self._pending_starts.pop(call_id, None)
                task = None
            if task is None:
                task = asyncio.create_task(self._start_impl(call_id, phone))
                self._pending_starts[call_id] = task
                task.add_done_callback(_consume_task_exception)
                task.add_done_callback(lambda done: self._cleanup_retired(call_id, done))
            try:
                reply = await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._pending_starts.pop(call_id, None)
                raise
            self._pending_starts.pop(call_id, None)
            return reply

    async def _start_impl(self, call_id: str, phone: str) -> Reply:
        """Передаёт телефон в Assistant и получает приветствие от LLM."""
        self._sweep_agents()
        clean_phone = normalize_caller_phone(phone)
        session = CallSession(call_id=call_id, phone=clean_phone, owner_lookup_done=True)
        self._current_sessions[call_id] = session
        agent = self._new_agent(phone=clean_phone)

        events: list[Event] = []
        transferred = False
        spoken = ""
        timed_out = False

        try:
            response_text, tool_calls = await self._run_agent_call(agent, _START_CUE)
            events, transferred = await self._apply_tool_calls(session, tool_calls)
            spoken = (
                say("tech_unavailable")
                if any(event.payload.get("integration_error") for event in events)
                else response_text.strip()
            )
        except TimeoutError:
            timed_out = True
            log.warning("cloudru agent timed out on start for call %s", call_id)
        except Exception:
            log.exception("cloudru agent failed on start for call %s", call_id)
            timed_out = True  # состояние агента после ошибки неизвестно

        if any(event.payload.get("integration_error") for event in events):
            self._agents.pop(call_id, None)
            return await self._handle_agent_failure(session, None, events)

        if timed_out:
            # Экземпляр и результат его инструментов неизвестны: не возобновляем
            # автоматически и не повторяем потенциально выполненное действие.
            self._agents.pop(call_id, None)
            return await self._handle_agent_failure(session, None, events)

        if not spoken:
            if transferred:
                spoken = say("transfer_requested")
            elif session.owner:
                spoken = say("greeting_known", name=_first_name(session.owner.full_name))
            else:
                spoken = say("greeting")

        session.state = CallState.TRANSFERRED if transferred else CallState.TOPIC
        if transferred or timed_out:
            self._agents.pop(call_id, None)
        else:
            if self._session_is_current(session):
                self._save_agent(call_id, agent)

        session.say(spoken)
        self._save_session(session)
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
            reply = await self._queued_turn(session, text)
            return reply.model_copy(
                update={
                    "text": f"{greeting.text} {reply.text}",
                    "events": greeting.events + reply.events,
                },
            )

        if not text:
            return Reply(text=say("didnt_hear"), state=session.state)
        return await self._queued_turn(session, text)

    async def _queued_turn(self, session: CallSession, text: str) -> Reply:
        """Не переиспользовать агента, пока предыдущий прерванный ход работает."""
        call_id = session.call_id
        lock = self._turn_locks.setdefault(call_id, asyncio.Lock())
        async with lock:
            previous = self._pending_turns.get(call_id)
            carried_events: list[Event] = []
            if previous is not None:
                try:
                    old_reply = await asyncio.shield(previous)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("cloudru interrupted turn failed for call %s", call_id)
                    self._pending_turns.pop(call_id, None)
                    self._retired_turns.discard(previous)
                    self._agents.pop(call_id, None)
                    session.hear(text)
                    return await self._handle_agent_failure(session, None, [])
                if previous not in self._retired_turns:
                    carried_events = old_reply.events
                self._retired_turns.discard(previous)
                self._pending_turns.pop(call_id, None)
            task = asyncio.create_task(self._turn(session, text))
            task.add_done_callback(_consume_task_exception)
            task.add_done_callback(lambda done: self._cleanup_retired(call_id, done))
            self._pending_turns[call_id] = task
            try:
                reply = await asyncio.shield(task)
            except asyncio.CancelledError:
                # The agent remains running. The next turn will wait for its result.
                raise
            except Exception:
                log.exception("cloudru turn processing failed for call %s", call_id)
                self._pending_turns.pop(call_id, None)
                self._agents.pop(call_id, None)
                return await self._handle_agent_failure(session, None, carried_events)
            self._pending_turns.pop(call_id, None)
            if carried_events:
                reply = reply.model_copy(update={"events": carried_events + reply.events})
            return reply

    async def _turn(self, session: CallSession, text: str) -> Reply:
        if session.state == CallState.TRANSFERRED:
            reply = Reply(text=say("already_transferred"), transfer=True, state=session.state)
            session.hear(text)
            session.say(reply.text)
            self._save_session(session)
            return reply

        signal = self.escalation.detect(text) if self.escalation is not None else None
        pre_events = self._note_complaint(session, signal)

        restored = session.call_id not in self._agents
        agent = self._get_or_restore_agent(session)
        context = _restore_context(session) if restored else ""
        session.hear(text)
        agent_text = f"{context}\nТекущая реплика: {text}" if context else text

        try:
            response_text, tool_calls = await self._run_agent_call(agent, agent_text)
        except TimeoutError:
            log.warning("cloudru agent timed out for call %s", session.call_id)
            # Не используем инстанс, пока фоновый поток ещё работает.
            self._agents.pop(session.call_id, None)
            return await self._handle_agent_failure(session, signal, pre_events)
        except Exception:
            log.exception("cloudru agent failed for call %s", session.call_id)
            self._agents.pop(session.call_id, None)
            return await self._handle_agent_failure(
                session,
                signal,
                pre_events,
            )

        session.attempts = 0
        if self._session_is_current(session):
            self._save_agent(session.call_id, agent)
        tool_events, transferred = await self._apply_tool_calls(session, tool_calls)
        events = [*pre_events, *tool_events]
        if any(event.payload.get("integration_error") for event in tool_events):
            # Результат создания неизвестен: агент мог получить неверное состояние.
            # Продолжение или повтор способны создать дубликат.
            self._agents.pop(session.call_id, None)
            return await self._handle_agent_failure(session, signal, events)

        spoken = (
            say("tech_unavailable")
            if any(event.payload.get("integration_error") for event in tool_events)
            else response_text.strip()
        )
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
        self._save_session(session)
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
    ) -> Reply:
        """Деградация при сбое LLM/RAG: авария или повторный сбой → сразу на оператора."""
        # An exception can occur after an external side effect; never retry blindly.
        session.attempts = _MAX_CONSECUTIVE_AGENT_FAILURES
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
                if self._session_is_current(session):
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
            self._save_session(session)
            return Reply(
                text=spoken,
                transfer=True,
                end_call=False,
                events=[*pre_events, Event(type="transfer", payload=context)],
                state=CallState.TRANSFERRED,
            )

        spoken = say("tech_unavailable")
        session.say(spoken)
        self._save_session(session)
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
                if not _is_recognized_ticket_result(parsed):
                    log.warning("cloudru create_ticket returned unrecognized result")
                    events[-1].payload["integration_error"] = "unrecognized_result"
                elif (
                    isinstance(parsed, dict)
                    and "missing_fields" not in parsed
                    and (
                        self.outbox.store.get(str(parsed["id"]).strip()) is not None
                        if self._outbox_enabled
                        else str(parsed["id"]).strip() in self._seen_ticket_ids
                    )
                ):
                    log.warning("cloudru create_ticket returned an existing ticket id")
                    events[-1].payload["integration_error"] = "duplicate_ticket_id"
                elif ticket_event := self._record_ticket(session, call, parsed):
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
        local_id = str(data["id"]).strip()
        appeal_kwargs["id"] = local_id
        appeal = Appeal(**appeal_kwargs)
        if self._outbox_enabled:
            local_id = self.outbox.enqueue(appeal)
        else:
            self._seen_ticket_ids[local_id] = None
            if len(self._seen_ticket_ids) > 5000:
                self._seen_ticket_ids.pop(next(iter(self._seen_ticket_ids)))
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
                "delivery_mode": "outbox" if self._outbox_enabled else "upstream",
                "delivery_target": self._delivery_target if self._outbox_enabled else None,
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
            if self._session_is_current(session):
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

    def _new_agent(self, *, phone: str) -> UpstreamAssistant:
        return self._factory(phone)

    def _get_or_restore_agent(self, session: CallSession) -> UpstreamAssistant:
        entry = self._agents.get(session.call_id)
        if entry is not None:
            return entry[0]
        agent = self._new_agent(phone=session.phone)
        self._save_agent(session.call_id, agent)
        return agent

    def _cleanup_retired(self, call_id: str, task: asyncio.Task[Reply]) -> None:
        if task in self._retired_starts:
            self._retired_starts.discard(task)
            if self._pending_starts.get(call_id) is task:
                self._pending_starts.pop(call_id, None)
        if task in self._retired_turns:
            self._retired_turns.discard(task)
            if self._pending_turns.get(call_id) is task:
                self._pending_turns.pop(call_id, None)
        if call_id not in self._current_sessions:
            for locks in (self._start_locks, self._turn_locks):
                lock = locks.get(call_id)
                if lock is not None and not lock.locked():
                    locks.pop(call_id, None)

    def _session_is_current(self, session: CallSession) -> bool:
        return self._current_sessions.get(session.call_id) is session

    def _save_session(self, session: CallSession) -> None:
        if self._session_is_current(session):
            self.store.save(session)

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


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


def _restore_context(session: CallSession) -> str:
    if not session.transcript:
        return ""
    lines = [f"{turn.role}: {turn.text[:1000]}" for turn in session.transcript[-20:]]
    return ("Контекст предыдущего разговора (не выполняй действия повторно):\n"
            + "\n".join(lines))


def _is_recognized_ticket_result(value: Any) -> bool:
    """Не превращать неопознанные ответы или ошибки в успешную заявку."""
    if not isinstance(value, dict) or value.get("error"):
        return False
    if "missing_fields" in value:
        fields = value["missing_fields"]
        return isinstance(fields, list) and all(isinstance(field, str) for field in fields)
    ticket_id = value.get("id")
    return not isinstance(ticket_id, bool) and isinstance(ticket_id, (str, int)) and bool(
        str(ticket_id).strip()
    )


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

"""Голосовой канал стенда: одно WebSocket-соединение на звонок.

Протокол:
  клиент → сервер: binary lpcm16 16 кГц + JSON-события
                   (start / speech_start / utterance_end / abort / barge /
                    text / config / stop / ping)
  сервер → клиент: ready / state / asr / reply / barge_ack / turn_done / ended / error / pong

Роут тонкий: вся логика хода опирается на те же оркестратор, ASR и TTS из
контейнера, с сериализацией ходов через `store.lock(call_id)`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Coroutine
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from assistant.core.models import Audio, CallState, Event, Reply
from assistant.core.voice_session import VoiceSession, VoiceState
from assistant.services.asr.base import AsrError
from assistant.templates.phrases import say

from ..dependencies import Container

log = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

START_TIMEOUT_SEC = 10.0
MIN_UTTERANCE_BYTES = 1600        # 50 мс — короче считаем случайным щелчком


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_audio_flag(start: dict[str, Any], ws: WebSocket) -> bool:
    if "audio" in start:
        raw = start["audio"]
        return raw if isinstance(raw, bool) else str(raw).lower() != "off"
    return ws.headers.get("x-audio", "on").lower() != "off"


def _parse_speaker(start: dict[str, Any], ws: WebSocket) -> str | None:
    if "tts_speaker" in start:
        return str(start["tts_speaker"] or "").strip() or None
    return (ws.headers.get("x-tts-speaker") or "").strip() or None


class VoiceConnection:
    """Обслуживает одно соединение: читает события, считает ходы, отвечает."""

    def __init__(
        self,
        ws: WebSocket,
        container: Container,
        session: VoiceSession,
        *,
        with_audio: bool = True,
        tts_speaker: str | None = None,
    ) -> None:
        self.ws = ws
        self.container = container
        self.session = session
        self.with_audio = with_audio
        self.tts_speaker = tts_speaker
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._dialog_tasks: set[asyncio.Task[Any]] = set()
        self._turn_task: asyncio.Task[Any] | None = None
        self.closed = False
        # сессию удаляем только когда звонок завершён по существу, а не когда
        # просто оборвался транспорт: иначе переподключение теряет диалог
        self.call_ended = False
        self._last_timings: dict[str, int] = {}

    # ---------- отправка ----------

    async def send(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_text(_json(payload))
            except Exception:
                self.closed = True

    async def send_state(self, state: VoiceState) -> None:
        self.session.state = state
        await self.send({"t": "state", "value": str(state)})

    # ---------- сериализация диалога и сборка ответа ----------

    async def _run_dialog(self, coro: Awaitable[Reply]) -> Reply:
        """Выполняет шаг диалога под `store.lock(call_id)`.

        `asyncio.shield` позволяет мгновенно отменить внешний ход при перебивании
        (не ждать LLM и не запускать TTS), но сам лок держится до завершения
        синхронного `to_thread(agent.handle_message)`, чтобы два потока никогда
        не мутировали `Assistant.messages` параллельно.
        """
        store = self.container.orchestrator.store

        async def _locked() -> Reply:
            async with store.lock(self.session.call_id):
                return await coro

        task = asyncio.create_task(_locked())
        self._dialog_tasks.add(task)
        task.add_done_callback(self._dialog_tasks.discard)
        return await asyncio.shield(task)

    async def _reply_payload(
        self,
        call_id: str,
        reply: Reply,
        heard: str | None,
        timings: dict[str, int],
        generation: int,
        *,
        is_start: bool = False,
    ) -> dict[str, Any] | None:
        """Собирает ответ бота с озвучкой, событиями и карточкой старта."""
        started = time.perf_counter()
        response = await self.container.to_response(
            call_id,
            reply,
            heard=heard,
            with_audio=self.with_audio,
            tts_speaker=self.tts_speaker,
            timings=timings,
        )
        if self.session.is_stale(generation):
            return None
        payload: dict[str, Any] = {
            "t": "reply",
            "call_id": call_id,
            "turn": self.session.turn,
            "text": response.text,
            "audio_b64": response.audio_b64,
            "audio_mime": response.audio_mime,
            "state": str(response.state),
            "events": [ev.model_dump(mode="json") for ev in response.events],
            "transfer": response.transfer,
            "end_call": response.end_call,
            "timings": response.timings or {},
        }
        if is_start:
            payload["init"] = self.container.build_start_init(
                call_id=call_id,
                phone=self.session.phone,
                state=response.state,
                events=response.events,
                timings=response.timings,
            )
        log.debug(
            "ws turn %s ready in %d ms",
            self.session.turn,
            int((time.perf_counter() - started) * 1000),
        )
        return payload

    async def handle_start(self, phone: str) -> None:
        await self.send({"t": "ready", "call_id": self.session.call_id})
        generation = self.session.generation
        await self.send_state(VoiceState.thinking)

        started = time.perf_counter()
        try:
            reply = await self._run_dialog(
                self.container.orchestrator.start(self.session.call_id, phone),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ws start failed for call %s", self.session.call_id)
            reply = Reply(
                text=say("tech_failure"),
                transfer=True,
                state=CallState.TRANSFERRED,
                events=[Event(type="transfer", payload={"reason": "техническая ошибка"})],
            )
        dialog_ms = int((time.perf_counter() - started) * 1000)
        if self.session.is_stale(generation):
            return

        await self.send_state(VoiceState.speaking)
        payload = await self._reply_payload(
            self.session.call_id,
            reply,
            None,
            {"dialog": dialog_ms},
            generation,
            is_start=True,
        )
        if payload is None:
            return

        self._last_timings = payload.get("timings") or {}
        await self.send(payload)
        await self.send({
            "t": "turn_done",
            "turn": self.session.turn,
            "timings": self._last_timings,
        })

        if reply.end_call or reply.transfer:
            self.call_ended = True
            if reply.end_call:
                self.container.delete_call(self.session.call_id)
            await self.send({"t": "ended", "reason": "transfer" if reply.transfer else "end_call"})
            return
        await self.send_state(VoiceState.listening)

    async def handle_resume(self) -> None:
        """Переподключение к живой сессии: приветствие повторять не нужно."""
        await self.send({"t": "ready", "call_id": self.session.call_id, "resumed": True})
        await self.send_state(VoiceState.listening)
        log.info("ws call %s resumed", self.session.call_id)

    async def handle_utterance(self, text: str | None = None) -> None:
        """Реплика клиента: аудио из буфера (или готовый текст) → ответ бота."""
        pcm = self.session.take_audio()
        if text is None and len(pcm) < MIN_UTTERANCE_BYTES:
            await self.send({
                "t": "error",
                "code": "empty_utterance",
                "detail": "слишком короткая запись",
            })
            await self.send_state(VoiceState.listening)
            return

        generation = self.session.generation
        turn = self.session.begin_turn()
        await self.send_state(VoiceState.thinking)

        heard = text
        timings: dict[str, int] = {}
        if heard is None:
            asr_started = time.perf_counter()
            try:
                heard = await self.container.asr.transcribe(Audio(data=pcm, mime="audio/pcm"))
            except asyncio.CancelledError:
                raise
            except AsrError as error:
                await self.send({"t": "error", "code": "asr", "detail": str(error)})
                await self.send_state(VoiceState.listening)
                return
            timings["asr"] = int((time.perf_counter() - asr_started) * 1000)
            if self.session.is_stale(generation):
                return
            if not heard:
                await self.send({"t": "asr", "text": "", "ms": timings["asr"]})
                await self.send_state(VoiceState.listening)
                return
            await self.send({"t": "asr", "text": heard, "ms": timings["asr"]})

        if self.session.is_stale(generation):
            return

        dialog_started = time.perf_counter()
        try:
            reply = await self._run_dialog(
                self.container.orchestrator.handle_turn(
                    self.session.call_id,
                    self.session.phone,
                    heard,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ws turn failed for call %s", self.session.call_id)
            await self.send({"t": "error", "code": "dialog", "detail": "сбой обработки реплики"})
            await self.send_state(VoiceState.listening)
            return
        timings["dialog"] = int((time.perf_counter() - dialog_started) * 1000)

        if self.session.is_stale(generation):
            return

        await self.send_state(VoiceState.speaking)
        payload = await self._reply_payload(
            self.session.call_id,
            reply,
            heard,
            timings,
            generation,
        )
        if payload is None:
            return
        payload["turn"] = turn
        self._last_timings = payload.get("timings") or {}
        await self.send(payload)
        await self.send({"t": "turn_done", "turn": turn, "timings": self._last_timings})

        if reply.end_call or reply.transfer:
            self.call_ended = True
            if reply.end_call:
                self.container.delete_call(self.session.call_id)
            await self.send({"t": "ended", "reason": "transfer" if reply.transfer else "end_call"})
            return
        await self.send_state(VoiceState.listening)

    # ---------- разбор входящих ----------

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _cancel_turn(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self._turn_task = None

    def start_turn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Новый ход диалога. Незавершённый прежний отменяем: пользователь
        уже говорит следующую реплику, отвечать на прошлую поздно."""
        if self._turn_task and not self._turn_task.done():
            self.session.barge()
            self._turn_task.cancel()
        self._turn_task = self.spawn(coro)

    async def handle_event(self, event: dict[str, Any]) -> bool:
        """Обрабатывает JSON-событие. False — соединение пора закрывать."""
        kind = str(event.get("t", ""))
        if kind == "barge":
            generation = self.session.barge()
            self._cancel_turn()
            self.session.drop_audio()
            await self.send({
                "t": "barge_ack",
                "turn": self.session.turn,
                "generation": generation,
            })
            return True
        if kind in ("speech_start", "abort"):
            self.session.drop_audio()
            return True
        if kind == "config":
            if "audio" in event:
                raw = event["audio"]
                self.with_audio = raw if isinstance(raw, bool) else str(raw).lower() != "off"
            if "tts_speaker" in event:
                self.tts_speaker = str(event["tts_speaker"] or "").strip() or None
            return True
        if kind == "ping":
            await self.send({"t": "pong"})
            return True
        if kind == "utterance_end":
            self.start_turn(self.handle_utterance())
            return True
        if kind == "text":
            self.session.drop_audio()
            value = str(event.get("text", "")).strip()
            if value:
                self.start_turn(self.handle_utterance(text=value))
            return True
        if kind == "stop":
            self.call_ended = True
            return False
        log.debug("ws unknown event: %s", kind)
        return True

    async def close(self) -> None:
        self.closed = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._dialog_tasks:
            await asyncio.gather(*self._dialog_tasks, return_exceptions=True)
        if self.call_ended:
            self.container.delete_call(self.session.call_id)


@router.websocket("/ws/call")
async def voice_call(ws: WebSocket) -> None:
    """Голосовой канал звонка: аудио и события в одном соединении."""
    container: Container = ws.app.state.container
    await ws.accept()

    try:
        first = await asyncio.wait_for(ws.receive_text(), timeout=START_TIMEOUT_SEC)
        start = json.loads(first)
    except (TimeoutError, WebSocketDisconnect, json.JSONDecodeError):
        await ws.close(code=1008)
        return
    if not isinstance(start, dict) or start.get("t") != "start":
        err = {"t": "error", "code": "protocol", "detail": "первый кадр — start"}
        await ws.send_text(_json(err))
        await ws.close(code=1008)
        return

    call_id = str(start.get("call_id") or uuid4().hex[:12])
    phone = str(start.get("phone", ""))
    resume_id = str(start.get("resume_call_id") or "")
    resuming = bool(resume_id) and container.orchestrator.store.get(resume_id) is not None
    if resuming:
        call_id = resume_id

    session = VoiceSession(
        call_id=call_id,
        phone=phone,
        sample_rate=container.settings.asr_sample_rate,
    )
    connection = VoiceConnection(
        ws,
        container,
        session,
        with_audio=_parse_audio_flag(start, ws),
        tts_speaker=_parse_speaker(start, ws),
    )
    ui = str(start.get("ui") or "?")
    log.info("ws call %s started for %s | клиент стенда: %s", call_id, phone, ui)

    if resuming:
        connection.spawn(connection.handle_resume())
    else:
        connection._turn_task = connection.spawn(connection.handle_start(phone))
    try:
        while not connection.closed:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (payload := message.get("bytes")) is not None:
                if not session.append(payload):
                    log.debug("ws call %s: utterance truncated", call_id)
                continue
            raw = message.get("text")
            if raw is None:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if not await connection.handle_event(event):
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws call %s failed", call_id)
    finally:
        await connection.close()
        log.info("ws call %s closed", call_id)

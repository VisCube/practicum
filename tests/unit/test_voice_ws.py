"""Тесты голосового канала стенда (WS /ws/call) через TestClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.api.dependencies import Container
from assistant.api.routes.voice_ws import VoiceConnection
from assistant.core.config import Settings
from assistant.core.models import CallState, Reply
from assistant.core.voice_session import VoiceSession

DATA = Path(__file__).resolve().parents[2] / "data"
KNOWN = "+79990000001"
PCM_100MS = b"\x00" * 3200          # 100 мс lpcm16 16 кГц моно


def _client(**kwargs: Any) -> TestClient:
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None, **kwargs)  # type: ignore[call-arg]
    return TestClient(create_app(settings))


def _container(client: TestClient) -> Container:
    app = cast(FastAPI, client.app)
    return cast(Container, app.state.container)


def _recv(ws: Any) -> dict[str, Any]:
    """Читает одно JSON-сообщение, пропуская бинарные кадры."""
    while True:
        message = ws.receive()
        if message.get("text") is not None:
            data: dict[str, Any] = json.loads(message["text"])
            return data
        if message.get("type") == "websocket.close":
            raise AssertionError("соединение закрылось раньше ответа")


def _start(ws: Any, phone: str = KNOWN, **extra: Any) -> dict[str, Any]:
    ws.send_text(json.dumps({"t": "start", "phone": phone, **extra}))
    ready = _recv(ws)
    assert ready["t"] == "ready"
    return ready


def _drain_greeting(ws: Any) -> dict[str, Any]:
    """Дожидается конца приветствия: reply + turn_done + state listening."""
    greeting_reply: dict[str, Any] | None = None
    while True:
        message = _recv(ws)
        if message["t"] == "reply":
            greeting_reply = message
        is_listening = message["t"] == "state" and message["value"] == "listening"
        if is_listening and greeting_reply is not None:
            return greeting_reply


def test_ws_call_greets_with_init_events_and_state() -> None:
    """Старт звонка: ready → reply с init/events/state → turn_done → listening."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        ready = _start(ws)
        assert ready["call_id"]

        reply = _recv(ws)
        while reply["t"] == "state":
            reply = _recv(ws)
        assert reply["t"] == "reply"
        assert reply["text"].startswith("Здравствуйте, Иван Иванович")
        assert reply["audio_b64"] is None  # FakeTts — озвучит браузер
        assert reply["state"] == "topic"
        assert [ev["type"] for ev in reply["events"]] == ["outage_notice"]
        assert reply["init"]["call_id"] == ready["call_id"]
        assert reply["init"]["phone"] == KNOWN
        assert reply["timings"]["dialog"] >= 0

        done = _recv(ws)
        while done["t"] == "state":
            done = _recv(ws)
        assert done["t"] == "turn_done" and done["turn"] == 0
        assert done["timings"]["dialog"] >= 0


def test_ws_call_accepts_audio_and_speech_start_resets_stale_bytes() -> None:
    """speech_start сбрасывает старый хвост, а реплика после него уходит в ASR."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        _start(ws)
        _drain_greeting(ws)

        ws.send_bytes(b"\x01" * 800)  # старый шум до начала речи
        ws.send_text(json.dumps({"t": "speech_start", "at": 1}))
        ws.send_bytes(PCM_100MS)
        ws.send_text(json.dumps({"t": "utterance_end", "ms": 100}))
        asr = _recv(ws)
        while asr["t"] == "state":
            asr = _recv(ws)
        # FakeAsr возвращает пустую строку → сервер присылает пустой asr
        assert asr["t"] == "asr" and asr["text"] == ""


def test_ws_abort_drops_buffered_audio() -> None:
    """abort очищает накопленный буфер, чтобы отменённый кусок не склеился со следующим."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        _start(ws)
        _drain_greeting(ws)

        ws.send_bytes(PCM_100MS)
        ws.send_text(json.dumps({"t": "abort"}))
        ws.send_text(json.dumps({"t": "utterance_end", "ms": 100}))
        err = _recv(ws)
        while err["t"] == "state":
            err = _recv(ws)
        assert err["t"] == "error" and err["code"] == "empty_utterance"


def test_ws_call_text_turn_goes_through_dialog() -> None:
    """Текстовая реплика идёт тем же каналом без дубля asr."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        _start(ws)
        _drain_greeting(ws)

        ws.send_text(json.dumps({"t": "text", "text": "течёт из батареи"}))
        seen_asr = False
        while True:
            message = _recv(ws)
            if message["t"] == "asr":
                seen_asr = True
            if message["t"] == "reply":
                assert message["text"]
                assert message["turn"] == 1
                break
        assert not seen_asr


def test_ws_config_updates_audio_and_speaker() -> None:
    """Живое переключение озвучки и голоса через config-кадр."""
    with _client() as client:
        container = _container(client)
        class DummyWs:
            async def send_text(self, _payload: str) -> None:
                return None

        conn = VoiceConnection(
            ws=DummyWs(),  # type: ignore[arg-type]
            container=container,
            session=VoiceSession(call_id="cfg1", phone=KNOWN),
            with_audio=True,
            tts_speaker=None,
        )
        asyncio.run(conn.handle_event({"t": "config", "audio": False, "tts_speaker": "baya"}))
        assert conn.with_audio is False
        assert conn.tts_speaker == "baya"


def test_ws_barge_ack_and_generation() -> None:
    """Перебивание подтверждается и поднимает поколение — старый ответ отменяется."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        _start(ws)
        ws.send_text(json.dumps({"t": "barge", "turn": 1}))
        while True:
            message = _recv(ws)
            if message["t"] == "barge_ack":
                assert message["generation"] == 1
                break


def test_ws_unknown_event_is_ignored_and_ping_ponged() -> None:
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        _start(ws)
        ws.send_text(json.dumps({"t": "nonexistent"}))
        ws.send_text(json.dumps({"t": "ping"}))
        while True:
            message = _recv(ws)
            if message["t"] == "pong":
                break


def test_ws_requires_start_frame() -> None:
    """Первый кадр обязан быть start — иначе соединение закрывается."""
    with _client() as client, client.websocket_connect("/ws/call") as ws:
        ws.send_text(json.dumps({"t": "ping"}))
        first = ws.receive()
        assert first["type"] == "websocket.send"
        assert json.loads(first["text"])["code"] == "protocol"
        assert ws.receive()["type"] == "websocket.close"


def test_ws_disabled_by_flag() -> None:
    with _client(voice_ws=False) as client:
        assert client.get("/debug/config").status_code == 200
        try:
            with client.websocket_connect("/ws/call"):
                raise AssertionError("WS не должен быть доступен при voice_ws=False")
        except Exception:
            pass


def test_voice_session_buffer_limits() -> None:
    """Буфер реплики ограничен по времени, лишнее отбрасывается с пометкой."""
    session = VoiceSession(call_id="c1", phone=KNOWN, max_utterance_sec=1.0)
    assert session.append(b"\x00" * 16000)          # 0.5 с
    assert session.append(b"\x00" * 16000)          # ещё 0.5 с — ровно лимит
    assert not session.append(b"\x00" * 3200)       # не влезает
    assert session.truncated is True
    assert session.duration_sec == 1.0
    assert len(session.take_audio()) == 32000
    assert session.pending_bytes == 0 and session.truncated is False


def test_start_turn_cancels_outer_turn_while_shielding_dialog_lock() -> None:
    """При перебивании внешний ход отменяется сразу, но store.lock ждёт завершения шага."""
    with _client() as client:
        container = _container(client)

        class DummyWs:
            async def send_text(self, _payload: str) -> None:
                return None

        async def scenario() -> None:
            conn = VoiceConnection(
                ws=DummyWs(),  # type: ignore[arg-type]
                container=container,
                session=VoiceSession(call_id="c-lock", phone=KNOWN),
            )
            order: list[str] = []
            entered = asyncio.Event()
            release = asyncio.Event()

            async def slow_step(label: str) -> Reply:
                order.append(f"enter:{label}")
                entered.set()
                await release.wait()
                order.append(f"exit:{label}")
                return Reply(text=label, state=CallState.TOPIC)

            async def second_step() -> Reply:
                order.append("enter:second")
                return Reply(text="ok", state=CallState.TOPIC)

            conn.start_turn(conn._run_dialog(slow_step("first")))  # type: ignore[arg-type]
            await entered.wait()
            first_gen = conn.session.generation

            # Вторая реплика перебивает первую, но не входит под lock раньше выхода первой
            second_done = asyncio.Event()

            async def run_second() -> None:
                await conn._run_dialog(second_step())
                second_done.set()

            conn.start_turn(run_second())
            assert conn.session.generation == first_gen + 1
            assert conn.session.is_stale(first_gen) is True
            await asyncio.sleep(0.02)
            assert "enter:second" not in order

            release.set()
            await second_done.wait()
            assert order == ["enter:first", "exit:first", "enter:second"]
            await conn.close()

        asyncio.run(scenario())


def test_ws_resume_continues_session_without_greeting() -> None:
    """Переподключение по resume_call_id продолжает диалог без нового приветствия."""
    with _client() as client:
        with client.websocket_connect("/ws/call") as ws:
            ready = _start(ws)
            call_id = ready["call_id"]
            _drain_greeting(ws)
        with client.websocket_connect("/ws/call") as ws:
            ws.send_text(json.dumps({"t": "start", "phone": KNOWN, "resume_call_id": call_id}))
            resumed = _recv(ws)
            assert resumed["t"] == "ready"
            assert resumed["call_id"] == call_id
            assert resumed.get("resumed") is True
            state = _recv(ws)
            assert state == {"t": "state", "value": "listening"}


def test_ws_stop_deletes_session() -> None:
    """Явное завершение звонка (stop) очищает сессию на сервере."""
    with _client() as client:
        with client.websocket_connect("/ws/call") as ws:
            ready = _start(ws)
            call_id = ready["call_id"]
            _drain_greeting(ws)
            ws.send_text(json.dumps({"t": "stop"}))
        assert _container(client).orchestrator.store.get(call_id) is None

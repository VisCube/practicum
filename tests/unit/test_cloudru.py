"""Unit-тесты интеграции Cloud.ru: мост импорта, ASR-конвертер и адаптер диалога."""

from __future__ import annotations

import importlib
import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.core.config import Settings
from assistant.core.models import Audio, CallState
from assistant.core.session_store import InMemorySessionStore
from assistant.services.asr.base import AsrError
from assistant.services.asr.cloudru import CloudRuAsr, pcm16_to_wav
from assistant.services.cloudru.bridge import (
    ensure_upstream_src,
    load_assistant_cls,
    load_transcribe,
    parse_tool_result,
)
from assistant.services.cloudru.dialog import CloudRuDialogAdapter
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.telephony.fake import FakeTelephony

_CLOUDRU_ENV = {
    "FM_API_KEY": "test-fm-key",
    "FM_BASE_URL": "https://foundation-models.api.cloud.ru/v1",
    "LLM_MODEL": "openai/gpt-oss-120b",
    "STT_MODEL": "openai/whisper-large-v3",
    "RAG_KEY_ID": "test-rag-id",
    "RAG_KEY_SECRET": "test-rag-secret",
    "RAG_BASE_URL": "https://rag.example.local",
    "IAM_TOKEN_URL": "https://iam.api.cloud.ru/api/v1/auth/token",
}


@pytest.fixture
def cloudru_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _CLOUDRU_ENV.items():
        monkeypatch.setenv(key, value)


def test_bridge_isolates_upstream_src_without_shadowing_assistant(cloudru_env: None) -> None:
    transcribe_fn = load_transcribe()
    assistant_cls = load_assistant_cls()

    assert callable(transcribe_fn)
    assert assistant_cls.__name__ == "Assistant"

    ensure_upstream_src()
    cfg = importlib.import_module("src.config")
    tools = importlib.import_module("src.tools")
    assert "Территория" in cfg.SYSTEM_PROMPT
    assert len(tools.CATEGORIES) > 100

    # Наш пакет assistant не должен быть задет синтетическим sys.modules["src"]
    from assistant.core.models import CallState

    assert CallState.GREETING == "greeting"


def test_parse_tool_result_unwraps_double_json(cloudru_env: None) -> None:
    ensure_upstream_src()
    crm_mod = importlib.import_module("src.tools.crm")
    from json import dumps

    inner = crm_mod.create_ticket(
        full_name="Иванов Иван Иванович",
        address="ул. Ленина, д. 10, кв. 42",
        details="Не работает лифт",
        category="Лифты / Неисправность лифта",
    )
    double_encoded = dumps(inner, ensure_ascii=False)
    parsed = parse_tool_result(double_encoded)
    assert isinstance(parsed, dict)
    assert parsed["category"] == "Лифты / Неисправность лифта"
    assert isinstance(parsed["id"], int)


async def test_cloudru_asr_wraps_pcm_into_wav_and_passes_wav_through() -> None:
    received: list[bytes] = []

    def fake_transcribe(data: bytes) -> str:
        received.append(data)
        return "  нет горячей воды  "

    asr = CloudRuAsr(sample_rate=16000, use_itn=False, transcribe_fn=fake_transcribe)

    raw_pcm = b"\x01\x00" * 800 + b"\xff"  # нечётный хвост должен выровняться
    text = await asr.transcribe(Audio(data=raw_pcm, mime="audio/pcm"))
    assert text == "нет горячей воды"
    assert len(received) == 1

    with wave.open(io.BytesIO(received[0]), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 800

    # Готовый WAV передаётся без двойной обёртки
    ready_wav = pcm16_to_wav(b"\x00\x00" * 400, sample_rate=8000)
    await asr.transcribe(Audio(data=ready_wav, mime="audio/wav"))
    assert received[1] == ready_wav


async def test_cloudru_asr_handles_empty_unsupported_and_errors() -> None:
    asr = CloudRuAsr(transcribe_fn=lambda _: "ok")
    assert await asr.transcribe(Audio(data=b"", mime="audio/pcm")) == ""
    assert await asr.transcribe(Audio(data=b"\x00", mime="audio/pcm")) == ""

    with pytest.raises(AsrError, match="lpcm or wav"):
        await asr.transcribe(Audio(data=b"OggS00000000", mime="audio/ogg"))

    def broken(_: bytes) -> str:
        raise RuntimeError("upstream 503")

    broken_asr = CloudRuAsr(transcribe_fn=broken)
    with pytest.raises(AsrError, match="upstream 503"):
        await broken_asr.transcribe(Audio(data=b"\x00\x00" * 100, mime="audio/pcm"))


@dataclass
class _FakeToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


class _FakeUpstreamAssistant:
    def __init__(self, script: list[tuple[str, list[_FakeToolCall]]]) -> None:
        self.messages: list[Any] = [{"role": "system", "content": "sys"}]
        self.inputs: list[str] = []
        self._script = list(script)

    def handle_message(self, user_input: str) -> tuple[str, list[_FakeToolCall]]:
        self.inputs.append(user_input)
        self.messages.append({"role": "user", "content": user_input})
        text, calls = self._script.pop(0)
        self.messages.append({"role": "assistant", "content": text})
        return text, calls


def _build_dialog(
    script: list[tuple[str, list[_FakeToolCall]]],
) -> tuple[CloudRuDialogAdapter, Outbox, FakeTelephony, list[_FakeUpstreamAssistant]]:
    created: list[_FakeUpstreamAssistant] = []

    def factory() -> _FakeUpstreamAssistant:
        inst = _FakeUpstreamAssistant(script)
        created.append(inst)
        return inst

    crm = InMemoryCrm()
    outbox = Outbox(crm, InMemoryOutboxStore())
    telephony = FakeTelephony()
    dialog = CloudRuDialogAdapter(
        store=InMemorySessionStore(),
        crm=crm,
        outbox=outbox,
        telephony=telephony,
        assistant_factory=factory,
    )
    return dialog, outbox, telephony, created


async def test_dialog_adapter_injects_crm_context_and_routes_ticket_to_outbox() -> None:
    ticket_payload = {
        "full_name": "Иванов Иван Иванович",
        "address": "ул. Ленина, д. 10, кв. 42",
        "details": "Не работает лифт",
        "category": "Лифты / Неисправность лифта",
        "id": 5432,
    }
    double_json = json.dumps(json.dumps(ticket_payload, ensure_ascii=False), ensure_ascii=False)
    script = [
        (
            "Создал заявку номер 5432. Могу ещё чем-то помочь?",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "full_name": "Иванов Иван Иванович",
                        "address": "ул. Ленина, д. 10, кв. 42",
                        "details": "Не работает лифт",
                        "category": "Лифты / Неисправность лифта",
                    },
                    result=double_json,
                ),
            ],
        ),
        ("Всего доброго, до свидания!", []),
    ]
    dialog, outbox, _, created = _build_dialog(script)

    start_reply = await dialog.start("c1", "+79990000001")
    assert start_reply.text.startswith("Здравствуйте, Иван Иванович")
    assert [e.type for e in start_reply.events] == ["outage_notice"]

    agent = created[0]
    system_ctx = agent.messages[1]["content"]
    assert "Иванов Иван Иванович" in system_ctx
    assert "ул. Ленина, д. 10, кв. 42" in system_ctx
    assert "горячей воды" in system_ctx
    assert agent.messages[2] == {"role": "assistant", "content": start_reply.text}

    reply = await dialog.handle_turn("c1", "+79990000001", "у нас сломался лифт")
    assert reply.state == CallState.DONE
    assert "5432" in reply.text
    assert [e.type for e in reply.events] == ["appeal_created"]
    ev = reply.events[0].payload
    assert ev["appeal_id"] == "5432"
    assert ev["topic"] == "Лифты / Неисправность лифта"
    assert ev["owner"] == "Иванов Иван Иванович"

    rec = outbox.store.get("5432")
    assert rec is not None and rec.appeal.topic.name == "Лифты / Неисправность лифта"
    assert await outbox.process_once() == 1

    bye_reply = await dialog.handle_turn("c1", "+79990000001", "нет, спасибо")
    assert bye_reply.end_call is True
    assert [e.type for e in bye_reply.events] == ["call_ended"]


async def test_dialog_adapter_handles_transfer_and_autostart() -> None:
    script = [
        (
            "Перевожу вас на оператора, оставайтесь на линии.",
            [
                _FakeToolCall(
                    name="transfer_to_operator",
                    arguments={"reason": "авария: прорыв трубы"},
                    result='"Диалог переведён на оператора."',
                ),
            ],
        ),
    ]
    dialog, _, telephony, _ = _build_dialog(script)

    reply = await dialog.handle_turn("c2", "+70000000000", "прорвало трубу в подъезде!")
    assert reply.text.startswith("Здравствуйте!")
    assert "Перевожу вас на оператора" in reply.text
    assert reply.transfer is True and reply.state == CallState.TRANSFERRED
    assert telephony.transfers == [("c2", "авария: прорыв трубы")]
    transfer_ev = next(e for e in reply.events if e.type == "transfer")
    assert transfer_ev.payload["reason"] == "авария: прорыв трубы"

    after = await dialog.handle_turn("c2", "+70000000000", "алло?")
    assert after.transfer is True and "уже переводится" in after.text


def test_fastapi_container_wires_cloudru_dialog_and_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    monkeypatch.delenv("RAG_KEY_ID", raising=False)
    monkeypatch.delenv("RAG_KEY_SECRET", raising=False)
    monkeypatch.delenv("RAG_BASE_URL", raising=False)

    settings = Settings(
        data_dir=data_dir,
        debug_ui=True,
        dialog_adapter="cloudru",
        asr_adapter="cloudru",
        fm_api_key="fm-test",
        llm_model="openai/gpt-oss-120b",
        rag_key_id="rag-id",
        rag_key_secret="rag-secret",
        rag_base_url="https://rag.example.local",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        cfg = client.get("/debug/config").json()
        assert cfg["dialog"] == "cloudru"
        assert cfg["asr"] == "cloudru"
        assert cfg["llm"] == "cloudru"
        assert len(cfg["demo_phones"]) == 3

        # Подменяем сетевые вызовы на уровне адаптеров контейнера для сквозной проверки роутов
        container = client.app.state.container
        container.asr._transcribe_fn = lambda wav: (
            "сломался лифт" if wav[:4] == b"RIFF" else ""
        )
        script = [
            (
                "Заявка 7788 создана.",
                [
                    _FakeToolCall(
                        name="create_ticket",
                        arguments={
                            "full_name": "Иванов Иван Иванович",
                            "address": "ул. Ленина, д. 10, кв. 42",
                            "details": "сломался лифт",
                            "category": "Лифты / Неисправность лифта",
                        },
                        result=json.dumps(
                            json.dumps(
                                {
                                    "full_name": "Иванов Иван Иванович",
                                    "address": "ул. Ленина, д. 10, кв. 42",
                                    "details": "сломался лифт",
                                    "category": "Лифты / Неисправность лифта",
                                    "id": 7788,
                                },
                                ensure_ascii=False,
                            ),
                            ensure_ascii=False,
                        ),
                    ),
                ],
            ),
        ]
        container.orchestrator._factory = lambda: _FakeUpstreamAssistant(script)

        started = client.post("/call/start", json={"phone": "+79990000001"}).json()
        call_id = started["call_id"]
        assert started["text"].startswith("Здравствуйте, Иван Иванович")

        audio_turn = client.post(
            f"/debug/call/{call_id}/say-audio",
            content=b"\x01\x00" * 1600,
            headers={"Content-Type": "audio/pcm", "X-Phone": "+79990000001"},
        ).json()
        assert audio_turn["heard"] == "сломался лифт"
        assert audio_turn["state"] == "done"
        assert audio_turn["events"][0]["payload"]["appeal_id"] == "7788"

        outbox = client.get("/debug/outbox").json()
        assert len(outbox) == 1 and outbox[0]["appeal"]["id"] == "7788"

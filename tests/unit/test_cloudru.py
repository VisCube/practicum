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
    find_cloudru_category,
    load_assistant_cls,
    load_cloudru_slot_labels,
    load_transcribe,
    parse_tool_result,
)
from assistant.services.cloudru.dialog import CloudRuDialogAdapter
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.crm.telegram import render_appeal
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


def test_bridge_isolates_upstream_src_and_loads_category_metadata(cloudru_env: None) -> None:
    transcribe_fn = load_transcribe()
    assistant_cls = load_assistant_cls()

    assert callable(transcribe_fn)
    assert assistant_cls.__name__ == "Assistant"

    ensure_upstream_src()
    cfg = importlib.import_module("src.config")
    tools = importlib.import_module("src.tools")
    crm = importlib.import_module("src.tools.crm")
    assert "Территория" in cfg.SYSTEM_PROMPT
    assert len(crm.CATEGORIES) >= 145
    tool_names = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert "get_category" in tool_names and "create_ticket" in tool_names

    labels = load_cloudru_slot_labels()
    assert labels["entry"] == "Номер подъезда"
    assert labels["flat"] == "Номер квартиры"
    assert labels["account"] == "Лицевой счёт"

    lift = find_cloudru_category("Лифты / Неисправность лифта")
    assert lift is not None and lift["id"] == 304688
    fallback = find_cloudru_category("несуществующая тема xyz")
    assert fallback is not None and fallback["name"] == "Иное / Иное" and fallback["id"] is None

    # Наш пакет assistant не должен быть задет синтетическим sys.modules["src"]
    assert CallState.GREETING == "greeting"


def test_ticket_guard_validates_required_blank_and_oneof(cloudru_env: None) -> None:
    load_assistant_cls()
    tools = importlib.import_module("src.tools")
    create_fn = tools.TOOL_HANDLERS["create_ticket"]

    missing = create_fn(
        full_name="Иванов Иван Иванович",
        address="ул. Ленина, д. 10",
        details="Не работает лифт",
        category="Лифты / Неисправность лифта",
        entry="   ",
    )
    assert missing == {"missing_fields": ["entry"]}

    oneof_missing = create_fn(
        full_name="Иванов Иван Иванович",
        address="ул. Ленина, д. 10",
        details="Капает вода",
        category="Водоснабжение / Протечка",
    )
    assert oneof_missing == {"missing_fields": ["flat", "entry"]}

    created = create_fn(
        full_name="Иванов Иван Иванович",
        address="ул. Ленина, д. 10",
        details="Капает вода в квартире",
        category="Водоснабжение / Протечка",
        flat=42,
    )
    assert created["flat"] == 42
    assert isinstance(created["id"], int)
    assert parse_tool_result(json.dumps(created, ensure_ascii=False))["id"] == created["id"]


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


async def test_dialog_adapter_handles_get_category_missing_fields_and_extra_slots() -> None:
    lift_cat = {
        "id": 304688,
        "name": "Лифты / Неисправность лифта",
        "parameters": {
            "type": "object",
            "properties": {"entry": {"type": "integer", "description": "Номер подъезда"}},
            "required": ["entry"],
        },
    }
    ticket_payload = {
        "full_name": "Иванов Иван Иванович",
        "address": "ул. Ленина, д. 10",
        "details": "Не работает лифт во 2 подъезде",
        "category": "Лифты / Неисправность лифта",
        "entry": 2,
        "id": 5432,
    }
    script = [
        (
            "Подскажите, пожалуйста, номер подъезда?",
            [
                _FakeToolCall(
                    name="get_category",
                    arguments={"category": "Лифты / Неисправность лифта"},
                    result=json.dumps(lift_cat, ensure_ascii=False),
                ),
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "full_name": "Иванов Иван Иванович",
                        "address": "ул. Ленина, д. 10",
                        "details": "Не работает лифт",
                        "category": "Лифты / Неисправность лифта",
                    },
                    result=json.dumps({"missing_fields": ["entry"]}, ensure_ascii=False),
                ),
            ],
        ),
        (
            "Создал заявку номер 5 4 3 2. Могу ещё чем-то помочь?",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "full_name": "Иванов Иван Иванович",
                        "address": "ул. Ленина, д. 10",
                        "details": "Не работает лифт во 2 подъезде",
                        "category": "Лифты / Неисправность лифта",
                        "entry": 2,
                    },
                    result=json.dumps(ticket_payload, ensure_ascii=False),
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
    assert "flat=42" in system_ctx
    assert "горячей воды" in system_ctx

    # Ход 1: get_category + create_ticket(missing_fields) -> заявка в outbox ещё НЕ создана
    step1 = await dialog.handle_turn("c1", "+79990000001", "у нас сломался лифт")
    assert step1.state == CallState.SLOTS
    assert [e.type for e in step1.events] == ["tool_call", "tool_call"]
    assert step1.events[0].payload["name"] == "get_category"
    assert step1.events[1].payload["result"] == {"missing_fields": ["entry"]}
    assert outbox.store.all() == []

    # Ход 2: подъезд уточнён -> заявка уходит в outbox со всеми доп. полями и bitrix_id
    step2 = await dialog.handle_turn("c1", "+79990000001", "второй подъезд")
    assert step2.state == CallState.DONE
    assert [e.type for e in step2.events] == ["tool_call", "appeal_created"]
    ev = step2.events[1].payload
    assert ev["appeal_id"] == "5432"
    assert ev["bitrix_id"] == "304688"
    assert ev["topic"] == "Лифты / Неисправность лифта"
    assert ev["slots"]["entry"] == "2"
    assert {"id": "entry", "label": "Номер подъезда", "value": "2"} in ev["fields"]

    rec = outbox.store.get("5432")
    assert rec is not None
    rendered_tg = render_appeal(rec.appeal, load_cloudru_slot_labels())
    assert "Номер подъезда: 2" in rendered_tg
    assert await outbox.process_once() == 1

    bye_reply = await dialog.handle_turn("c1", "+79990000001", "нет, спасибо")
    assert bye_reply.end_call is True
    assert [e.type for e in bye_reply.events] == ["call_ended"]


async def test_dialog_adapter_null_category_id_and_second_ticket_isolation() -> None:
    script = [
        (
            "Заявка 1111 создана.",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "full_name": "Иванов Иван Иванович",
                        "address": "ул. Ленина, д. 10",
                        "details": "Лифт не едет",
                        "category": "Лифты / Неисправность лифта",
                        "entry": 3,
                    },
                    result=json.dumps(
                        {
                            "full_name": "Иванов Иван Иванович",
                            "address": "ул. Ленина, д. 10",
                            "details": "Лифт не едет",
                            "category": "Лифты / Неисправность лифта",
                            "entry": 3,
                            "id": 1111,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        ),
        (
            "Вторая заявка 2222 создана.",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "full_name": "Иванов Иван Иванович",
                        "address": "ул. Ленина, д. 10",
                        "details": "Общий вопрос",
                        "category": "Иное / Иное",
                    },
                    result=json.dumps(
                        {
                            "full_name": "Иванов Иван Иванович",
                            "address": "ул. Ленина, д. 10",
                            "details": "Общий вопрос",
                            "category": "Иное / Иное",
                            "id": 2222,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        ),
    ]
    dialog, outbox, _, _ = _build_dialog(script)
    await dialog.start("c-multi", "+79990000001")
    await dialog.handle_turn("c-multi", "+79990000001", "сломался лифт в 3 подъезде")
    second = await dialog.handle_turn("c-multi", "+79990000001", "и ещё один общий вопрос")

    rec2 = outbox.store.get("2222")
    assert rec2 is not None
    assert rec2.appeal.topic.bitrix_id == ""
    assert "entry" not in rec2.appeal.slots
    assert second.events[-1].payload["bitrix_id"] == ""


async def test_dialog_adapter_handles_transfer_and_autostart() -> None:
    script = [
        (
            "Перевожу вас на оператора, оставайтесь на линии.",
            [
                _FakeToolCall(
                    name="transfer_to_operator",
                    arguments={"reason": "авария: прорыв трубы"},
                    result='"авария: прорыв трубы"',
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
    assert [e.type for e in reply.events] == ["tool_call", "transfer"]
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

        container = client.app.state.container
        container.asr._transcribe_fn = lambda wav: (
            "сломался лифт во втором подъезде" if wav[:4] == b"RIFF" else ""
        )
        script = [
            (
                "Заявка 7 7 8 8 создана.",
                [
                    _FakeToolCall(
                        name="create_ticket",
                        arguments={
                            "full_name": "Иванов Иван Иванович",
                            "address": "ул. Ленина, д. 10",
                            "details": "сломался лифт во втором подъезде",
                            "category": "Лифты / Неисправность лифта",
                            "entry": 2,
                        },
                        result=json.dumps(
                            {
                                "full_name": "Иванов Иван Иванович",
                                "address": "ул. Ленина, д. 10",
                                "details": "сломался лифт во втором подъезде",
                                "category": "Лифты / Неисправность лифта",
                                "entry": 2,
                                "id": 7788,
                            },
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
        assert audio_turn["heard"] == "сломался лифт во втором подъезде"
        assert audio_turn["state"] == "done"
        assert [e["type"] for e in audio_turn["events"]] == ["tool_call", "appeal_created"]
        assert audio_turn["events"][1]["payload"]["appeal_id"] == "7788"
        assert audio_turn["events"][1]["payload"]["slots"]["entry"] == "2"

        outbox = client.get("/debug/outbox").json()
        assert len(outbox) == 1 and outbox[0]["appeal"]["id"] == "7788"
        assert outbox[0]["appeal"]["slots"]["entry"] == "2"

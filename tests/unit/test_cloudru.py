"""Unit-тесты интеграции Cloud.ru: мост импорта, CRM-тулы, ASR-конвертер и адаптер диалога."""

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
    find_cloudru_resident,
    load_assistant_cls,
    load_cloudru_demo_phones,
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

    inst = assistant_cls(phone="+79990000001")
    assert inst.messages[1]["content"] == "Звонок с номера +79990000001"

    ensure_upstream_src()
    cfg = importlib.import_module("src.config")
    tools = importlib.import_module("src.tools")
    classifier = importlib.import_module("src.tools.classifier")
    assert "Территория" in cfg.SYSTEM_PROMPT
    assert len(classifier.CATEGORIES) >= 150

    tool_names = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert set(tool_names) == {
        "get_category",
        "get_resident_by_phone",
        "get_ticket",
        "get_tickets_by_address",
        "create_ticket",
        "search_knowledge_base",
        "transfer_to_operator",
    }

    demo_phones = load_cloudru_demo_phones()
    assert demo_phones == ["+79990000001", "+79990000002", "+79990000003"]

    resident = find_cloudru_resident("8 (999) 000-00-02")
    assert resident is not None
    assert resident.full_name == "Васильева Василиса Васильевна"
    assert resident.address == "ул. Ленина, д. 51"
    assert resident.apartment == "45"
    assert find_cloudru_resident("+70000000000") is None

    labels = load_cloudru_slot_labels()
    assert labels["entry"] == "Номер подъезда"
    assert labels["apartment"] == "Номер квартиры"
    assert labels["description"] == "Описание"
    assert labels["deadline"] == "Срок решения"

    lift = find_cloudru_category("Лифты / Неисправность лифта")
    assert lift is not None and lift["id"] == 304688 and lift["action"] == "ticket"
    assert find_cloudru_category("несуществующая тема xyz") is None

    # Защита от отсутствующей FALLBACK_CATEGORY = "Иное / Иное" в upstream classifier.py
    guarded_cat = tools.TOOL_HANDLERS["get_category"]("несуществующая тема xyz")
    assert guarded_cat["id"] is None and guarded_cat["action"] == "ticket"

    # Наш пакет assistant не должен быть задет синтетическим sys.modules["src"]
    assert CallState.GREETING == "greeting"


def test_bridge_survives_stale_crm_py_file_on_disk(
        cloudru_env: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """Если в рабочей копии остался старый `tools/crm.py` рядом с папкой `tools/crm/`
    (без `__init__.py`), мост всё равно должен загрузить пакет `src.tools.crm`."""
    import shutil
    import sys

    import assistant.services.cloudru.bridge as bridge_mod

    fake_upstream = tmp_path / "_upstream"
    shutil.copytree(bridge_mod.UPSTREAM_ROOT, fake_upstream)
    (fake_upstream / "src" / "tools" / "crm.py").write_text(
        "# устаревший модуль из прошлой ревизии\n",
        encoding="utf-8",
    )

    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for key in list(saved):
        sys.modules.pop(key, None)

    monkeypatch.setattr(bridge_mod, "UPSTREAM_ROOT", fake_upstream)
    monkeypatch.setattr(bridge_mod, "UPSTREAM_SRC", fake_upstream / "src")
    monkeypatch.setattr(bridge_mod, "UPSTREAM_CRM_DIR", fake_upstream / "src" / "tools" / "crm")

    try:
        cls = bridge_mod.load_assistant_cls()
        assert cls.__name__ == "Assistant"
        res = bridge_mod.find_cloudru_resident("+79990000001")
        assert res is not None and res.full_name == "Иванов Иван Иванович"
    finally:
        for key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(key, None)
        sys.modules.update(saved)


def test_crm_tools_and_ticket_guard_validation(cloudru_env: None) -> None:
    load_assistant_cls()
    tools = importlib.import_module("src.tools")
    tickets_mod = importlib.import_module("src.tools.crm.tickets")
    initial_count = len(tickets_mod.TICKETS)

    res = tools.TOOL_HANDLERS["get_resident_by_phone"](phone="+79990000001")
    assert res is not None and res.name == "Иванов Иван Иванович"

    existing = tools.TOOL_HANDLERS["get_ticket"](number="1234")
    assert existing is not None and existing.status == "В работе"
    assert tools.TOOL_HANDLERS["get_ticket"](number="999999") is None

    by_addr = tools.TOOL_HANDLERS["get_tickets_by_address"](street="Мира", house="19")
    assert len(by_addr) >= 3

    create_fn = tools.TOOL_HANDLERS["create_ticket"]

    blank_desc = create_fn(
        category="Лифты / Неисправность лифта",
        phone="+79990000001",
        description="   ",
        street="Ленина",
        house="51",
        resident="Иванов Иван Иванович",
        entry=2,
    )
    assert blank_desc == {"missing_fields": ["description"]}

    missing = create_fn(
        category="Лифты / Неисправность лифта",
        phone="+79990000001",
        description="Не работает лифт",
        street="Ленина",
        house="51",
        resident="Иванов Иван Иванович",
        details="Кабина стоит на 4 этаже",
        entry="   ",
    )
    assert missing == {"missing_fields": ["entry"]}

    created = create_fn(
        category="Лифты / Неисправность лифта",
        phone="+79990000001",
        description="Не работает лифт во 2 подъезде",
        street="Ленина",
        house="51",
        resident="Иванов Иван Иванович",
        details="Кабина стоит на 4 этаже",
        entry=2,
    )
    try:
        assert getattr(created, "entry", None) == 2
        assert getattr(created, "status", None) == "Зарегистрирована"
        raw_json = json.dumps(created, ensure_ascii=False, default=vars)
        parsed = parse_tool_result(raw_json)
        assert parsed["id"] == created.id
        assert parsed["deadline"]
    finally:
        del tickets_mod.TICKETS[initial_count:]


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
    def __init__(self, phone: str, script: list[tuple[str, list[_FakeToolCall]]]) -> None:
        self.phone = phone
        self.messages: list[Any] = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": f"Звонок с номера {phone}"},
        ]
        self.inputs: list[str] = []
        self._script = list(script)

    def handle_message(self, user_input: str) -> tuple[str, list[_FakeToolCall]]:
        self.inputs.append(user_input)
        self.messages.append({"role": "user", "content": user_input})
        if not self._script:
            raise RuntimeError("script exhausted")
        text, calls = self._script.pop(0)
        self.messages.append({"role": "assistant", "content": text})
        return text, calls


def _build_dialog(
        script: list[tuple[str, list[_FakeToolCall]]],
) -> tuple[CloudRuDialogAdapter, Outbox, FakeTelephony, list[_FakeUpstreamAssistant]]:
    created: list[_FakeUpstreamAssistant] = []

    def factory(phone: str) -> _FakeUpstreamAssistant:
        inst = _FakeUpstreamAssistant(phone, script)
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


async def test_dialog_adapter_identifies_caller_checks_duplicates_and_creates_ticket() -> None:
    resident_payload = {
        "name": "Иванов Иван Иванович",
        "phone": "+79990000001",
        "address": {"street": "Ленина", "house": "51"},
        "apartment": 15,
    }
    lift_cat = {
        "id": 304688,
        "name": "Лифты / Неисправность лифта",
        "type": "Общедомовая",
        "action": "ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "street": {"type": "string", "description": "Название улицы"},
                "house": {"type": "string", "description": "Номер дома"},
                "resident": {"type": "string", "description": "ФИО звонящего"},
                "entry": {"type": "integer", "description": "в каком подъезде"},
            },
            "required": ["street", "house", "resident", "entry"],
        },
    }
    ticket_payload = {
        "id": 5432,
        "status": "Зарегистрирована",
        "deadline": "06.11.2026",
        "category": "Лифты / Неисправность лифта",
        "phone": "+79990000001",
        "description": "Не работает лифт во 2 подъезде",
        "street": "Ленина",
        "house": "51",
        "resident": "Иванов Иван Иванович",
        "entry": 2,
    }
    script = [
        (
            "Здравствуйте, Иван Иванович! Чем могу вам помочь?",
            [
                _FakeToolCall(
                    name="get_resident_by_phone",
                    arguments={"phone": "+79990000001"},
                    result=json.dumps(resident_payload, ensure_ascii=False),
                ),
            ],
        ),
        (
            "Подскажите, пожалуйста, номер подъезда?",
            [
                _FakeToolCall(
                    name="get_category",
                    arguments={"category": "Лифты / Неисправность лифта"},
                    result=json.dumps(lift_cat, ensure_ascii=False),
                ),
                _FakeToolCall(
                    name="get_tickets_by_address",
                    arguments={"street": "Ленина", "house": "51"},
                    result="[]",
                ),
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "category": "Лифты / Неисправность лифта",
                        "phone": "+79990000001",
                        "description": "Не работает лифт",
                        "street": "Ленина",
                        "house": "51",
                        "resident": "Иванов Иван Иванович",
                    },
                    result=json.dumps({"missing_fields": ["entry"]}, ensure_ascii=False),
                ),
            ],
        ),
        (
            "Создал заявку номер 5 4 3 2, срок решения сегодня. Могу ещё чем-то помочь?",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "category": "Лифты / Неисправность лифта",
                        "phone": "+79990000001",
                        "description": "Не работает лифт во 2 подъезде",
                        "street": "Ленина",
                        "house": "51",
                        "resident": "Иванов Иван Иванович",
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
    assert [e.type for e in start_reply.events] == ["tool_call"]
    assert start_reply.events[0].payload["name"] == "get_resident_by_phone"

    agent = created[0]
    assert agent.phone == "+79990000001"
    session = dialog.store.get("c1")
    assert session is not None
    assert session.owner is not None
    assert session.owner.full_name == "Иванов Иван Иванович"
    assert session.owner.address == "ул. Ленина, д. 51"
    assert session.owner.apartment == "15"
    # В публичном транскрипте не должно быть служебной реплики старта
    assert [t.role for t in session.transcript] == ["bot"]

    # Ход 1: get_category + get_tickets_by_address + create_ticket(missing_fields)
    step1 = await dialog.handle_turn("c1", "+79990000001", "у нас сломался лифт")
    assert step1.state == CallState.SLOTS
    assert [e.type for e in step1.events] == ["tool_call", "tool_call", "tool_call"]
    assert step1.events[0].payload["name"] == "get_category"
    assert step1.events[1].payload["name"] == "get_tickets_by_address"
    assert step1.events[2].payload["result"] == {"missing_fields": ["entry"]}
    assert outbox.store.all() == []

    # Ход 2: подъезд уточнён -> заявка уходит в outbox со всеми полями и bitrix_id
    step2 = await dialog.handle_turn("c1", "+79990000001", "второй подъезд, всё верно")
    assert step2.state == CallState.DONE
    assert [e.type for e in step2.events] == ["tool_call", "appeal_created"]
    ev = step2.events[1].payload
    assert ev["appeal_id"] == "5432"
    assert ev["bitrix_id"] == "304688"
    assert ev["topic"] == "Лифты / Неисправность лифта"
    assert ev["identified"] is True
    assert ev["owner"] == "Иванов Иван Иванович"
    assert ev["slots"]["entry"] == "2"
    assert ev["slots"]["deadline"] == "06.11.2026"
    assert {"id": "entry", "label": "Номер подъезда", "value": "2"} in ev["fields"]

    rec = outbox.store.get("5432")
    assert rec is not None
    rendered_tg = render_appeal(rec.appeal, load_cloudru_slot_labels())
    assert "Заявитель: Иванов Иван Иванович <i>(из CRM)</i>" in rendered_tg
    assert "ФИО заявителя: Иванов Иван Иванович" not in rendered_tg
    assert "Улица: Ленина" in rendered_tg
    assert "Дом: 51" in rendered_tg
    assert "Номер подъезда: 2" in rendered_tg
    assert "Срок решения: 06.11.2026" in rendered_tg
    assert await outbox.process_once() == 1

    bye_reply = await dialog.handle_turn("c1", "+79990000001", "нет, спасибо")
    assert bye_reply.end_call is True
    assert [e.type for e in bye_reply.events] == ["call_ended"]


async def test_dialog_adapter_ticket_status_and_null_category_id_isolation() -> None:
    script = [
        (
            "Здравствуйте! Чем могу помочь?",
            [
                _FakeToolCall(
                    name="get_resident_by_phone",
                    arguments={"phone": "+70000000000"},
                    result="null",
                ),
            ],
        ),
        (
            "Заявка 1234 сейчас в работе, ожидаемый срок — 10 ноября.",
            [
                _FakeToolCall(
                    name="get_ticket",
                    arguments={"number": "1234"},
                    result=json.dumps(
                        {
                            "id": 1234,
                            "status": "В работе",
                            "deadline": "10.11.2026",
                            "category": (
                                "Освещение, электричество / "
                                "Неисправность освещения в подъезде / во дворе"
                            ),
                            "phone": "+79990000001",
                            "description": "Не работает освещение в подъезде",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        ),
        (
            "Заявка 1111 создана.",
            [
                _FakeToolCall(
                    name="create_ticket",
                    arguments={
                        "category": "Лифты / Неисправность лифта",
                        "phone": "+70000000000",
                        "description": "Лифт не едет",
                        "street": "Ленина",
                        "house": "51",
                        "resident": "Сидоров Сергей",
                        "entry": 3,
                    },
                    result=json.dumps(
                        {
                            "id": 1111,
                            "status": "Зарегистрирована",
                            "deadline": "06.11.2026",
                            "category": "Лифты / Неисправность лифта",
                            "phone": "+70000000000",
                            "description": "Лифт не едет",
                            "street": "Ленина",
                            "house": "51",
                            "resident": "Сидоров Сергей",
                            "entry": 3,
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
                        "category": "Канализация / Протечка канализации (ДЛЯ КНОПКИ)",
                        "phone": "+70000000000",
                        "description": "Запах канализации в подвале",
                        "street": "Ленина",
                        "house": "51",
                        "resident": "Сидоров Сергей",
                    },
                    result=json.dumps(
                        {
                            "id": 2222,
                            "status": "Зарегистрирована",
                            "deadline": "06.11.2026",
                            "category": "Канализация / Протечка канализации (ДЛЯ КНОПКИ)",
                            "phone": "+70000000000",
                            "description": "Запах канализации в подвале",
                            "street": "Ленина",
                            "house": "51",
                            "resident": "Сидоров Сергей",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        ),
    ]
    dialog, outbox, _, _ = _build_dialog(script)
    await dialog.start("c-multi", "+70000000000")

    status_turn = await dialog.handle_turn("c-multi", "+70000000000", "какой статус у заявки 1234?")
    assert status_turn.state == CallState.DONE
    assert [e.type for e in status_turn.events] == ["tool_call"]
    assert status_turn.events[0].payload["name"] == "get_ticket"

    await dialog.handle_turn("c-multi", "+70000000000", "сломался лифт в 3 подъезде")
    second = await dialog.handle_turn("c-multi", "+70000000000", "и запах канализации в подвале")

    rec2 = outbox.store.get("2222")
    assert rec2 is not None
    assert rec2.appeal.topic.bitrix_id == ""
    assert "entry" not in rec2.appeal.slots
    assert second.events[-1].payload["bitrix_id"] == ""


async def test_dialog_adapter_handles_transfer_autostart_and_start_fallback() -> None:
    script = [
        (
            "Здравствуйте! Чем могу помочь?",
            [
                _FakeToolCall(
                    name="get_resident_by_phone",
                    arguments={"phone": "+70000000000"},
                    result="null",
                ),
            ],
        ),
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
    assert [e.type for e in reply.events] == ["tool_call", "tool_call", "transfer"]
    transfer_ev = next(e for e in reply.events if e.type == "transfer")
    assert transfer_ev.payload["reason"] == "авария: прорыв трубы"

    after = await dialog.handle_turn("c2", "+70000000000", "алло?")
    assert after.transfer is True and "уже переводится" in after.text

    # Фолбэк на старте при недоступности LLM: жилец всё равно находится в локальной CRM-заглушке
    broken_dialog, _, _, _ = _build_dialog([])
    fallback_start = await broken_dialog.start("c-fallback", "+79990000002")
    assert fallback_start.text.startswith("Здравствуйте, Василиса Васильевна")


def test_fastapi_container_wires_cloudru_demo_phones_ui_input_and_asr(
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
        assert cfg["demo_phones"] == ["+79990000001", "+79990000002", "+79990000003"]

        html = client.get("/debug/").text
        assert 'id="phoneInput"' in html
        assert 'value="+79990000001"' in html
        assert 'id="phonePresets"' in html

        container = client.app.state.container
        container.asr._transcribe_fn = lambda wav: (
            "сломался лифт во втором подъезде" if wav[:4] == b"RIFF" else ""
        )
        script = [
            (
                "Здравствуйте, Иван Иванович! Чем могу помочь?",
                [
                    _FakeToolCall(
                        name="get_resident_by_phone",
                        arguments={"phone": "+79990000001"},
                        result=json.dumps(
                            {
                                "name": "Иванов Иван Иванович",
                                "phone": "+79990000001",
                                "address": {"street": "Ленина", "house": "51"},
                                "apartment": 15,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
            ),
            (
                "Заявка 7 7 8 8 создана.",
                [
                    _FakeToolCall(
                        name="create_ticket",
                        arguments={
                            "category": "Лифты / Неисправность лифта",
                            "phone": "+79990000001",
                            "description": "сломался лифт во втором подъезде",
                            "street": "Ленина",
                            "house": "51",
                            "resident": "Иванов Иван Иванович",
                            "entry": 2,
                        },
                        result=json.dumps(
                            {
                                "id": 7788,
                                "status": "Зарегистрирована",
                                "deadline": "06.11.2026",
                                "category": "Лифты / Неисправность лифта",
                                "phone": "+79990000001",
                                "description": "сломался лифт во втором подъезде",
                                "street": "Ленина",
                                "house": "51",
                                "resident": "Иванов Иван Иванович",
                                "entry": 2,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
            ),
        ]
        captured_phones: list[str] = []

        def factory(phone: str) -> _FakeUpstreamAssistant:
            captured_phones.append(phone)
            return _FakeUpstreamAssistant(phone, script)

        container.orchestrator._factory = factory

        started = client.post("/call/start", json={"phone": "+79990000001"}).json()
        call_id = started["call_id"]
        assert captured_phones == ["+79990000001"]
        assert started["text"].startswith("Здравствуйте, Иван Иванович")
        assert [e["type"] for e in started["events"]] == ["tool_call"]
        assert started["events"][0]["payload"]["name"] == "get_resident_by_phone"

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
        assert outbox[0]["appeal"]["owner"]["full_name"] == "Иванов Иван Иванович"
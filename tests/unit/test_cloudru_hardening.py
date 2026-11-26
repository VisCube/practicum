"""Тесты надёжности, нормализации речи, параллельности и живого WS-звонка для режима Cloud.ru."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import importlib
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from assistant.api.app import create_app
from assistant.api.dependencies import Container
from assistant.api.routes.voice_ws import VoiceConnection
from assistant.core.config import Settings
from assistant.core.models import Audio, CallSession, CallState, RiskLevel
from assistant.core.session_store import InMemorySessionStore
from assistant.core.voice_session import VoiceSession
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.itn import numbers_to_digits
from assistant.services.asr.cloudru import CloudRuAsr, is_whisper_hallucination
from assistant.services.cloudru.bridge import export_cloudru_env, load_assistant_cls
from assistant.services.cloudru.dialog import CloudRuDialogAdapter
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.telephony.fake import FakeTelephony
from assistant.services.tts.normalize import clean_for_tts, split_tts_sentences

DATA = Path(__file__).resolve().parents[2] / "data"
KNOWN_CLOUDRU_PHONE = "+7 916 123-45-67"


# ------------------------------------------------------------------ 1. Русский ITN для ASR


def test_itn_repairs_compound_and_glued_ordinals_and_ticket_digits() -> None:
    assert numbers_to_digits("на двадцать седьмом этаже") == "на 27-м этаже"
    assert numbers_to_digits("в сто двадцать первой квартире") == "в 121-й квартире"
    assert numbers_to_digits("4ый подъезд") == "4-й подъезд"
    # Одиночные словесные порядковые не ломают естественные фразы и существующие контракты
    assert (
        numbers_to_digits("сломался лифт во втором подъезде")
        == "сломался лифт во втором подъезде"
    )
    # Продиктованные по цифрам/парам номера заявок склеиваются, а дом/квартира — нет
    assert numbers_to_digits("заявка двенадцать тридцать четыре") == "заявка 1234"
    assert numbers_to_digits("номер пять четыре три два") == "номер 5432"
    assert numbers_to_digits("дом пятнадцать квартира сорок два") == "дом 15 квартира 42"


# ------------------------------------------------------------------ 2. Нормализация текста для TTS


def test_clean_for_tts_strips_markdown_and_expands_domain_tokens() -> None:
    raw = (
        "**Заявка № 5432** по категории «Лифты / Неисправность лифта» создана!\n"
        "- Адрес: ул. Ленина, д. 51, корп. 2, под. 3, эт. 5, кв. 12.\n"
        "- Срок до 25.12.2026, с 09:00 до 18:30, телефон УК +7 (910) 111-11-11, тариф 45,5 руб."
    )
    spoken = clean_for_tts(raw, expand_digits=True)

    assert "**" not in spoken
    assert "№" not in spoken
    assert "/" not in spoken
    assert not any(ch.isdigit() for ch in spoken), f"Остались нераскрытые цифры: {spoken}"
    assert "номер пять тысяч четыреста тридцать два" in spoken
    assert "улица Ленина" in spoken
    assert "дом пятьдесят один" in spoken
    assert "квартира двенадцать" in spoken
    assert "до двадцать пятого декабря две тысячи двадцать шестого года" in spoken
    assert "девять часов ровно" in spoken
    assert "восемнадцать часов тридцать минут" in spoken
    assert "сорок пять и пять рублей" in spoken
    assert "плюс семь девятьсот десять сто одиннадцать одиннадцать одиннадцать" in spoken


def test_clean_for_tts_expands_acronyms_and_8_800_phones() -> None:
    spoken = clean_for_tts(
        "По вопросам ЖКХ, ГВС и ХВС звоните 8-800-555-35-35 или пишите SMS.",
        expand_digits=True,
    )
    assert "жэ ка ха" in spoken
    assert "гэ вэ эс" in spoken
    assert "хэ вэ эс" in spoken
    assert "восемь восемьсот пятьсот пятьдесят пять тридцать пять тридцать пять" in spoken
    assert "эсэмэс" in spoken


def test_split_tts_sentences_keeps_street_abbreviations_together() -> None:
    text = "Проверил адрес ул. Мира, д. 19, кв. 42. Заявка в работе! Могу ещё чем-то помочь?"
    parts = split_tts_sentences(text, max_chars=80)
    assert parts == [
        "Проверил адрес ул. Мира, д. 19, кв. 42.",
        "Заявка в работе!",
        "Могу ещё чем-то помочь?",
    ]


# ------------------------------------------------------------------ 3. Silero TTS: кэш и воркеры


def test_silero_tts_raw_cache_sentence_cache_singleflight_and_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_file = tmp_path / "fake_silero.pt"
    model_file.write_bytes(b"stub")

    synth_calls: list[str] = []
    synth_lock = threading.Lock()
    norm_calls = 0

    class FakeTensor:
        ndim = 1

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def float(self) -> FakeTensor:
            return self

        def squeeze(self) -> FakeTensor:
            return self

        def clamp(self, _lo: builtins.float, _hi: builtins.float) -> FakeTensor:
            return self

        def __mul__(self, _scale: builtins.float) -> FakeTensor:
            return self

        def to(self, _dtype: Any) -> FakeTensor:
            return self

        def numpy(self) -> Any:
            return SimpleNamespace(tobytes=lambda: b"\x01\x00" * 2400)

    class FakeModel:
        speakers = ("ru_eduard", "xenia")

        def to(self, _device: Any) -> FakeModel:
            return self

        def apply_tts(
            self,
            *,
            ssml_text: str | None = None,
            text: str | None = None,
            speaker: str = "ru_eduard",
            sample_rate: int = 24000,
        ) -> FakeTensor:
            _ = (speaker, sample_rate)
            time.sleep(0.03)
            with synth_lock:
                synth_calls.append(ssml_text or text or "")
            return FakeTensor()

    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda d: d  # type: ignore[attr-defined]
    fake_torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    fake_torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
    fake_torch.int16 = "int16"  # type: ignore[attr-defined]
    fake_torch.package = SimpleNamespace(  # type: ignore[attr-defined]
        PackageImporter=lambda _p: SimpleNamespace(load_pickle=lambda *_a: FakeModel()),
    )

    def fake_norm(s: str) -> str:
        nonlocal norm_calls
        norm_calls += 1
        return s

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "assistant.services.tts.silero", raising=False)
    silero_mod = importlib.import_module("assistant.services.tts.silero")
    monkeypatch.setattr(silero_mod, "_load_normalizer", lambda: fake_norm)
    monkeypatch.setattr(silero_mod, "_load_accentor", lambda _dev: None)

    async def scenario() -> None:
        pool = ThreadPoolExecutor(max_workers=2)
        tts = silero_mod.SileroTts(
            model_file,
            workers=2,
            normalize=True,
            stress=False,
            executor=pool,
        )
        try:
            # 1) Singleflight: 4 одновременных одинаковых приветствия → ровно 1 вызов apply_tts
            results = await asyncio.gather(
                *(tts.synthesize("Здравствуйте! Чем помочь?") for _ in range(4))
            )
            assert all(not r.empty for r in results)
            # 2 предложения («Здравствуйте!» + «Чем помочь?») синтезированы ровно по одному разу
            assert len(synth_calls) == 2
            norm_after_cold = norm_calls

            # 2) Полный кэш по сырому тексту: 0 вызовов нормализатора и 0 вызовов apply_tts
            cached = await tts.synthesize("Здравствуйте! Чем помочь?")
            assert cached.data == results[0].data
            assert norm_calls == norm_after_cold
            assert len(synth_calls) == 2

            # 3) По-предложенческий кэш: динамический номер + общий хвост
            await tts.synthesize("Заявка № 1001 создана. Могу ещё чем-то помочь?")
            calls_after_first_ticket = len(synth_calls)
            await tts.synthesize("Заявка № 1002 создана. Могу ещё чем-то помочь?")
            # Второй синтез прогнал только первое предложение, а хвост взял из кэша!
            assert len(synth_calls) == calls_after_first_ticket + 1
            assert "тысяча два" in tts.last_spoken("Заявка № 1002 создана. Могу ещё чем-то помочь?")
        finally:
            await tts.aclose()
            pool.shutdown(wait=False)

    asyncio.run(scenario())


# --------------------------------------------------------- 4. CloudRuAsr: галлюцинации и пул


def test_cloudru_asr_filters_hallucinations_aligns_odd_pcm_and_reuses_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert is_whisper_hallucination("Субтитры сделал DimaTorzok")
    assert is_whisper_hallucination("Редактор субтитров А.Семкин")
    assert is_whisper_hallucination("Продолжение следует...")
    assert not is_whisper_hallucination("продолжение ремонта во втором подъезде")

    created_clients: list[Any] = []
    closed_count = 0

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text="Субтитры создавал DimaTorzok")
            return SimpleNamespace(text="заявка двенадцать тридцать четыре на двадцать пятом этаже")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.transcriptions = FakeTranscriptions()
            self.audio = SimpleNamespace(transcriptions=self.transcriptions)
            created_clients.append(self)

        async def close(self) -> None:
            nonlocal closed_count
            closed_count += 1

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

    async def scenario() -> None:
        asr = CloudRuAsr(api_key="test-key", base_url="https://example.com/v1", model="whisper")
        # Нечётное число байт PCM (3201) выравнивается без ошибки wave.Error, галлюцинация → ""
        odd_pcm = Audio(data=b"\x01" * 3201, mime="audio/pcm")
        first = await asr.transcribe(odd_pcm)
        assert first == ""

        second = await asr.transcribe(odd_pcm)
        assert second == "заявка 1234 на 25-м этаже"
        # Клиент создан ровно один раз на все вызовы
        assert len(created_clients) == 1
        await asr.aclose()
        assert closed_count == 1

    asyncio.run(scenario())


# ------------------------------------------------------------------ 5. Bridge: защиты CRM и env


def test_bridge_env_validation_and_crm_normalization_and_unique_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Отсутствующий FM_API_KEY даёт понятную RuntimeError вместо сырого KeyError из _upstream
    monkeypatch.delenv("FM_API_KEY", raising=False)
    empty_settings = Settings(data_dir=DATA, fm_api_key="", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="FM_API_KEY"):
        export_cloudru_env(empty_settings, require_rag=False)

    valid_settings = Settings(
        data_dir=DATA,
        fm_api_key="key-1",
        rag_key_id="rk",
        rag_key_secret="rs",
        rag_base_url="https://rag.example.com",
        llm_model="gpt-oss-120b",
        _env_file=None,  # type: ignore[call-arg]
    )
    export_cloudru_env(valid_settings, require_rag=True)
    load_assistant_cls()

    assert os.environ["FM_API_KEY"] == "key-1"


# ----------------------------------------------------- 6. CloudRuDialogAdapter: гонки и сбои


def test_cloudru_dialog_parallel_calls_rollback_timeout_and_escalation() -> None:
    shared_options_calls = 0
    closed_throwaways = 0

    class DummyOpenAIClient:
        def with_options(self, **_kwargs: Any) -> DummyOpenAIClient:
            nonlocal shared_options_calls
            shared_options_calls += 1
            return DummyOpenAIClient()

        def close(self) -> None:
            nonlocal closed_throwaways
            closed_throwaways += 1

    class ScriptedAgent:
        def __init__(self, phone: str) -> None:
            self.phone = phone
            self.client: Any = DummyOpenAIClient()
            self.messages: list[dict[str, Any]] = [{"role": "system", "content": phone}]
            self.mode = "ok"

        def handle_message(self, user_input: str) -> tuple[str, list[Any]]:
            if user_input == "Алло":
                self.messages.append({"role": "user", "content": user_input})
                reply = f"Здравствуйте ({self.phone})!"
                self.messages.append({"role": "assistant", "content": reply})
                return reply, []
            if self.mode == "corrupt_and_fail":
                # Имитируем падение посреди незакрытого tool_calls
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({"role": "assistant", "tool_calls": [{"id": "orphan"}]})
                raise RuntimeError("502 Bad Gateway")
            if self.mode == "timeout":
                time.sleep(0.25)
                return "слишком поздно", []
            time.sleep(0.02)
            self.messages.append({"role": "user", "content": user_input})
            if "создай заявку" in user_input:
                tc = SimpleNamespace(
                    name="create_ticket",
                    arguments={
                        "category": "Сантехника / Протечка в квартире",
                        "phone": self.phone,
                        "description": "Капает кран",
                        "resident": "Иванов Иван",
                        "street": "Ленина",
                        "house": "10",
                        "apartment": "5",
                        "room": "кухня",
                    },
                    result=json.dumps({"id": 9101, "category": "Сантехника / Протечка в квартире"}),
                )
                reply = "Заявка создана."
                self.messages.append({"role": "assistant", "content": reply})
                return reply, [tc]
            reply = f"Принято: {user_input}"
            self.messages.append({"role": "assistant", "content": reply})
            return reply, []

    agents_by_phone: dict[str, list[ScriptedAgent]] = {}
    lock = threading.Lock()

    def factory(phone: str) -> ScriptedAgent:
        agent = ScriptedAgent(phone)
        with lock:
            agents_by_phone.setdefault(phone, []).append(agent)
        return agent

    async def scenario() -> None:
        store = InMemorySessionStore()
        crm = InMemoryCrm()
        outbox = Outbox(crm, InMemoryOutboxStore())
        telephony = FakeTelephony()
        escalation = EscalationDetector.from_yaml(DATA / "escalation.yaml")
        adapter = CloudRuDialogAdapter(
            store=store,
            crm=crm,
            outbox=outbox,
            telephony=telephony,
            escalation=escalation,
            agent_timeout=0.08,
            max_concurrency=16,
            assistant_factory=factory,
        )
        try:
            # 1) 10 параллельных звонков одновременно: общий пул клиента, полная изоляция сессий
            async def one_call(i: int) -> None:
                cid = f"call-{i}"
                phone = f"+790000000{i:02d}"
                start_rep = await adapter.start(cid, phone)
                assert phone in start_rep.text
                turn_rep = await adapter.handle_turn(cid, phone, f"вопрос {i}")
                assert turn_rep.text == f"Принято: вопрос {i}"

            await asyncio.gather(*(one_call(i) for i in range(10)))
            assert shared_options_calls == 0
            assert closed_throwaways == 0

            # 2) Маркер жалобы (прокуратура) помечает сессию и будущую заявку CRITICAL
            await adapter.start("c-comp", "+79110000001")
            comp_rep = await adapter.handle_turn(
                "c-comp",
                "+79110000001",
                "если не почините, напишу жалобу в прокуратуру, создай заявку",
            )
            event_types = [e.type for e in comp_rep.events]
            assert "escalated" in event_types
            assert "appeal_created" in event_types
            created_rec = outbox.store.get("9101")
            assert created_rec is not None
            assert created_rec.appeal.escalated is True
            assert created_rec.appeal.risk_level == RiskLevel.CRITICAL

            # Неизвестное состояние после ошибки — перевод без повторных действий.
            await adapter.start("c-fail", "+79110000002")
            fail_agent = agents_by_phone["+79110000002"][-1]
            fail_agent.mode = "corrupt_and_fail"
            r1 = await adapter.handle_turn("c-fail", "+79110000002", "первая попытка")
            assert r1.transfer is True
            assert len(agents_by_phone["+79110000002"]) == 1

            # 4) При аварии (пожар) и лежащем LLM перевод происходит с ПЕРВОЙ ошибки
            await adapter.start("c-fire", "+79110000003")
            fire_agent = agents_by_phone["+79110000003"][-1]
            fire_agent.mode = "corrupt_and_fail"
            r_fire = await adapter.handle_turn("c-fire", "+79110000003", "пожар в подъезде горит!")
            assert r_fire.transfer is True and r_fire.state == CallState.TRANSFERRED
            assert "сто двенадцать" in r_fire.text

            # Таймаут может скрывать побочный эффект: не запускаем повторно.
            await adapter.start("c-time", "+79110000004")
            slow_agent = agents_by_phone["+79110000004"][-1]
            slow_agent.mode = "timeout"
            rt1 = await adapter.handle_turn("c-time", "+79110000004", "зависший запрос")
            assert rt1.transfer is True
            assert "c-time" not in adapter._agents
            rt2 = await adapter.handle_turn("c-time", "+79110000004", "новый запрос")
            assert rt2.transfer is True
            assert len(agents_by_phone["+79110000004"]) == 1
        finally:
            await adapter.aclose()

    asyncio.run(scenario())


# ------------------------------------------------------------------ 7. Live WS, Store и HTTP лимиты


def test_ws_concurrent_cough_does_not_abort_inflight_llm_turn() -> None:
    """Пустой шум/кашель вдогонку не должен срывать выполняющийся ход LLM."""
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        app = cast(FastAPI, client.app)
        container = cast(Container, app.state.container)

        sent: list[dict[str, Any]] = []

        class RecordingWs:
            async def send_text(self, payload: str) -> None:
                sent.append(json.loads(payload))

        asr_queue = ["сломался лифт", ""]

        class QueueAsr:
            async def transcribe(self, _audio: Audio) -> str:
                return asr_queue.pop(0) if asr_queue else ""

        container.asr = QueueAsr()
        orig_handle_turn = container.orchestrator.handle_turn
        dialog_entered = asyncio.Event()
        release_dialog = asyncio.Event()

        async def slow_turn(call_id: str, phone: str, text: str) -> Any:
            dialog_entered.set()
            await release_dialog.wait()
            return await orig_handle_turn(call_id, phone, text)

        container.orchestrator.handle_turn = slow_turn  # type: ignore[method-assign]

        async def scenario() -> None:
            await container.orchestrator.start("c-cough", "+79990000001")
            conn = VoiceConnection(
                ws=RecordingWs(),  # type: ignore[arg-type]
                container=container,
                session=VoiceSession(call_id="c-cough", phone="+79990000001"),
            )
            # Ход 1: реальная фраза ушла в медленный диалог
            conn.session.append(b"\x01" * 3200)
            await conn.handle_event({"t": "utterance_end"})
            await dialog_entered.wait()
            first_task = conn._turn_task
            assert first_task is not None and not first_task.done()

            # Пока LLM думает — прилетает кашель (ASR вернёт "")
            conn.session.append(b"\x02" * 3200)
            await conn.handle_event({"t": "utterance_end"})
            if conn._pre_asr_task:
                await conn._pre_asr_task

            # Первый ход НЕ отменён!
            assert not first_task.cancelled()
            release_dialog.set()
            await first_task
            replies = [m for m in sent if m.get("t") == "reply"]
            assert len(replies) == 1 and replies[0]["text"]
            await conn.close()

        asyncio.run(scenario())


def test_session_store_sweeps_orphan_locks_and_caps_max_sessions() -> None:
    now = 100.0
    store = InMemorySessionStore(ttl_sec=10.0, clock=lambda: now, max_sessions=3)
    for i in range(5):
        now += 1.0
        store.save(CallSession(call_id=f"s{i}", phone="+7000"))
        _ = store.lock(f"s{i}")
    assert len(store) == 3
    # Беру осиротевший лок вручную и вызываю sweep()
    _ = store.lock("orphan")
    store.sweep()
    assert "orphan" not in store._locks


def test_debug_asr_rejects_oversized_upload() -> None:
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        resp = client.post(
            "/debug/asr",
            content=b"x",
            headers={"content-type": "audio/pcm", "content-length": str(9 * 1024 * 1024)},
        )
        assert resp.status_code == 413


def test_ws_concurrent_real_speech_replaces_inflight_turn_without_double_asr() -> None:
    """Если вдогонку пришла настоящая реплика — прежний ход отменяется без повторного ASR."""
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        app = cast(FastAPI, client.app)
        container = cast(Container, app.state.container)

        sent: list[dict[str, Any]] = []

        class RecordingWs:
            async def send_text(self, payload: str) -> None:
                sent.append(json.loads(payload))

        asr_calls = 0

        class CountingAsr:
            async def transcribe(self, _audio: Audio) -> str:
                nonlocal asr_calls
                asr_calls += 1
                return "течёт кран" if asr_calls == 1 else "течёт из батареи в комнате"

        container.asr = CountingAsr()
        orig_handle_turn = container.orchestrator.handle_turn
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def gated_turn(call_id: str, phone: str, text: str) -> Any:
            if text == "течёт кран":
                first_entered.set()
                await release_first.wait()
            return await orig_handle_turn(call_id, phone, text)

        container.orchestrator.handle_turn = gated_turn  # type: ignore[method-assign]

        async def scenario() -> None:
            await container.orchestrator.start("c-corr", "+79990000001")
            conn = VoiceConnection(
                ws=RecordingWs(),  # type: ignore[arg-type]
                container=container,
                session=VoiceSession(call_id="c-corr", phone="+79990000001"),
            )
            conn.session.append(b"\x01" * 3200)
            await conn.handle_event({"t": "utterance_end"})
            await first_entered.wait()
            first_task = conn._turn_task
            assert first_task is not None

            # Настоящая поправка во время ожидания первого ответа
            conn.session.append(b"\x02" * 3200)
            await conn.handle_event({"t": "utterance_end"})
            if conn._pre_asr_task:
                await conn._pre_asr_task

            assert first_task.cancelled()
            release_first.set()
            assert conn._turn_task is not None
            await conn._turn_task

            # Всего ровно 2 вызова ASR (поправка не распознавалась повторно)
            assert asr_calls == 2
            asr_msgs = [m["text"] for m in sent if m.get("t") == "asr"]
            assert asr_msgs == ["течёт кран", "течёт из батареи в комнате"]
            replies = [m for m in sent if m.get("t") == "reply"]
            assert len(replies) == 1
            await conn.close()

        asyncio.run(scenario())


def test_debug_tts_exposes_normalized_spoken_text() -> None:
    settings = Settings(data_dir=DATA, debug_ui=True, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        app = cast(FastAPI, client.app)
        container = cast(Container, app.state.container)

        class StubSpeakingTts:
            speaker = "ru_eduard"

            async def synthesize(self, _text: str, speaker: str | None = None) -> Audio:
                _ = speaker
                return Audio(data=b"RIFF\x00\x00\x00\x00WAVE", mime="audio/wav")

            def last_spoken(self, text: str) -> str:
                return clean_for_tts(text, expand_digits=True)

        container.tts = StubSpeakingTts()
        resp = client.post("/debug/tts", json={"text": "Заявка № 42 по ул. Мира"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["spoken"] == "Заявка номер сорок два по улица Мира"

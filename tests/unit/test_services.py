"""Unit-тесты: адаптеры, outbox, Telegram CRM, LLM-задачи"""

from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from assistant.core.models import Appeal, Audio, Passage, RiskLevel, Topic, Turn
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.llm_tasks import MAX_ANSWER_CHARS, LlmTasks
from assistant.services.asr.fake import FakeAsr
from assistant.services.crm.base import CrmError
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox, OutboxStatus
from assistant.services.crm.telegram import TelegramCrm, render_appeal
from assistant.services.knowledge_base.retriever import MarkdownRetriever
from assistant.services.llm.fake import FakeLlm
from assistant.services.telephony.fake import FakeTelephony
from assistant.services.tts.fake import FakeTts

DATA = Path(__file__).resolve().parents[2] / "data"


def _appeal(call_id: str = "c1") -> Appeal:
    topic = Topic(id="leak", name="Протечка", bitrix_id="TOPIC_LEAK", risk_level=RiskLevel.CRITICAL)
    return Appeal(call_id=call_id, phone="+79990000001", owner=None, topic=topic,
                  slots={"apartment": "42"}, risk_level=topic.risk_level)


def _tg_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- fake CRM

async def test_fake_crm_lookup():
    crm = InMemoryCrm()
    owner = await crm.find_owner_by_phone("+79990000001")
    assert owner and owner.apartment == "42"
    assert await crm.find_owner_by_phone("+70000000000") is None
    outages = await crm.active_outages(owner)
    assert len(outages) == 1
    other_owner = await crm.find_owner_by_phone("+79990000003")
    assert await crm.active_outages(other_owner) == []


# ---------------------------------------------------------------- outbox

async def test_outbox_happy_path():
    crm = InMemoryCrm()
    outbox = Outbox(crm, InMemoryOutboxStore())
    appeal = _appeal()
    local_id = outbox.enqueue(appeal)
    assert local_id == appeal.id and outbox.store.get(appeal.id).status == OutboxStatus.PENDING
    assert await outbox.process_once() == 1
    record = outbox.store.get(appeal.id)
    assert record.status == OutboxStatus.SENT and record.appeal.bitrix_deal_id == f"DEAL-{appeal.id}"
    outbox.enqueue(appeal)
    assert await outbox.process_once() == 0
    assert len(crm.deals) == 1


async def test_outbox_retries_with_backoff_then_succeeds():
    crm = InMemoryCrm(fail_times=2)
    outbox = Outbox(crm, InMemoryOutboxStore(), max_attempts=5, backoff_sec=10)
    appeal = _appeal()
    outbox.enqueue(appeal)
    start_time = datetime(2030, 1, 1, 12, 0, 0)

    assert await outbox.process_once(start_time) == 0
    record = outbox.store.get(appeal.id)
    assert record.attempts == 1 and record.status == OutboxStatus.PENDING and record.last_error
    assert await outbox.process_once(start_time + timedelta(seconds=5)) == 0
    assert record.attempts == 1
    assert await outbox.process_once(start_time + timedelta(seconds=11)) == 0
    assert record.attempts == 2
    assert await outbox.process_once(start_time + timedelta(seconds=32)) == 1
    assert record.status == OutboxStatus.SENT
    assert crm.calls.count("create_deal") == 3


async def test_outbox_gives_up():
    crm = InMemoryCrm(fail_times=99)
    outbox = Outbox(crm, InMemoryOutboxStore(), max_attempts=2, backoff_sec=0)
    outbox.enqueue(_appeal())
    await outbox.process_once()
    await outbox.process_once()
    record = outbox.store.all()[0]
    assert record.status == OutboxStatus.FAILED and record.attempts == 2


async def test_outbox_times_out_hanging_crm():
    crm = InMemoryCrm(delay=5.0)
    outbox = Outbox(crm, InMemoryOutboxStore(), max_attempts=3, backoff_sec=0, crm_timeout=0.05)
    outbox.enqueue(_appeal())
    assert await outbox.process_once() == 0
    record = outbox.store.all()[0]
    assert record.attempts == 1 and record.status == OutboxStatus.PENDING and "timeout" in record.last_error.lower()


# ---------------------------------------------------------------- retriever

async def test_retriever():
    knowledge_base = MarkdownRetriever.from_dir(DATA / "knowledge")
    assert len(knowledge_base.passages) >= 8

    results = await knowledge_base.search("как проходит поверка счётчика")
    assert results and results[0].source == "meter_verification" and "поверк" in results[0].heading.lower()

    results = await knowledge_base.search("что такое КР на СОИ в квитанции")
    assert results[0].heading == "Что такое КР на СОИ"

    results = await knowledge_base.search("хочу справку об отсутствии задолженности", sources=["certificates"])
    assert results and all(passage.source == "certificates" for passage in results)

    assert await knowledge_base.search("ну и что") == []


async def test_retriever_ignores_weak_matches():
    knowledge_base = MarkdownRetriever.from_dir(DATA / "knowledge")
    results = await knowledge_base.search("оформить")
    assert all(passage.score >= knowledge_base.min_score for passage in results)


# ---------------------------------------------------------------- fake адаптеры

async def test_fakes_speech_llm_telephony():
    asr = FakeAsr("течёт кран")
    tts = FakeTts()
    llm = FakeLlm('{"topic":"leak"}')
    telephony = FakeTelephony()

    assert await asr.transcribe(Audio(data=b"\x00\x01", mime="audio/pcm")) == "течёт кран"
    audio = await tts.synthesize("Здравствуйте")
    assert audio.empty and tts.spoken == ["Здравствуйте"]
    assert await llm.complete("sys", "usr", json_mode=True) == '{"topic":"leak"}'
    await telephony.transfer_to_operator("c1", "эскалация")
    assert telephony.transfers == [("c1", "эскалация")]


# ---------------------------------------------------------------- LLM-задачи

async def test_llm_tasks_distrust_the_model():
    classifier = TopicClassifier.from_yaml(DATA / "classifier.yaml")

    def llm_reply(system, user):
        if "классификатор" in system:
            return '```json\n{"topic": "nonexistent"}\n```'
        if "Извлеки" in system:
            return '{"apartment": "седьмая", "floor": 7, "leak_source": "из космоса", "bogus": "x"}'
        return "  НЕТ ОТВЕТА  "

    llm_tasks = LlmTasks(FakeLlm(llm_reply))
    candidates = classifier.candidates_for_llm()
    assert "other" not in {candidate["id"] for candidate in candidates}
    assert await llm_tasks.classify("что-то", candidates) is None

    specs = [classifier.slots[slot_id] for slot_id in ("apartment", "floor", "leak_source")]
    slots = await llm_tasks.extract_slots("реплика", specs)
    assert slots == {"apartment": "7", "floor": "7"}

    assert await llm_tasks.answer("вопрос", [Passage(source="s", heading="h", text="t")]) is None


async def test_llm_answer_is_trimmed_for_speech():
    llm_tasks = LlmTasks(FakeLlm("Это очень длинное предложение для проверки. " * 40))
    answer = await llm_tasks.answer("вопрос", [Passage(source="s", heading="h", text="t")])
    assert answer and len(answer) <= MAX_ANSWER_CHARS and answer.endswith(".")


# ---------------------------------------------------------------- Telegram CRM

async def test_telegram_crm_sends_and_is_idempotent():
    sent_requests = []

    def handler(request: httpx.Request):
        sent_requests.append(request.read().decode())
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    inner_crm = InMemoryCrm()
    crm = TelegramCrm(inner_crm, "TOKEN", "-100",
                      channel_username="my_channel",
                      slot_labels={"apartment": "Квартира"}, client=_tg_client(handler))
    appeal = _appeal()
    appeal.escalated, appeal.escalation_markers = True, ["прокуратура"]
    appeal.transcript = [Turn(role="user", text="течёт <сильно>")]

    assert await crm.create_deal(appeal) == "TG-42"
    assert appeal.channel_link == "https://t.me/my_channel/42"
    assert await crm.create_deal(appeal) == "TG-42" and len(sent_requests) == 1
    request_body = sent_requests[0]
    assert "Заявка № " in request_body and "СРОЧНО" in request_body and "прокуратура" in request_body
    assert "Квартира: 42" in request_body and "&lt;сильно&gt;" in request_body
    assert "sendMessage" in str(crm._url) and "TOKEN" in str(crm._url)

    assert (await crm.find_owner_by_phone("+79990000001")).apartment == "42"

    # Без настроенного канала ссылка не формируется
    crm_no_channel = TelegramCrm(InMemoryCrm(), "TOKEN", "-100", client=_tg_client(handler))
    appeal2 = _appeal()
    assert await crm_no_channel.create_deal(appeal2) == "TG-42"
    assert appeal2.channel_link is None


async def test_telegram_crm_errors_become_crm_error():
    def handler(request):
        return httpx.Response(429, json={"ok": False, "description": "Too Many Requests"})

    crm = TelegramCrm(InMemoryCrm(), "T", "-1", client=_tg_client(handler))
    with pytest.raises(CrmError, match="Too Many Requests"):
        await crm.create_deal(_appeal())

    def broken_handler(request):
        raise httpx.ConnectError("no network")

    crm = TelegramCrm(InMemoryCrm(), "T", "-1", client=_tg_client(broken_handler))
    with pytest.raises(CrmError, match="no network"):
        await crm.create_deal(_appeal())


def test_render_appeal_fits_telegram_limit():
    appeal = _appeal()
    appeal.transcript = [Turn(role="bot", text="x" * 500)] * 20
    assert len(render_appeal(appeal, {})) <= 4096
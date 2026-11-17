"""Сквозные сценарии через оркестратор на fake-адаптерах"""

from pathlib import Path

import pytest

from assistant.core.models import CallState, TopicKind
from assistant.core.orchestrator import Orchestrator
from assistant.core.session_store import InMemorySessionStore
from assistant.handlers.consultation import ConsultationHandler
from assistant.handlers.intake import IntakeHandler
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.llm_tasks import LlmTasks
from assistant.services.crm.fake import InMemoryCrm
from assistant.services.crm.outbox import InMemoryOutboxStore, Outbox
from assistant.services.knowledge_base.retriever import MarkdownRetriever
from assistant.services.llm.fake import FakeLlm
from assistant.services.telephony.fake import FakeTelephony

DATA = Path(__file__).resolve().parents[2] / "data"
KNOWN, UNKNOWN = "+79990000001", "+70000000000"


class World:
    """Тестовая сборка оркестратора на fake-адаптерах - точка входа для сценарных тестов"""

    def __init__(self, llm=None, crm=None, **orch_kwargs):
        """Собирает оркестратор с реальными данными из /data и заданными адаптерами"""
        self.crm = crm or InMemoryCrm()
        self.outbox = Outbox(self.crm, InMemoryOutboxStore())
        self.tel = FakeTelephony()
        classifier = TopicClassifier.from_yaml(DATA / "classifier.yaml")
        llm_tasks = LlmTasks(llm) if llm else None
        self.orch = Orchestrator(
            store=InMemorySessionStore(), crm=self.crm, classifier=classifier,
            escalation=EscalationDetector.from_yaml(DATA / "escalation.yaml"),
            handlers={
                TopicKind.APPEAL: IntakeHandler(self.outbox, classifier.slots, llm=llm_tasks),
                TopicKind.CONSULT: ConsultationHandler(
                    MarkdownRetriever.from_dir(DATA / "knowledge"), llm_tasks),
            },
            telephony=self.tel, llm=llm_tasks, crm_timeout=0.5, **orch_kwargs,
        )

    async def say(self, text, call_id="c1", phone=KNOWN):
        """Отправляет реплику в оркестратор от имени звонящего"""
        return await self.orch.handle_turn(call_id, phone, text)

    def session(self, call_id="c1"):
        """Возвращает текущую сессию звонка из хранилища"""
        return self.orch.store.get(call_id)

    def appeal(self, reply):
        """Извлекает заявку из outbox по событию appeal_created в reply"""
        appeal_event = next(e for e in reply.events if e.type == "appeal_created")
        return self.outbox.store.get(appeal_event.payload["appeal_id"]).appeal


@pytest.fixture
def world():
    return World()


# ---------------------------------------------------------------- старт звонка

async def test_known_owner_greeted_by_name_then_outage_notice(world):
    reply = await world.orch.start("c1", KNOWN)
    assert reply.text.startswith("Здравствуйте, Иван Иванович")
    assert "горячей воды" in reply.text
    assert [e.type for e in reply.events] == ["outage_notice"]
    assert reply.state == CallState.TOPIC


async def test_unknown_owner_plain_greeting(world):
    reply = await world.orch.start("c1", UNKNOWN)
    assert reply.text.startswith("Здравствуйте!") and not reply.events


async def test_handle_turn_autostarts_and_keeps_greeting(world):
    reply = await world.say("нет горячей воды")
    assert reply.text.startswith("Здравствуйте, Иван Иванович")
    assert reply.events[0].type == "outage_notice"
    assert reply.state == CallState.CONFIRM and "квартира — 42" in reply.text


async def test_handle_turn_autostart_with_empty_text_is_just_greeting(world):
    reply = await world.say("")
    assert reply.text.startswith("Здравствуйте, Иван Иванович") and reply.state == CallState.TOPIC


async def test_empty_text_in_session_asks_to_repeat(world):
    await world.orch.start("c1", KNOWN)
    await world.say("течёт с потолка")
    reply = await world.say("   ")
    assert "не расслышал" in reply.text.lower()
    assert reply.state == CallState.CONFIRM


async def test_goodbye_right_after_outage_notice(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("а, понял, спасибо, всего доброго")
    assert reply.end_call and reply.events[0].type == "call_ended"


# ---------------------------------------------------------------- приём заявки

async def test_intake_known_owner_full_flow(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("у меня течёт батарея, залило пол")
    assert "Протечка" in reply.text and "Откуда идёт вода" in reply.text
    assert reply.state == CallState.SLOTS

    reply = await world.say("ну сверху от соседей льёт")
    assert reply.state == CallState.CONFIRM
    assert "квартира — 42" in reply.text and "сверху от соседей" in reply.text
    assert "Всё верно?" in reply.text and "Оформляем" not in reply.text

    reply = await world.say("да, всё верно")
    assert reply.state == CallState.DONE
    appeal_event = reply.events[0]
    assert appeal_event.type == "appeal_created"
    appeal_id = appeal_event.payload["appeal_id"]
    assert appeal_id in reply.text and "срочная" in reply.text
    assert appeal_event.payload["owner"] == "Иванов Иван Иванович"
    assert appeal_event.payload["escalated"] is False
    assert {f["id"] for f in appeal_event.payload["fields"]} == {"apartment", "leak_source", "address", "full_name"}

    record = world.outbox.store.get(appeal_id)
    assert record and record.appeal.owner.full_name == "Иванов Иван Иванович"
    assert record.appeal.slots["leak_source"] == "сверху от соседей"
    assert await world.outbox.process_once() == 1

    reply = await world.say("спасибо, всё")
    assert reply.end_call and reply.events[0].type == "call_ended"


async def test_intake_unknown_owner_identification(world):
    await world.orch.start("c1", UNKNOWN)
    reply = await world.say("нет света во всей квартире", phone=UNKNOWN)
    assert reply.state == CallState.IDENTIFY and "фамилию" in reply.text
    reply = await world.say("Смирнова Анна Павловна", phone=UNKNOWN)
    assert "адрес" in reply.text.lower()
    reply = await world.say("улица Садовая дом 3", phone=UNKNOWN)
    assert "квартиры" in reply.text
    reply = await world.say("квартира пятнадцать", phone=UNKNOWN)
    assert reply.state == CallState.CONFIRM and "Смирнова" in reply.text and "15" in reply.text
    reply = await world.say("да", phone=UNKNOWN)
    assert reply.state == CallState.DONE
    record = world.outbox.store.all()[0]
    assert record.appeal.owner is None
    assert record.appeal.slots == {"full_name": "Смирнова Анна Павловна",
                                   "address": "улица Садовая дом 3", "apartment": "15"}


async def test_prefill_from_first_utterance(world):
    await world.orch.start("c1", UNKNOWN)
    await world.say("лифт застрял на 7 этаже", phone=UNKNOWN)
    assert world.session().slots["floor"] == "7"


async def test_confirm_no_reasks_topic_slots_only(world):
    await world.orch.start("c1", KNOWN)
    await world.say("течёт с потолка")
    reply = await world.say("нет, неправильно")
    assert reply.state == CallState.SLOTS and "Откуда идёт вода" in reply.text
    assert world.session().slots["apartment"] == "42"


async def test_slot_retry_then_choice_hint(world):
    await world.orch.start("c1", KNOWN)
    await world.say("затопило")
    reply = await world.say("ну не знаю")
    assert "выберите один из вариантов" in reply.text and reply.state == CallState.SLOTS


async def test_second_appeal_in_same_call_keeps_identity(world):
    await world.orch.start("c1", UNKNOWN)
    await world.say("нет света", phone=UNKNOWN)
    await world.say("Смирнова Анна Павловна", phone=UNKNOWN)
    await world.say("улица Садовая дом 3", phone=UNKNOWN)
    await world.say("квартира 15", phone=UNKNOWN)
    await world.say("да", phone=UNKNOWN)
    reply = await world.say("и ещё отопления нет", phone=UNKNOWN)
    assert world.session().topic.id == "no_heating"
    assert reply.state == CallState.CONFIRM
    assert "Смирнова" in reply.text and "15" in reply.text


# ---------------------------------------------------------------- тема не ясна

async def test_greeting_is_not_a_topic_attempt(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("добрый день")
    assert "Слушаю вас" in reply.text and world.session().attempts == 0
    reply = await world.say("алло, это управляющая компания?")
    assert "Слушаю вас" in reply.text and world.session().attempts == 0
    reply = await world.say("нет горячей воды")
    assert reply.state == CallState.CONFIRM


async def test_unknown_topic_twice_falls_back_with_details_prefilled(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("ну вот такое дело у меня")
    assert "не совсем понял" in reply.text and reply.state == CallState.TOPIC
    reply = await world.say("странное что-то в подъезде")
    session = world.session()
    assert session.topic.id == "other"
    assert session.slots["details"] == "странное что-то в подъезде"
    assert reply.state == CallState.CONFIRM
    assert "странное что-то в подъезде" in reply.text
    assert "Другое" not in reply.text


async def test_fallback_topic_upgrades_when_real_topic_named(world):
    await world.orch.start("c1", UNKNOWN)
    await world.say("ну вот такое дело", phone=UNKNOWN)
    reply = await world.say("странное что-то", phone=UNKNOWN)
    assert reply.state == CallState.IDENTIFY
    reply = await world.say("да отопления нет у меня", phone=UNKNOWN)
    session = world.session()
    assert session.topic.id == "no_heating"
    assert "details" not in session.slots
    assert reply.state == CallState.IDENTIFY


async def test_confirm_no_in_fallback_returns_to_topic(world):
    await world.orch.start("c1", KNOWN)
    await world.say("ну вот такое дело")
    await world.say("странное что-то")
    reply = await world.say("нет")
    session = world.session()
    assert reply.state == CallState.TOPIC and session.topic is None
    assert "details" not in session.slots
    reply = await world.say("нет горячей воды")
    assert world.session().topic.id == "no_water" and reply.state == CallState.CONFIRM


async def test_unknown_topic_without_fallback_transfers():
    world = World()
    world.orch.classifier.fallback = None
    await world.orch.start("c1", KNOWN)
    await world.say("ну вот такое дело")
    reply = await world.say("странное что-то")
    assert reply.transfer and reply.state == CallState.TRANSFERRED
    assert reply.events[0].type == "transfer"
    assert reply.events[0].payload["reason"] == "тема не определена"
    assert "не смог разобрать" in reply.text.lower()


# ---------------------------------------------------------------- жалобы (флаг, не перевод)

async def test_complaint_flags_appeal_but_does_not_transfer(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("если не почините, я пойду в прокуратуру — у меня протечка, всё залило")
    assert not reply.transfer and reply.state == CallState.SLOTS
    assert [e.type for e in reply.events] == ["escalated"]
    assert reply.events[0].payload["markers"] == ["прокуратура"]
    assert "приоритетное" in reply.text and "Откуда идёт вода" in reply.text
    assert reply.text.startswith("Понимаю вас.")

    await world.say("сверху от соседей")
    reply = await world.say("да")
    assert reply.state == CallState.DONE
    appeal = world.appeal(reply)
    assert appeal.escalated is True
    assert reply.events[0].payload["escalated"] is True
    assert not world.tel.transfers


async def test_complaint_is_announced_once_per_marker(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("в прокуратуру напишу, течёт у меня")
    assert [e.type for e in reply.events] == ["escalated"]
    reply = await world.say("и в ГЖИ тоже, сверху от соседей льёт")
    assert [e.type for e in reply.events] == ["escalated"]
    assert reply.events[0].payload["markers"] == ["ГЖИ"]
    reply = await world.say("нет, в прокуратуру, я сказал")
    assert not any(e.type == "escalated" for e in reply.events)
    assert not reply.text.startswith("Понимаю вас.")
    assert world.session().escalation_markers == ["прокуратура", "ГЖИ"]


async def test_complaint_transfers_when_configured_so():
    world = World(transfer_on_complaint=True)
    await world.orch.start("c1", KNOWN)
    reply = await world.say("если не почините, я пойду в прокуратуру")
    assert reply.transfer and reply.state == CallState.TRANSFERRED
    assert reply.events[0].payload["reason"] == "угроза жалобы"
    assert reply.events[0].payload["markers"] == ["прокуратура"]


# ---------------------------------------------------------------- перевод на оператора

async def test_operator_request_transfers_with_context(world):
    await world.orch.start("c1", KNOWN)
    await world.say("течёт с потолка")
    reply = await world.say("дайте оператора")
    assert reply.transfer and reply.state == CallState.TRANSFERRED
    assert [e.type for e in reply.events] == ["transfer"]
    transfer_context = reply.events[0].payload
    assert transfer_context["reason"] == "клиент просит оператора"
    assert transfer_context["markers"] == ["просит оператора"]
    assert transfer_context["topic"].startswith("Протечка")
    assert transfer_context["slots"]["apartment"] == "42"
    assert transfer_context["owner"] == "Иванов Иван Иванович"
    assert world.tel.transfers[0][0] == "c1"

    reply = await world.say("алло?")
    assert reply.transfer and "уже переводится" in reply.text


async def test_operator_request_at_very_start(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("соедините меня с живым человеком")
    assert reply.transfer
    assert reply.events[0].payload["topic"] is None


async def test_word_person_alone_is_not_transfer(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("у меня в квартире пожилой человек, а отопления нет")
    assert not reply.transfer
    assert world.session().topic.id == "no_heating"


# ---------------------------------------------------------------- консультация

async def test_consultation_from_knowledge_base(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("подскажите, как проходит поверка счётчика")
    assert reply.state == CallState.DONE
    assert "поверк" in reply.text.lower()
    assert "Могу ещё чем-то помочь" in reply.text
    reply = await world.say("а что такое КР на СОИ?")
    assert "общего имущества" in reply.text.lower()


@pytest.mark.parametrize("text", ["всё равно течёт", "ничего не работает", "нет, ещё вопрос есть"])
async def test_not_goodbye_in_done(world, text):
    await world.orch.start("c1", KNOWN)
    await world.say("подскажите, как проходит поверка счётчика")
    reply = await world.say(text)
    assert not reply.end_call


@pytest.mark.parametrize("text", ["нет", "спасибо, всё", "всё, до свидания", "нет, спасибо большое", "ничего не надо"])
async def test_goodbye_in_done(world, text):
    await world.orch.start("c1", KNOWN)
    await world.say("подскажите, как проходит поверка счётчика")
    reply = await world.say(text)
    assert reply.end_call and reply.events[0].type == "call_ended"


# ---------------------------------------------------------------- LLM и деградация

async def test_llm_fallback_classification_and_slots():
    def llm_reply(system, user):
        if "классификатор" in system:
            return '{"topic": "no_heating"}'
        if "Извлеки" in system:
            return '{"apartment": "9"}'
        return "Поверку проводит аккредитованная организация прямо у вас дома."

    world = World(llm=FakeLlm(llm_reply))
    await world.orch.start("c1", UNKNOWN)
    reply = await world.say("дома как в холодильнике, невозможно", phone=UNKNOWN)
    session = world.session()
    assert session.topic.id == "no_heating"
    assert session.slots.get("apartment") == "9"
    assert reply.state == CallState.IDENTIFY

    world2 = World(llm=FakeLlm(llm_reply))
    await world2.orch.start("c1", KNOWN)
    reply = await world2.say("как поверить счётчик")
    assert "аккредитованная организация прямо у вас дома" in reply.text


async def test_crm_timeout_degrades_gracefully():
    world = World(crm=InMemoryCrm(delay=2.0))
    reply = await world.orch.start("c1", KNOWN)
    assert reply.text.startswith("Здравствуйте!")
    reply = await world.say("нет воды")
    assert reply.state == CallState.IDENTIFY


async def test_telephony_failure_still_tells_client_about_transfer():
    class BrokenTelephony(FakeTelephony):
        async def transfer_to_operator(self, call_id, reason):
            raise RuntimeError("SIP down")

    world = World()
    world.orch.telephony = BrokenTelephony()
    await world.orch.start("c1", KNOWN)
    reply = await world.say("дайте оператора")
    assert reply.transfer and "оператора" in reply.text


async def test_confirm_no_with_only_crm_slot_reasks_it(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("нет горячей воды")
    assert reply.state == CallState.CONFIRM
    reply = await world.say("нет")
    assert reply.state == CallState.SLOTS and "номер квартиры" in reply.text
    reply = await world.say("квартира 7")
    assert reply.state == CallState.CONFIRM and "квартира — 7" in reply.text


async def test_prefill_number_needs_context(world):
    await world.orch.start("c1", UNKNOWN)
    await world.say("лифт застрял на 7 этаже", phone=UNKNOWN)
    session = world.session()
    assert session.slots["floor"] == "7"
    assert "apartment" not in session.slots


async def test_prefill_apartment_with_cue(world):
    await world.orch.start("c1", UNKNOWN)
    reply = await world.say("квартира 15, течёт с потолка", phone=UNKNOWN)
    session = world.session()
    assert session.slots["apartment"] == "15" and session.slots["leak_source"] == "с потолка"
    assert reply.state == CallState.IDENTIFY and "фамилию" in reply.text

# ---------------------------------------------------------------- аварии (перевод сразу, своя фраза)

async def test_emergency_transfers_with_safety_phrase(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("у нас пожар на пятом этаже")
    assert reply.transfer and reply.state == CallState.TRANSFERRED
    assert "сто двенадцать" in reply.text
    transfer = next(e for e in reply.events if e.type == "transfer")
    assert transfer.payload["reason"] == "авария: пожар"
    assert transfer.payload["markers"] == ["пожар"]
    assert transfer.payload["emergency"] is True
    assert transfer.payload["owner"] == "Иванов Иван Иванович"
    assert world.tel.transfers == [("c1", "авария: пожар")]

    reply = await world.say("алло?")
    assert reply.transfer and "уже переводится" in reply.text


async def test_gas_emergency_has_its_own_instruction(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("пахнет газом на кухне")
    assert reply.transfer and "сто четыре" in reply.text
    assert "тема обращения" not in reply.text   # никаких вопросов и слотов


async def test_emergency_on_very_first_turn_keeps_greeting(world):
    reply = await world.say("прорвало трубу, хлещет!")
    assert reply.text.startswith("Здравствуйте, Иван Иванович")
    assert reply.transfer and "диспетчером" in reply.text
    assert [e.type for e in reply.events][-1] == "transfer"


async def test_emergency_mid_slot_filling_carries_context(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("течёт с потолка")
    assert reply.state == CallState.CONFIRM
    reply = await world.say("ой, подождите, прорвало трубу, потоп!")
    assert reply.transfer
    transfer = next(e for e in reply.events if e.type == "transfer")
    assert transfer.payload["reason"] == "авария: прорыв, потоп"
    assert transfer.payload["topic"].startswith("Протечка")
    assert transfer.payload["slots"]["leak_source"] == "с потолка"


async def test_emergency_beats_operator_and_complaint(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("пожар! дайте оператора, я в прокуратуру напишу")
    assert reply.transfer and "сто двенадцать" in reply.text
    assert [e.type for e in reply.events] == ["transfer"]   # escalated-события нет — не до жалоб
    assert reply.events[0].payload["reason"] == "авария: пожар"
    assert not world.session().escalated


@pytest.mark.parametrize("text", [
    "пожарная сигнализация орёт в подъезде",
    "не горит лампочка в подъезде",
    "затопили соседи сверху",
    "газон во дворе не косят",
])
async def test_ordinary_appeal_is_not_emergency(world, text):
    await world.orch.start("c1", KNOWN)
    reply = await world.say(text)
    assert not reply.transfer and reply.state != CallState.TRANSFERRED
    assert not world.tel.transfers


async def test_leak_without_panic_goes_through_slots(world):
    await world.orch.start("c1", KNOWN)
    reply = await world.say("затопило, с потолка капает")
    assert not reply.transfer
    assert world.session().topic.id == "leak" and reply.state == CallState.CONFIRM
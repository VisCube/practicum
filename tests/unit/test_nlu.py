"""Unit-тесты: NLU, слоты, хранилище сессий"""

from pathlib import Path

import pytest

from assistant.core.models import CallSession, CallState, SlotSpec, TopicKind
from assistant.core.session_store import InMemorySessionStore
from assistant.nlu.classifier import TopicClassifier
from assistant.nlu.escalation import EscalationDetector
from assistant.nlu.slots import RuleSlotExtractor, missing
from assistant.nlu.text import contains_stem, words_to_digits
from assistant.nlu.yesno import is_no, is_yes
from assistant.templates.phrases import PHRASES, is_goodbye, is_greeting

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def classifier() -> TopicClassifier:
    return TopicClassifier.from_yaml(DATA / "classifier.yaml")


@pytest.fixture(scope="module")
def escalation_detector() -> EscalationDetector:
    return EscalationDetector.from_yaml(DATA / "escalation.yaml")


# ---------------------------------------------------------------- text

@pytest.mark.parametrize(
    "text,stem,hit",
    [
        ("пойду в прокуратуру", "прокуратур", True),
        ("Течёт кран", "течет", True),
        ("протекает", "течет", False),
        ("напишу в жилищную инспекцию", "жилищн инспекц", True),
        ("жилищная, говорю, инспекция", "жилищн инспекц", False),
        ("у нас газ пахнет", "газ.", True),
        ("пахнет газом", "газ.", False),
        ("газон не косят", "газ.", False),
        ("свет отключили", "свет отключ", True),
    ],
)
def test_contains_stem(text, stem, hit):
    assert contains_stem(text, stem) is hit


def test_words_to_digits():
    assert words_to_digits("квартира сорок два") == "квартира 42"
    assert words_to_digits("сто двадцать пять") == "125"
    assert words_to_digits("на седьмом этаже") == "на 7 этаже"
    assert words_to_digits("два три") == "2 3"
    assert words_to_digits("нет воды") == "нет воды"


# ---------------------------------------------------------------- classifier

@pytest.mark.parametrize(
    "text,topic",
    [
        ("У меня течёт батарея, залило пол", "leak"),
        ("нет горячей воды третий день", "no_water"),
        ("где вода, горячая вода", "no_water"),
        ("холодная вода еле идёт, напора нет", "no_water"),
        ("отопления нет", "no_heating"),
        ("батареи холодные, замерзаем", "no_heating"),
        ("дома холодно, не греют батареи", "no_heating"),
        ("света нет в квартире", "no_electricity"),
        ("выбило автомат, всё обесточено", "no_electricity"),
        ("в подъезде пахнет газом", "gas"),
        ("хочу узнать порядок поверки счётчика", "ipu"),
        ("что такое КР на СОИ в квитанции", "charges"),
        ("нужна справка об отсутствии задолженности", "certificate"),
        ("лифт застрял между этажами", "elevator"),
        ("пересчитайте мне за январь, начислили лишнего", "recalculation"),
        ("хочу написать жалобу на бездействие", "complaint"),
    ],
)
def test_classify(classifier, text, topic):
    match = classifier.classify(text)
    assert match is not None and match.topic.id == topic


@pytest.mark.parametrize(
    "text",
    [
        "добрый день, просто поболтать",
        "Светлана Петровна Иванова",
        "у нас газон не косят",
        "прочитал в газете",
        "пятнадцать",
    ],
)
def test_classify_none(classifier, text):
    assert classifier.classify(text) is None


def test_fallback_and_kinds(classifier):
    assert classifier.fallback is not None and classifier.fallback.id == "other"
    assert classifier.by_id("ipu").kind == TopicKind.CONSULT
    assert classifier.by_id("leak").kind == TopicKind.APPEAL


def test_tie_breaks_toward_higher_risk(classifier):
    match = classifier.classify("течёт из батареи")
    assert match.topic.id == "leak" and match.score == 1


def test_more_hits_beat_risk(classifier):
    match = classifier.classify("батареи холодные, отопления нет, но и труба чуть течёт")
    assert match.topic.id == "no_heating" and match.score == 2


def test_topic_names_are_speakable(classifier):
    for topic in classifier.topics:
        assert "/" not in topic.name, f"{topic.id}: TTS прочитает слэш вслух"


# ---------------------------------------------------------------- slots

def test_slots(classifier):
    extractor = RuleSlotExtractor()
    assert extractor.extract(classifier.slots["apartment"], "квартира сорок... то есть 42") == "42"
    assert extractor.extract(classifier.slots["apartment"], "квартира сорок два") == "42"
    assert extractor.extract(classifier.slots["apartment"], "12а") == "12а"
    assert extractor.extract(classifier.slots["floor"], "на седьмом этаже") == "7"
    assert extractor.extract(classifier.slots["leak_source"], "ну сверху от соседей льёт") == "сверху от соседей"
    assert extractor.extract(classifier.slots["full_name"], "Иванов Иван Иванович") == "Иванов Иван Иванович"
    assert extractor.extract(classifier.slots["full_name"], "а зачем вам моё имя?") is None
    long_text = " ".join(["слово"] * 15)
    assert extractor.extract(SlotSpec(id="x", label="x", question="?", kind="text"), long_text) is None
    assert missing(["apartment", "leak_source"], {"apartment": "5"}) == ["leak_source"]


# ---------------------------------------------------------------- escalation

def test_complaint_markers_do_not_request_operator(escalation_detector):
    detection = escalation_detector.detect("если не сделаете, пойду в прокуратуру")
    assert detection.complaint == ["прокуратура"] and not detection.wants_operator and detection.transfer == []


def test_operator_request(escalation_detector):
    detection = escalation_detector.detect("Соедините с оператором!")
    assert detection.wants_operator and "просит оператора" in detection.transfer and detection.complaint == []


def test_multiword_marker_and_dedup(escalation_detector):
    detection = escalation_detector.detect("напишу в жилищную инспекцию и в жилинспекцию")
    assert detection.complaint == ["ГЖИ"]


@pytest.mark.parametrize(
    "text",
    [
        "течёт кран",
        "у меня в квартире пожилой человек",
        "десять человек без воды сидят",
        "администратор сайта не отвечает",
    ],
)
def test_no_false_escalation(escalation_detector, text):
    detection = escalation_detector.detect(text)
    assert not detection.wants_operator and detection.complaint == []


def test_both_signals_in_one_utterance(escalation_detector):
    detection = escalation_detector.detect("дайте оператора, я в суд подам")
    assert detection.wants_operator and detection.complaint == ["суд"]


# ---------------------------------------------------------------- yes / no

def test_yesno():
    assert is_yes("да, всё верно") and is_yes("Правильно") and is_yes("ага")
    assert is_no("нет, не так") and not is_yes("нет")
    assert not is_no("да")
    assert not is_yes("да нет, неправильно") and is_no("да нет, неправильно")
    assert not is_yes("да не так всё")


# ---------------------------------------------------------------- goodbye / greeting

@pytest.mark.parametrize(
    "text,bye",
    [
        ("нет", True),
        ("нет, спасибо", True),
        ("всё, до свидания", True),
        ("а, понял, спасибо, всего доброго", True),
        ("ничего не надо", True),
        ("нет, ещё вопрос есть", False),
        ("всё равно течёт", False),
        ("ничего не работает", False),
        ("спасибо, а ещё батарея холодная", False),
        ("", False),
    ],
)
def test_is_goodbye(text, bye):
    assert is_goodbye(text) is bye


@pytest.mark.parametrize(
    "text,hello",
    [
        ("добрый день", True),
        ("алло, это управляющая компания?", True),
        ("здравствуйте", True),
        ("добрый день, нет воды", False),
        ("привет макака", False),
        ("", False),
    ],
)
def test_is_greeting(text, hello):
    assert is_greeting(text) is hello


def test_phrases_have_no_dangling_placeholders():
    """Каждый {плейсхолдер} должен быть в известном списке — иначе say() упадёт в проде."""
    known = {"name", "address", "description", "topic", "question", "choices", "summary",
             "label", "value", "appeal_id", "sla", "answer"}
    import string
    for key, template in PHRASES.items():
        fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
        assert fields <= known, f"{key}: неизвестные плейсхолдеры {fields - known}"


# ---------------------------------------------------------------- session store

def test_session_store_ttl_and_locks():
    clock = [1000.0]
    store = InMemorySessionStore(ttl_sec=10, clock=lambda: clock[0])
    session = CallSession(call_id="c1", phone="+79990000000")
    session.say("Здравствуйте")
    session.hear("Течёт")
    store.save(session)
    retrieved = store.get("c1")
    assert retrieved and retrieved.state == CallState.GREETING and len(retrieved.transcript) == 2
    assert store.lock("c1") is store.lock("c1")

    clock[0] += 8
    assert store.get("c1") is not None
    clock[0] += 11
    store.save(CallSession(call_id="c2", phone="+7"))
    assert store.get("c1") is None and len(store) == 1

    store.delete("c2")
    assert store.get("c2") is None and len(store) == 0


def test_number_in_context(classifier):
    extractor = RuleSlotExtractor()
    slot_apartment, slot_floor = classifier.slots["apartment"], classifier.slots["floor"]
    assert extractor.extract_in_context(slot_apartment, "квартира 42 течёт") == "42"
    assert extractor.extract_in_context(slot_apartment, "живу в кв. 12а") == "12а"
    assert extractor.extract_in_context(slot_apartment, "лифт застрял на 7 этаже") is None
    assert extractor.extract_in_context(slot_floor, "лифт застрял на 7 этаже") == "7"
    assert extractor.extract_in_context(slot_apartment, "в квитанции 42 рубля лишних") is None
    assert extractor.extract_in_context(slot_apartment, "квартира сорок два, этаж пять") == "42"
    assert extractor.extract_in_context(slot_floor, "квартира сорок два, этаж пять") == "5"
    assert extractor.extract_in_context(slot_apartment, "42") is None


def test_number_slots_have_cues(classifier):
    """Число из свободной речи снимается только по подсказке — у каждого number-слота она должна быть."""
    for slot in classifier.slots.values():
        if slot.kind == "number":
            assert slot.cues, f"слот {slot.id}: нет cues — префилл из первой реплики не сработает"

# ---------------------------------------------------------------- emergency

EMERGENCY = [
    ("пожар", "пожар"),
    ("у нас пожар на пятом этаже", "пожар"),
    ("горим! быстрее!", "пожар"),
    ("горит щиток", "пожар"),
    ("загорелась проводка в подъезде", "пожар"),
    ("открытый огонь на балконе", "пожар"),
    ("сильный дым на этаже", "задымление"),
    ("весь подъезд в дыму", "задымление"),
    ("задымление в коридоре", "задымление"),
    ("потоп", "потоп"),
    ("у нас потоп", "потоп"),
    ("прорыв", "прорыв"),
    ("прорвало трубу", "прорыв"),
    ("прорван трубопровод", "прорыв"),
    ("хлещет вода из батареи", "прорыв"),
    ("пахнет газом", "запах газа"),
    ("запах газа в подъезде", "запах газа"),
    ("газом воняет у плиты", "запах газа"),
    ("утечка газа", "запах газа"),
    ("квартира сорок два да пахнет газом", "запах газа"),   # посреди слот-филлинга
    ("прорыв в работе УК", "прорыв"),   # ложняк принят осознанно: цена — перевод жалобщика на оператора
]

NOT_EMERGENCY = [
    "не горит лампочка", "лампочка не горит", "не горит свет в квартире", "свет горит круглосуточно",
    "пожарный шкаф открыт", "учебная пожарная тревога", "пожарная сигнализация орёт",
    "дымоход нужно почистить",
    "газон не поливают", "газовая плита не включается", "газовщики не пришли",
    "затопили соседи сверху", "капает с потолка", "протечка в ванной", "течёт кран", "нет горячей воды",
    "оператор", "в прокуратуру", "добрый день", "",
]


@pytest.mark.parametrize("text,label", EMERGENCY)
def test_emergency_detected(escalation_detector, text, label):
    detection = escalation_detector.detect(text)
    assert detection.is_emergency and label in detection.emergency
    assert detection.wants_operator and detection.say
    assert detection.reason.startswith("авария: ")


@pytest.mark.parametrize("text", NOT_EMERGENCY)
def test_emergency_not_detected(escalation_detector, text):
    detection = escalation_detector.detect(text)
    assert not detection.is_emergency and detection.say is None


def test_emergency_say_phrases_by_kind(escalation_detector):
    assert "сто двенадцать" in escalation_detector.detect("у нас пожар").say
    assert "сто четыре" in escalation_detector.detect("пахнет газом").say
    assert "диспетчером" in escalation_detector.detect("прорвало трубу").say


def test_emergency_outranks_operator_and_complaint(escalation_detector):
    detection = escalation_detector.detect("пожар! дайте оператора, я в прокуратуру напишу")
    assert detection.is_emergency and detection.reason == "авария: пожар"
    assert detection.transfer == ["просит оператора"] and detection.complaint == ["прокуратура"]


def test_emergency_markers_dedup_and_first_say_wins(escalation_detector):
    detection = escalation_detector.detect("пожар, всё в дыму, прорвало трубу")
    assert detection.emergency == ["пожар", "задымление", "прорыв"]
    assert "сто двенадцать" in detection.say   # фраза — у первого сработавшего маркера


def test_emergency_say_key_must_exist(tmp_path):
    bad_yaml = tmp_path / "escalation.yaml"
    bad_yaml.write_text("emergency:\n  - {phrase: пожар., label: пожар, say: nope}\nemergency_say: {}\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        EscalationDetector.from_yaml(bad_yaml)


def test_emergency_say_section_is_not_a_marker_kind(escalation_detector):
    """emergency_say — тексты фраз, не маркеры: ключи fire/flood/gas ни на что не матчатся"""
    assert not escalation_detector.detect("fire flood gas").is_emergency